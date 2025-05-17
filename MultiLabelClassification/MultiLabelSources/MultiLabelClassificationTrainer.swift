import CoreML
import CreateML
import CreateMLComponents
import CSInterface
import Foundation

public final class MultiLabelClassificationTrainer: ScreeningTrainerProtocol {
    public typealias TrainingResultType = MultiLabelTrainingResult

    public struct ManifestEntry: Decodable {
        let filename: String
        let annotations: [String]
    }

    private let resourcesDirectoryPathOverride: String?
    private let outputDirectoryPathOverride: String?
    private let annotationFileNameOverride: String?

    public var outputDirPath: String {
        if let overridePath = outputDirectoryPathOverride {
            return overridePath
        }
        var dir = URL(fileURLWithPath: #filePath)
        dir.deleteLastPathComponent()
        dir.deleteLastPathComponent()
        return dir.appendingPathComponent("OutputModels").path
    }

    public var classificationMethod: String { "MultiLabel" }

    public var resourcesDirectoryPath: String {
        if let overridePath = resourcesDirectoryPathOverride {
            return overridePath
        }
        var dir = URL(fileURLWithPath: #filePath)
        dir.deleteLastPathComponent()
        dir.deleteLastPathComponent()
        return dir.appending(path: "Resources").path
    }

    // ラベル判定の信頼度閾値
    private let predictionThreshold: Float = 0.5

    public init(
        resourcesDirectoryPathOverride: String? = nil,
        outputDirectoryPathOverride: String? = nil,
        annotationFileNameOverride: String? = nil
    ) {
        self.resourcesDirectoryPathOverride = resourcesDirectoryPathOverride
        self.outputDirectoryPathOverride = outputDirectoryPathOverride
        self.annotationFileNameOverride = annotationFileNameOverride
    }

    public func train(
        author: String,
        modelName: String,
        version: String,
        modelParameters: CreateML.MLImageClassifier.ModelParameters
    ) async -> MultiLabelTrainingResult? {
        let outputDir: URL
        do {
            outputDir = try createOutputDirectory(
                modelName: modelName,
                version: version
            )
        } catch {
            print("❌ 出力ディレクトリの作成に失敗しました – \(error.localizedDescription)")
            return nil
        }

        let resourcesDir = URL(fileURLWithPath: resourcesDirectoryPath)

        let currentAnnotationFileName: String
        if let overrideName = annotationFileNameOverride {
            currentAnnotationFileName = overrideName
            print("ℹ️ DI経由でアノテーションファイル名「\(currentAnnotationFileName)」を使用します。")
        } else {
            let fileManager = FileManager.default
            do {
                let items = try fileManager.contentsOfDirectory(
                    at: resourcesDir,
                    includingPropertiesForKeys: nil,
                    options: .skipsHiddenFiles
                )
                if let jsonFile = items.first(where: { $0.pathExtension.lowercased() == "json" }) {
                    currentAnnotationFileName = jsonFile.lastPathComponent
                    print("ℹ️ アノテーションファイル「\(currentAnnotationFileName)」を検出しました。場所: \(resourcesDirectoryPath)")
                } else {
                    print("❌ トレーニングエラー: リソースディレクトリ「\(resourcesDirectoryPath)」でJSONアノテーションファイルが見つかりませんでした。(オーバーライドも未指定)")
                    return nil
                }
            } catch {
                print(
                    "❌ トレーニングエラー: リソースディレクトリ「\(resourcesDirectoryPath)」の内容読み取り中にエラーが発生しました: \(error.localizedDescription)"
                )
                return nil
            }
        }

        let annotationFileURL = resourcesDir.appending(path: currentAnnotationFileName)

        guard FileManager.default.fileExists(atPath: annotationFileURL.path) else {
            print("❌ アノテーションファイルが見つかりません: \(annotationFileURL.path)")
            return nil
        }

        guard
            let manifestData = try? Data(contentsOf: annotationFileURL),
            let entries = try? JSONDecoder().decode([ManifestEntry].self, from: manifestData),
            !entries.isEmpty
        else {
            print("❌ アノテーションファイルの読み取りまたはデコードに失敗しました: \(annotationFileURL.path)")
            return nil
        }

        let annotatedFeatures: [AnnotatedFeature<URL, Set<String>>] = entries.compactMap { entry in
            let fileURL = resourcesDir.appending(path: entry.filename)
            return AnnotatedFeature(feature: fileURL, annotation: Set(entry.annotations))
        }

        let labels = Set(annotatedFeatures.flatMap(\.annotation)).sorted()
        guard !labels.isEmpty else {
            print("❌ アノテーションファイルでラベルが検出されませんでした。")
            return nil
        }
        print("📚 ラベル: \(labels.joined(separator: ", "))")

        let classifier = FullyConnectedNetworkMultiLabelClassifier<Float, String>(
            labels: Set(labels)
        )
        let featureExtractor = ImageFeaturePrint(revision: 1)
        let pipeline = featureExtractor.appending(classifier)

        let reader = ImageReader()
        let (trainSet, validationSet) = annotatedFeatures.randomSplit(by: 0.8)

        guard
            let trainingFeatures = try? await reader.applied(to: trainSet),
            let validationFeatures = try? await reader.applied(to: validationSet)
        else {
            print("❌ 画像リーダーの適用に失敗しました。学習データまたは検証データの処理中にエラーが発生しました。")
            return nil
        }

        print("⏳ トレーニング中 – 学習データ: \(trainingFeatures.count) / 検証データ: \(validationFeatures.count)")

        let t0 = Date()
        let fittedPipeline: ComposedTransformer<
            ImageFeaturePrint,
            FullyConnectedNetworkMultiLabelClassifier<Float, String>.Transformer
        >
        do {
            fittedPipeline = try await pipeline.fitted(to: trainingFeatures, validateOn: validationFeatures)
        } catch {
            print("❌ トレーニングに失敗しました – \(error.localizedDescription)")
            return nil
        }
        let trainingTime = Date().timeIntervalSince(t0)
        print("🎉 \(String(format: "%.2f", trainingTime)) 秒でトレーニングが完了しました")

        var perLabelMetricsResults: [String: (tp: Int, fp: Int, fn: Int)] = [:]
        for label in labels {
            perLabelMetricsResults[label] = (tp: 0, fp: 0, fn: 0)
        }

        if let validationPredictions = try? await fittedPipeline.applied(to: validationFeatures) {
            print("🧪 検証データで予測を取得しました。サンプル数: \(validationPredictions.count)")
            for i in 0 ..< validationSet.count {
                let trueAnnotations = validationSet[i].annotation
                let annotatedPrediction = validationPredictions[i]
                let actualDistribution = annotatedPrediction.feature

                var predictedLabels = Set<String>()
                for labelInDataset in labels {
                    if let score = actualDistribution[labelInDataset], score >= predictionThreshold {
                        predictedLabels.insert(labelInDataset)
                    }
                }

                for label in labels {
                    let trulyHasLabel = trueAnnotations.contains(label)
                    let predictedHasLabel = predictedLabels.contains(label)

                    if trulyHasLabel, predictedHasLabel {
                        perLabelMetricsResults[label]?.tp += 1
                    } else if !trulyHasLabel, predictedHasLabel {
                        perLabelMetricsResults[label]?.fp += 1
                    } else if trulyHasLabel, !predictedHasLabel {
                        perLabelMetricsResults[label]?.fn += 1
                    }
                }
            }
        } else {
            print("⚠️ 検証データでの予測取得に失敗しました。ラベル別指標は計算できません。")
        }

        struct PerLabelCalculatedMetrics {
            let label: String
            let recall: Double
            let precision: Double
        }
        var calculatedMetricsForDescription: [PerLabelCalculatedMetrics] = []

        for label in labels.sorted() {
            if let counts = perLabelMetricsResults[label] {
                let recall = (counts.tp + counts.fn == 0) ? 0.0 : Double(counts.tp) / Double(counts.tp + counts.fn)
                let precision = (counts.tp + counts.fp == 0) ? 0.0 : Double(counts.tp) / Double(counts.tp + counts.fp)
                calculatedMetricsForDescription.append(PerLabelCalculatedMetrics(
                    label: label,
                    recall: recall,
                    precision: precision
                ))
                print(
                    "    🔖 ラベル: \(label) - 再現率: \(String(format: "%.2f", recall * 100))%, 適合率: \(String(format: "%.2f", precision * 100))% (TP: \(counts.tp), FP: \(counts.fp), FN: \(counts.fn))"
                )
            }
        }

        var descriptionParts: [String] = []

        if !labels.isEmpty {
            descriptionParts.append("ラベル: " + labels.joined(separator: ", "))
        } else {
            descriptionParts.append("ラベル情報なし")
        }

        descriptionParts.append("最大反復回数 (指定値): \(modelParameters.maxIterations)回")
        descriptionParts.append(String(
            format: "学習データ数: %d枚, 検証データ数: %d枚",
            trainingFeatures.count,
            validationFeatures.count
        ))

        if !calculatedMetricsForDescription.isEmpty {
            descriptionParts.append("ラベル別検証指標 (しきい値: \(predictionThreshold)):")
            for metrics in calculatedMetricsForDescription {
                let metricsString = String(
                    format: "    %@: 再現率 %.1f%%, 適合率 %.1f%%",
                    metrics.label,
                    metrics.recall * 100,
                    metrics.precision * 100
                )
                descriptionParts.append(metricsString)
            }
        } else {
            descriptionParts.append("ラベル別検証指標: 計算スキップまたは失敗")
        }

        descriptionParts.append("(検証: 80/20ランダム分割)")

        let modelShortDescription = descriptionParts.joined(separator: "\n")

        let modelMetadata = ModelMetadata(
            description: modelShortDescription,
            version: version,
            author: author
        )

        let modelURL = outputDir.appendingPathComponent("\(modelName)_\(classificationMethod)_\(version).mlmodel")
        do {
            try fittedPipeline.export(to: modelURL, metadata: modelMetadata)
            print("✅ モデルを \(modelURL.path) に保存しました")
        } catch {
            print("❌ モデルのエクスポートに失敗しました – \(error.localizedDescription)")
            return nil
        }

        let finalMeanAP: Double? = nil
        let finalPerLabelSummary = calculatedMetricsForDescription
            .isEmpty ? "評価スキップまたは失敗" : "ラベル別 再現率/適合率はモデルDescription参照"
        var avgRecallDouble: Double? = nil
        var avgPrecisionDouble: Double? = nil

        if !calculatedMetricsForDescription.isEmpty {
            avgRecallDouble = calculatedMetricsForDescription.map(\.recall)
                .reduce(0, +) / Double(calculatedMetricsForDescription.count)
            avgPrecisionDouble = calculatedMetricsForDescription.map(\.precision)
                .reduce(0, +) / Double(calculatedMetricsForDescription.count)
        }

        return MultiLabelTrainingResult(
            modelName: modelName,
            trainingDurationInSeconds: trainingTime,
            modelOutputPath: modelURL.path,
            trainingDataPath: annotationFileURL.path,
            classLabels: labels,
            maxIterations: modelParameters.maxIterations,
            meanAveragePrecision: finalMeanAP,
            perLabelMetricsSummary: finalPerLabelSummary,
            averageRecallAcrossLabels: avgRecallDouble,
            averagePrecisionAcrossLabels: avgPrecisionDouble
        )
    }
}
