import CoreML
import CreateML
import CSInterface
import Foundation

public class MultiClassClassificationTrainer: ScreeningTrainerProtocol {
    public typealias TrainingResultType = MultiClassTrainingResult

    // DI 用のプロパティ
    private let resourcesDirectoryPathOverride: String?
    private let outputDirectoryPathOverride: String?

    public var outputDirPath: String {
        if let overridePath = outputDirectoryPathOverride {
            return overridePath
        }
        var dir = URL(fileURLWithPath: #filePath)
        dir.deleteLastPathComponent()
        dir.deleteLastPathComponent()
        return dir.appendingPathComponent("OutputModels").path
    }

    public var classificationMethod: String { "MultiClass" }

    public var resourcesDirectoryPath: String {
        if let overridePath = resourcesDirectoryPathOverride {
            return overridePath
        }
        var dir = URL(fileURLWithPath: #filePath)
        dir.deleteLastPathComponent()
        dir.deleteLastPathComponent()
        return dir.appendingPathComponent("Resources").path
    }

    public init(
        resourcesDirectoryPathOverride: String? = nil,
        outputDirectoryPathOverride: String? = nil
    ) {
        self.resourcesDirectoryPathOverride = resourcesDirectoryPathOverride
        self.outputDirectoryPathOverride = outputDirectoryPathOverride
    }

    public func train(
        author: String,
        modelName: String,
        version: String,
        modelParameters: CreateML.MLImageClassifier.ModelParameters
    )
        async -> MultiClassTrainingResult?
    {
        let resourcesPath = resourcesDirectoryPath
        let resourcesDir = URL(fileURLWithPath: resourcesPath)
        let trainingDataParentDir = resourcesDir

        guard FileManager.default.fileExists(atPath: trainingDataParentDir.path) else {
            print("❌ エラー: トレーニングデータ親ディレクトリが見つかりません 。 \(trainingDataParentDir.path)")
            return nil
        }

        let finalOutputDir: URL

        do {
            finalOutputDir = try createOutputDirectory(
                modelName: modelName,
                version: version
            )

            let contents = try FileManager.default.contentsOfDirectory(
                at: trainingDataParentDir,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: .skipsHiddenFiles
            )
            let allClassDirs = contents.filter { url in
                var isDirectory: ObjCBool = false
                return FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) && isDirectory
                    .boolValue
            }
            let classLabelsFromFileSystem = allClassDirs.map(\.lastPathComponent).sorted()
            print("📚 ファイルシステムから検出されたクラスラベル: \(classLabelsFromFileSystem.joined(separator: ", "))")

            // トレーニングに使用する総サンプル数を計算
            var totalImageSamples = 0
            for classDirURL in allClassDirs {
                if let files = try? FileManager.default.contentsOfDirectory(
                    at: classDirURL,
                    includingPropertiesForKeys: [.isRegularFileKey],
                    options: .skipsHiddenFiles
                ) {
                    totalImageSamples += files.filter { !$0.hasDirectoryPath }.count
                }
            }

            print("\n🚀 多クラス分類モデル [\(modelName)] のトレーニングを開始します...")
            let trainingDataSource = MLImageClassifier.DataSource.labeledDirectories(at: trainingDataParentDir)

            print("⏳ \(modelName) モデルトレーニング実行中 (最大反復: \(modelParameters.maxIterations)回)... ")
            let startTime = Date()
            let model = try MLImageClassifier(trainingData: trainingDataSource, parameters: modelParameters)
            let endTime = Date()
            let duration = endTime.timeIntervalSince(startTime)
            print("🎉 [\(modelName)] のトレーニングに成功しました！ (所要時間: \(String(format: "%.2f", duration))秒)")

            let trainingEvaluation = model.trainingMetrics
            let validationEvaluation = model.validationMetrics

            let trainingDataAccuracyPercentage = (1.0 - trainingEvaluation.classificationError) * 100
            let trainingAccuracyPercentageString = String(format: "%.2f", trainingDataAccuracyPercentage)
            print("  📊 トレーニングデータ正解率: \(trainingAccuracyPercentageString)%")

            let validationDataAccuracyPercentage = (1.0 - validationEvaluation.classificationError) * 100
            let validationAccuracyPercentageString = String(format: "%.2f", validationDataAccuracyPercentage)
            print("  📈 検証データ正解率: \(validationAccuracyPercentageString)%")

            var perClassRecallRates: [Double] = []
            var perClassPrecisionRates: [Double] = []

            let confusionMatrix = validationEvaluation.confusion
            print("📄 Confusion Matrix Description: \(confusionMatrix.description)")
            print("📄 Confusion Matrix Rows Count: \(confusionMatrix.rows.count)")
            var labelSet = Set<String>()
            for row in confusionMatrix.rows {
                if let actual = row["True Label"]?.stringValue {
                    labelSet.insert(actual)
                }
                if let predicted = row["Predicted"]?.stringValue {
                    labelSet.insert(predicted)
                }
            }
            let labelsFromConfusion = Array(labelSet).sorted()
            print("📊 混同行列から取得した評価用クラスラベル: \(labelsFromConfusion.joined(separator: ", "))")

            let finalDetectedClassLabels: [String]
            if labelsFromConfusion.isEmpty, !classLabelsFromFileSystem.isEmpty {
                print("⚠️ 混同行列からラベルが取得できなかったため、ファイルシステムのラベルを使用します。")
                finalDetectedClassLabels = classLabelsFromFileSystem
            } else {
                finalDetectedClassLabels = labelsFromConfusion
            }

            struct PerClassValidationMetrics {
                let label: String
                let recall: Double
                let precision: Double
            }
            var detailedClassMetrics: [PerClassValidationMetrics] = []

            for label in labelsFromConfusion {
                // TP (True Positive): 真のラベルが `label` で、予測も `label`
                let truePositivesCount = confusionMatrix.rows.reduce(0.0) { acc, row in
                    guard
                        row["True Label"]?.stringValue == label,
                        row["Predicted"]?.stringValue == label,
                        let count = row["Count"]?.intValue.map(Double.init)
                    else { return acc }
                    return acc + count
                }

                // FP (False Positive): 真のラベルは `label` 以外だが、予測は `label`
                var falsePositivesCount: Double = 0
                for row in confusionMatrix.rows {
                    guard
                        let actual = row["True Label"]?.stringValue,
                        let predicted = row["Predicted"]?.stringValue,
                        let count = row["Count"]?.intValue.map(Double.init),
                        actual != label, predicted == label
                    else { continue }
                    falsePositivesCount += count
                }

                // FN (False Negative): 真のラベルは `label` だが、予測は `label` 以外
                var falseNegativesCount: Double = 0
                for row in confusionMatrix.rows {
                    guard
                        let actual = row["True Label"]?.stringValue,
                        let predicted = row["Predicted"]?.stringValue,
                        let count = row["Count"]?.intValue.map(Double.init),
                        actual == label, predicted != label
                    else { continue }
                    falseNegativesCount += count
                }

                // TN (True Negative): 真のラベルも予測も `label` 以外
                var trueNegativesCount: Double = 0
                for row in confusionMatrix.rows {
                    guard
                        let actual = row["True Label"]?.stringValue,
                        let predicted = row["Predicted"]?.stringValue,
                        let count = row["Count"]?.intValue.map(Double.init),
                        actual != label, predicted != label
                    else { continue }
                    trueNegativesCount += count
                }

                let recallRate = (truePositivesCount + falseNegativesCount == 0) ? 0 : truePositivesCount /
                    (truePositivesCount + falseNegativesCount)
                let precisionRate = (truePositivesCount + falsePositivesCount == 0) ? 0 : truePositivesCount /
                    (truePositivesCount + falsePositivesCount)

                perClassRecallRates.append(recallRate)
                perClassPrecisionRates.append(precisionRate)
                print(
                    "    🔎 クラス: \(label) - 再現率: \(String(format: "%.2f", recallRate * 100))%, 適合率: \(String(format: "%.2f", precisionRate * 100))%"
                )

                detailedClassMetrics.append(PerClassValidationMetrics(
                    label: label,
                    recall: recallRate,
                    precision: precisionRate
                ))
            }

            let macroAverageRecallRate = perClassRecallRates.isEmpty ? 0 : perClassRecallRates
                .reduce(0, +) / Double(perClassRecallRates.count)
            let macroAveragePrecisionRate = perClassPrecisionRates.isEmpty ? 0 : perClassPrecisionRates
                .reduce(0, +) / Double(perClassPrecisionRates.count)

            print("    📊 検証データ マクロ平均再現率: \(String(format: "%.2f", macroAverageRecallRate * 100))%")
            print("    🎯 検証データ マクロ平均適合率: \(String(format: "%.2f", macroAveragePrecisionRate * 100))%")

            // .mlmodel のメタデータに含める shortDescription を動的に生成
            var descriptionParts: [String] = []

            // 1. クラス構成 (総サンプル数で代用)
            descriptionParts.append(String(format: "総学習サンプル数: %d枚", totalImageSamples))

            // クラスラベルリストを追加 (任意) - labelsFromConfusion を使用
            if !labelsFromConfusion.isEmpty {
                descriptionParts.append("検出クラス (検証時): " + labelsFromConfusion.joined(separator: ", "))
            }

            // 2. 最大反復回数
            descriptionParts.append("最大反復回数: \(modelParameters.maxIterations)回")

            // 3. 正解率情報 (全体)
            descriptionParts.append(String(
                format: "訓練正解率 (全体): %.1f%%, 検証正解率 (全体): %.1f%%",
                trainingDataAccuracyPercentage,
                validationDataAccuracyPercentage
            ))

            // 4. クラス別指標 (再現率・適合率)
            if !detailedClassMetrics.isEmpty {
                descriptionParts.append("クラス別検証指標:")
                for metrics in detailedClassMetrics {
                    let metricsString = String(
                        format: "    %@: 再現率 %.1f%%, 適合率 %.1f%%",
                        metrics.label,
                        metrics.recall * 100,
                        metrics.precision * 100
                    )
                    descriptionParts.append(metricsString)
                }
            }

            // 5. 検証方法
            descriptionParts.append("(検証: 自動分割)")

            let modelMetadataShortDescription = descriptionParts.joined(separator: "\n")

            let metadata = MLModelMetadata(
                author: author,
                shortDescription: modelMetadataShortDescription,
                version: version
            )

            let outputModelURL = finalOutputDir
                .appendingPathComponent("\(modelName)_\(classificationMethod)_\(version).mlmodel")

            print("  💾 [\(modelName)_\(classificationMethod)_\(version).mlmodel] を保存中: \(outputModelURL.path)")
            try model.write(to: outputModelURL, metadata: metadata)
            print("  ✅ [\(modelName)_\(classificationMethod)_\(version).mlmodel] は正常に保存されました。")

            return MultiClassTrainingResult(
                modelName: modelName,
                trainingDataAccuracy: trainingDataAccuracyPercentage / 100.0,
                validationDataAccuracy: validationDataAccuracyPercentage / 100.0,
                trainingDataErrorRate: trainingEvaluation.classificationError,
                validationDataErrorRate: validationEvaluation.classificationError,
                trainingTimeInSeconds: duration,
                modelOutputPath: outputModelURL.path,
                trainingDataPath: trainingDataParentDir.path,
                classLabels: classLabelsFromFileSystem,
                maxIterations: modelParameters.maxIterations,
                macroAverageRecall: macroAverageRecallRate,
                macroAveragePrecision: macroAveragePrecisionRate,
                detectedClassLabelsList: finalDetectedClassLabels
            )

        } catch let error as CreateML.MLCreateError {
            print("  ❌ モデル [\(modelName)] のトレーニングまたは保存エラー 。CreateMLエラー: \(error.localizedDescription)")
            return nil
        } catch {
            print("  ❌ トレーニングプロセス中に予期しないエラーが発生しました 。 \(error.localizedDescription)")
            if let nsError = error as NSError? {
                print("    詳細なエラー情報: \(nsError.userInfo)")
            }
            return nil
        }
    }
}
