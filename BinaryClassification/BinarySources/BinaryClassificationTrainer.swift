import CoreML
import CreateML
import CSInterface
import Foundation

public class BinaryClassificationTrainer: ScreeningTrainerProtocol {
    public typealias TrainingResultType = BinaryTrainingResult

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

    public var classificationMethod: String { "Binary" }

    public var resourcesDirectoryPath: String {
        if let overridePath = resourcesDirectoryPathOverride {
            return overridePath
        }
        var dir = URL(fileURLWithPath: #filePath)
        dir.deleteLastPathComponent()
        dir.deleteLastPathComponent()
        return dir.appendingPathComponent("Resources").path
    }

    public init(resourcesDirectoryPathOverride: String? = nil, outputDirectoryPathOverride: String? = nil) {
        self.resourcesDirectoryPathOverride = resourcesDirectoryPathOverride
        self.outputDirectoryPathOverride = outputDirectoryPathOverride
    }

    public func train(
        author: String,
        modelName: String,
        version: String,
        modelParameters: CreateML.MLImageClassifier.ModelParameters
    ) async -> BinaryTrainingResult? {
        let resourcesPath = resourcesDirectoryPath
        let resourcesDirURL = URL(fileURLWithPath: resourcesPath)

        // 出力ディレクトリ設定
        let outputDirectoryURL: URL
        do {
            outputDirectoryURL = try createOutputDirectory(
                modelName: modelName,
                version: version
            )
        } catch {
            print("❌ エラー: 出力ディレクトリ設定に失敗 \(error.localizedDescription)")
            return nil
        }

        print("🚀 \(modelName) トレーニング開始...")

        // 主要トレーニング処理実行
        return await executeTrainingCore(
            modelName: modelName,
            trainingDataParentDirURL: resourcesDirURL,
            outputDirURL: outputDirectoryURL,
            author: author,
            version: version,
            modelParameters: modelParameters
        )
    }

    /// 主要なトレーニング処理
    private func executeTrainingCore(
        modelName: String,
        trainingDataParentDirURL: URL,
        outputDirURL: URL,
        author: String,
        version: String,
        modelParameters: CreateML.MLImageClassifier.ModelParameters
    ) async -> BinaryTrainingResult? {
        // トレーニングデータ親ディレクトリ存在確認
        guard FileManager.default.fileExists(atPath: trainingDataParentDirURL.path) else {
            print("❌ エラー: \(modelName) トレーニングデータ親ディレクトリが見つかりません: \(trainingDataParentDirURL.path)")
            return nil
        }

        // デバッグ用: トレーニングデータ親ディレクトリ内容表示試行
        do {
            _ = try FileManager.default.contentsOfDirectory(atPath: trainingDataParentDirURL.path)
        } catch {
            print("⚠️ 警告: トレーニングデータ親ディレクトリ内容表示失敗: \(error.localizedDescription)")
            // 処理続行
        }

        // サブディレクトリをクラスラベルとしてデータソース作成
        let trainingDataSource = MLImageClassifier.DataSource.labeledDirectories(at: trainingDataParentDirURL)

        do {
            let trainingStartTime = Date()

            // モデルパラメータは引数から渡されるものを使用

            print("⏳ \(modelName) モデルトレーニング実行中 (最大反復: \(modelParameters.maxIterations)回)... ")
            let imageClassifier = try MLImageClassifier(
                trainingData: trainingDataSource,
                parameters: modelParameters // Use injected modelParameters
            )
            print("✅ \(modelName) モデルトレーニング完了")

            let trainingEndTime = Date()
            let trainingDurationSeconds = trainingEndTime.timeIntervalSince(trainingStartTime)

            print("🎉 \(modelName) トレーニング成功 (所要時間: \(String(format: "%.2f", trainingDurationSeconds))秒)")

            // 評価指標
            let trainingMetrics = imageClassifier.trainingMetrics
            let validationMetrics = imageClassifier.validationMetrics

            let trainingAccuracyPercentage = (1.0 - trainingMetrics.classificationError) * 100.0
            print("  📊 トレーニングデータ正解率: \(String(format: "%.2f", trainingAccuracyPercentage))%")

            let validationAccuracyPercentage = (1.0 - validationMetrics.classificationError) * 100.0
            print("  📈 検証データ正解率: \(String(format: "%.2f", validationAccuracyPercentage))%")

            var recallRate = 0.0
            var precisionRate = 0.0

            let confusionMatrix: MLDataTable

            confusionMatrix = validationMetrics.confusion

            // MLDataTableの行構成: actualLabel | predictedLabel | count
            var labelSet = Set<String>()
            var rowCount = 0
            for row in confusionMatrix.rows {
                rowCount += 1
                // print("デバッグ: 混同行列の処理中の行: \(row)")
                if let actual = row["True Label"]?.stringValue {
                    labelSet.insert(actual)
                }
                if let predicted = row["Predicted"]?.stringValue {
                    labelSet.insert(predicted)
                }
            }
            let classLabelsFromConfusion = Array(labelSet).sorted()

            // 二値分類の場合、再現率と適合率を計算
            if classLabelsFromConfusion.count == 2 {
                // classLabelsFromConfusion はソート済み想定 (例: ["Negative", "Positive"])
                // 2番目のラベルを陽性クラスとする
                let negativeLabel = classLabelsFromConfusion[0]
                let positiveLabel = classLabelsFromConfusion[1]

                var truePositives = 0
                var falsePositives = 0
                var falseNegatives = 0

                for row in confusionMatrix.rows {
                    guard
                        let actual = row["True Label"]?.stringValue,
                        let predicted = row["Predicted"]?.stringValue,
                        let cnt = row["Count"]?.intValue
                    else { continue }

                    if actual == positiveLabel, predicted == positiveLabel {
                        truePositives += cnt
                    } else if actual == negativeLabel, predicted == positiveLabel {
                        falsePositives += cnt
                    } else if actual == positiveLabel, predicted == negativeLabel {
                        falseNegatives += cnt
                    }
                }

                if (truePositives + falseNegatives) > 0 {
                    recallRate = Double(truePositives) / Double(truePositives + falseNegatives)
                }
                if (truePositives + falsePositives) > 0 {
                    precisionRate = Double(truePositives) / Double(truePositives + falsePositives)
                }
                print("    🔍 検証データ 再現率 (陽性クラス: \(positiveLabel)): \(String(format: "%.2f", recallRate * 100))%")
                print("    🎯 検証データ 適合率 (陽性クラス: \(positiveLabel)): \(String(format: "%.2f", precisionRate * 100))%")
            } else {
                print("    ⚠️ 再現率・適合率は二値分類の場合のみ計算 (現在クラス数: \(classLabelsFromConfusion.count))")
            }

            // 各クラスの画像枚数とクラス名リスト取得
            var imageCountsPerClass: [String: Int] = [:]
            var classNamesFromDataDirs: [String] = []

            let classLabelDirURLs = (
                try? FileManager.default.contentsOfDirectory(
                    at: trainingDataParentDirURL,
                    includingPropertiesForKeys: [.isDirectoryKey],
                    options: .skipsHiddenFiles
                )
                .filter { url in
                    var isDir: ObjCBool = false
                    FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir)
                    return isDir.boolValue && !url.lastPathComponent.hasPrefix(".")
                }
                .sorted(by: { $0.lastPathComponent < $1.lastPathComponent })
            ) ?? []

            for labelDirURL in classLabelDirURLs {
                let className = labelDirURL.lastPathComponent
                classNamesFromDataDirs.append(className)
                if let files = try? FileManager.default.contentsOfDirectory(
                    at: labelDirURL,
                    includingPropertiesForKeys: [.isRegularFileKey],
                    options: .skipsHiddenFiles
                ) {
                    imageCountsPerClass[className] = files.filter { !$0.hasDirectoryPath }.count
                } else {
                    imageCountsPerClass[className] = 0
                }
            }

            // .mlmodel メタデータ用 shortDescription 生成
            var descriptionParts: [String] = []

            // 1. クラス構成情報
            if !classNamesFromDataDirs.isEmpty {
                let classCountsStrings = classNamesFromDataDirs.map { className in
                    let count = imageCountsPerClass[className] ?? 0
                    return "\(className): \(count)枚"
                }
                descriptionParts.append("クラス構成: " + classCountsStrings.joined(separator: "; "))
            } else {
                descriptionParts.append("クラス構成情報なし")
            }

            // 2. 最大反復回数
            descriptionParts.append("最大反復回数: \(modelParameters.maxIterations)回")

            // 3. 正解率情報
            descriptionParts.append(String(
                format: "訓練正解率: %.1f%%, 検証正解率: %.1f%%",
                trainingAccuracyPercentage,
                validationAccuracyPercentage
            ))

            // 4. 陽性クラス情報 (再現率・適合率)
            if classLabelsFromConfusion.count == 2 {
                // classLabelsFromConfusion はソート済み想定 (例: ["Not Scary", "Scary"])
                // 2番目のラベルを陽性クラスとする
                let positiveLabelForDesc = classLabelsFromConfusion[1]
                descriptionParts.append(String(
                    format: "陽性クラス: %@, 再現率: %.1f%%, 適合率: %.1f%%",
                    positiveLabelForDesc,
                    recallRate * 100,
                    precisionRate * 100
                ))
            } else if !classLabelsFromConfusion.isEmpty {
                descriptionParts.append("(詳細な分類指標は二値分類のみ)")
            }

            // 5. 検証方法
            descriptionParts.append("(検証: 自動分割)")

            let modelMetadataShortDescription = descriptionParts.joined(separator: "\n")

            let modelMetadata = MLModelMetadata(
                author: author,
                shortDescription: modelMetadataShortDescription,
                version: version
            )

            let outputModelFileURL = outputDirURL
                .appendingPathComponent("\(modelName)_\(classificationMethod)_\(version).mlmodel")

            print("💾 \(modelName) (\(version)) 保存中: \(outputModelFileURL.path)")
            try imageClassifier.write(to: outputModelFileURL, metadata: modelMetadata)
            print("✅ \(modelName) (\(version)) 保存完了")

            // 結果レポート用にデータディレクトリ由来のクラスラベルリストを採用
            let detectedClassLabels = classNamesFromDataDirs

            // トレーニング結果返却
            return BinaryTrainingResult(
                modelName: modelName,
                trainingDataAccuracyPercentage: trainingAccuracyPercentage,
                validationDataAccuracyPercentage: validationAccuracyPercentage,
                trainingDataMisclassificationRate: trainingMetrics.classificationError,
                validationDataMisclassificationRate: validationMetrics.classificationError,
                trainingDurationInSeconds: trainingDurationSeconds,
                trainedModelFilePath: outputModelFileURL.path,
                sourceTrainingDataDirectoryPath: trainingDataParentDirURL.path,
                detectedClassLabelsList: detectedClassLabels,
                maxIterations: modelParameters.maxIterations // Use maxIterations from modelParameters
            )

        } catch let createMLError as CreateML.MLCreateError {
            switch createMLError {
                case .io:
                    print("❌ \(modelName) 保存エラー (I/O): \(createMLError.localizedDescription)")
                default:
                    print("❌ \(modelName) トレーニングエラー (CreateML): \(createMLError.localizedDescription)")
                    print("  詳細情報: \(createMLError)")
            }
            return nil
        } catch {
            print("❌ \(modelName) トレーニング/保存中に予期しないエラー: \(error.localizedDescription)")
            return nil
        }
    }
}
