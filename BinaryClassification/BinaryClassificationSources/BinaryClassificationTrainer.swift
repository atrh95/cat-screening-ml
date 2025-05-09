import CoreML
import CreateML
import Foundation
import SCSInterface

public class BinaryClassificationTrainer: ScreeningTrainerProtocol {
    public typealias TrainingResultType = BinaryTrainingResult

    public var modelName: String { "ScaryCatScreeningML_Binary" }
    public var customOutputDirPath: String { "BinaryClassification/OutputModels" }

    public var resourcesDirectoryPath: String {
        var dir = URL(fileURLWithPath: #filePath)
        dir.deleteLastPathComponent()
        dir.deleteLastPathComponent()
        return dir.appendingPathComponent("Resources").path
    }

    public init() {}

    public func train(author: String, shortDescription: String, version: String) async -> BinaryTrainingResult? {
        let resourcesPath = resourcesDirectoryPath
        let resourcesDir = URL(fileURLWithPath: resourcesPath)
        let trainingDataParentDir = resourcesDir

        // --- Output Directory Setup ---
        var projectRoot =
            URL(fileURLWithPath: #filePath) // .../BinaryClassificationSources/BinaryClassificationTrainer.swift
        projectRoot.deleteLastPathComponent() // .../BinaryClassificationSources/
        projectRoot.deleteLastPathComponent() // .../BinaryClassification/
        projectRoot.deleteLastPathComponent() // プロジェクトルートへ
        let baseOutputDir = projectRoot

        let baseTargetOutputDir: URL
        let customPath = customOutputDirPath
        if !customPath.isEmpty {
            let customURL = URL(fileURLWithPath: customPath)
            if customURL.isFileURL, customPath.hasPrefix("/") {
                baseTargetOutputDir = customURL
            } else {
                baseTargetOutputDir = baseOutputDir.appendingPathComponent(customPath)
            }
        } else {
            print("⚠️ 警告: customOutputDirPathが空です。デフォルトのOutputModelsを使用します。")
            baseTargetOutputDir = baseOutputDir.appendingPathComponent("OutputModels")
        }

        let fileManager = FileManager.default

        do {
            try fileManager.createDirectory(at: baseTargetOutputDir, withIntermediateDirectories: true, attributes: nil)
            print("📂 ベース出力ディレクトリ: \(baseTargetOutputDir.path)")
        } catch {
            print("❌ エラー: ベース出力ディレクトリの作成に失敗しました: \(baseTargetOutputDir.path) - \(error.localizedDescription)")
            return nil
        }

        var resultCounter = 1
        var finalOutputDir: URL
        let resultDirPrefix = "Binary_Result_"

        repeat {
            let resultDirName = "\(resultDirPrefix)\(resultCounter)"
            finalOutputDir = baseTargetOutputDir.appendingPathComponent(resultDirName)
            resultCounter += 1
        } while fileManager.fileExists(atPath: finalOutputDir.path)

        do {
            try fileManager.createDirectory(at: finalOutputDir, withIntermediateDirectories: false, attributes: nil)
            print("💾 結果保存ディレクトリ: \(finalOutputDir.path)")
        } catch {
            print("❌ エラー: 結果保存ディレクトリの作成に失敗しました: \(finalOutputDir.path) - \(error.localizedDescription)")
            return nil
        }
        // --- End Output Directory Setup ---

        print("🚀 \(modelName)のトレーニングを開始します...")

        return await executeTrainingCore(
            trainingDataParentDir: trainingDataParentDir,
            outputDir: finalOutputDir,
            author: author,
            shortDescription: shortDescription,
            version: version
        )
    }

    private func executeTrainingCore(
        trainingDataParentDir: URL,
        outputDir: URL,
        author: String,
        shortDescription: String,
        version: String
    ) async -> BinaryTrainingResult? {
        guard FileManager.default.fileExists(atPath: trainingDataParentDir.path) else {
            print("❌ エラー: \(modelName)のトレーニングデータ親ディレクトリが見つかりません: \(trainingDataParentDir.path)")
            return nil
        }

        do {
            _ = try FileManager.default.contentsOfDirectory(atPath: trainingDataParentDir.path)
        } catch {
            print("⚠️ 警告: トレーニングデータ親ディレクトリの内容をリストできませんでした: \(error)")
        }

        let trainingDataSource = MLImageClassifier.DataSource.labeledDirectories(at: trainingDataParentDir)

        do {
            // --- Training and Evaluation ---
            let startTime = Date()

            let model = try MLImageClassifier(trainingData: trainingDataSource)

            let endTime = Date()
            let trainingDurationInSeconds = endTime.timeIntervalSince(startTime)

            print("🎉 \(modelName)のトレーニングに成功しました！ (所要時間: \(String(format: "%.2f", trainingDurationInSeconds))秒)")

            let trainingDataMisclassificationRate = model.trainingMetrics.classificationError
            let trainingDataAccuracyPercentage = (1.0 - trainingDataMisclassificationRate) * 100
            let trainingErrorStr = String(format: "%.2f", trainingDataMisclassificationRate * 100)
            let trainingAccStr = String(format: "%.2f", trainingDataAccuracyPercentage)
            print("  📊 トレーニングデータ正解率: \(trainingAccStr)%")

            let validationDataMisclassificationRate = model.validationMetrics.classificationError
            let validationDataAccuracyPercentage = (1.0 - validationDataMisclassificationRate) * 100
            let validationErrorStr = String(format: "%.2f", validationDataMisclassificationRate * 100)
            let validationAccStr = String(format: "%.2f", validationDataAccuracyPercentage)
            print("  📈 検証データ正解率: \(validationAccStr)%")
            // --- End Training and Evaluation ---

            let metadata = MLModelMetadata(
                author: author,
                shortDescription: shortDescription,
                version: version
            )

            let fileManager = FileManager.default
            let outputModelURL = outputDir.appendingPathComponent("\(modelName)_\(version).mlmodel")

            print("💾 \(modelName) (\(version)) を保存中: \(outputModelURL.path)")
            try model.write(to: outputModelURL, metadata: metadata)
            print("✅ \(modelName) (\(version)) は正常に保存されました。")

            // Get Class Labels
            let classLabels: [String]
            do {
                let contents = try FileManager.default.contentsOfDirectory(atPath: trainingDataParentDir.path)
                // 隠しファイルを除外し、ディレクトリのみをフィルタリング & ソート
                classLabels = contents.filter { item in
                    var isDirectory: ObjCBool = false
                    let fullPath = trainingDataParentDir.appendingPathComponent(item).path
                    return !item.hasPrefix(".") && FileManager.default
                        .fileExists(atPath: fullPath, isDirectory: &isDirectory) && isDirectory.boolValue
                }.sorted()
            } catch {
                print("⚠️ クラスラベルの取得に失敗しました: \(trainingDataParentDir.path) - \(error.localizedDescription)")
                classLabels = []
            }

            return BinaryTrainingResult(
                modelName: modelName,
                trainingDataAccuracyPercentage: trainingDataAccuracyPercentage,
                validationDataAccuracyPercentage: validationDataAccuracyPercentage,
                trainingDataMisclassificationRate: trainingDataMisclassificationRate,
                validationDataMisclassificationRate: validationDataMisclassificationRate,
                trainingDurationInSeconds: trainingDurationInSeconds,
                trainedModelFilePath: outputModelURL.path,
                sourceTrainingDataDirectoryPath: trainingDataParentDir.path,
                detectedClassLabelsList: classLabels
            )

        } catch let error as CreateML.MLCreateError {
            switch error {
                case .io:
                    print("❌ モデル\(modelName)の保存エラー: I/Oエラー - \(error.localizedDescription)")
                default:
                    print("❌ モデル\(self.modelName)のトレーニングエラー: 未知の Create MLエラー - \(error.localizedDescription)")
                    print("  詳細なCreate MLエラー: \(error)")
            }
            return nil
        } catch {
            print("❌ \(modelName)のトレーニングまたは保存中に予期しないエラーが発生しました: \(error.localizedDescription)")
            return nil
        }
    }
}
