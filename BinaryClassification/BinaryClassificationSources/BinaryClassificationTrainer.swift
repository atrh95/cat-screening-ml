import CoreML
import CreateML
import CSInterface
import Foundation

public class BinaryClassificationTrainer: ScreeningTrainerProtocol {
    public typealias TrainingResultType = BinaryTrainingResult

    public var modelName: String { "ScaryCatScreeningML_Binary" }
    public var customOutputDirPath: String { "BinaryClassification/OutputModels" }

    public var outputRunNamePrefix: String { "Binary" }

    public var resourcesDirectoryPath: String {
        var dir = URL(fileURLWithPath: #filePath)
        dir.deleteLastPathComponent()
        dir.deleteLastPathComponent()
        return dir.appendingPathComponent("Resources").path
    }

    public init() {}

    public func train(
        author: String,
        shortDescription: String,
        version: String,
        maxIterations: Int
    ) async -> BinaryTrainingResult? {
        let resourcesPath = resourcesDirectoryPath
        let resourcesDir = URL(fileURLWithPath: resourcesPath)
        let trainingDataParentDir = resourcesDir

        // --- Output Directory Setup ---
        let finalOutputDir: URL
        do {
            finalOutputDir = try setupVersionedRunOutputDirectory(
                version: version,
                trainerFilePath: #filePath
            )
        } catch {
            print("❌ エラー: 出力ディレクトリの設定に失敗しました - \(error.localizedDescription)")
            return nil
        }
        // --- End Output Directory Setup ---

        print("🚀 \(modelName)のトレーニングを開始します...")

        return await executeTrainingCore(
            trainingDataParentDir: trainingDataParentDir,
            outputDir: finalOutputDir,
            author: author,
            shortDescription: shortDescription,
            version: version,
            maxIterations: maxIterations
        )
    }

    private func executeTrainingCore(
        trainingDataParentDir: URL,
        outputDir: URL,
        author: String,
        shortDescription: String,
        version: String,
        maxIterations: Int
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

            var parameters = MLImageClassifier.ModelParameters()
            parameters.featureExtractor = .scenePrint(revision: 1)
            parameters.maxIterations = maxIterations
            parameters.validation = .split(strategy: .automatic)

            let model = try MLImageClassifier(trainingData: trainingDataSource, parameters: parameters)

            let endTime = Date()
            let trainingDurationInSeconds = endTime.timeIntervalSince(startTime)

            print("🎉 \(modelName)のトレーニングに成功しました！ (所要時間: \(String(format: "%.2f", trainingDurationInSeconds))秒)")

            let trainingMetrics = model.trainingMetrics
            let validationMetrics = model.validationMetrics

            let trainingDataAccuracyPercentage = (1.0 - trainingMetrics.classificationError) * 100.0
            let trainingAccStr = String(format: "%.2f", trainingDataAccuracyPercentage)
            print("  📊 トレーニングデータ正解率: \(trainingAccStr)%")

            let validationDataAccuracyPercentage = (1.0 - validationMetrics.classificationError) * 100.0
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
                trainingDataMisclassificationRate: trainingMetrics.classificationError,
                validationDataMisclassificationRate: validationMetrics.classificationError,
                trainingDurationInSeconds: trainingDurationInSeconds,
                trainedModelFilePath: outputModelURL.path,
                sourceTrainingDataDirectoryPath: trainingDataParentDir.path,
                detectedClassLabelsList: classLabels,
                maxIterations: maxIterations
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
