import CoreML
import CreateML
import Foundation
@testable import MultiLabelClassification
import Vision
import XCTest

class MultiLabelClassificationTests: XCTestCase {
    var trainer: MultiLabelClassificationTrainer!
    let fileManager = FileManager.default
    let authorName: String = "Test Author"
    let testModelName: String = "TestCats_MultiLabel_Run"
    let testModelVersion: String = "v1"

    let algorithm = MLImageClassifier.ModelParameters.ModelAlgorithmType.transferLearning(
        featureExtractor: .scenePrint(revision: 1),
        classifier: .logisticRegressor
    )

    var modelParameters: MLImageClassifier.ModelParameters {
        MLImageClassifier.ModelParameters(
            validation: .split(strategy: .automatic),
            maxIterations: 1,
            augmentation: [],
            algorithm: algorithm
        )
    }

    var testResourcesRootPath: String {
        var currentTestFileDir = URL(fileURLWithPath: #filePath)
        currentTestFileDir.deleteLastPathComponent()
        return currentTestFileDir.appendingPathComponent("TestResources").path
    }

    private func findAnnotationFileName() -> String? {
        do {
            let resourceURL = URL(fileURLWithPath: testResourcesRootPath)
            let items = try fileManager.contentsOfDirectory(at: resourceURL, includingPropertiesForKeys: nil)
            if let jsonFile = items.first(where: { $0.pathExtension.lowercased() == "json" }) {
                return jsonFile.lastPathComponent
            } else {
                XCTFail("テストリソースディレクトリにJSONファイルが見つかりません: \(testResourcesRootPath)")
                return nil
            }
        } catch {
            XCTFail("テストリソースディレクトリの読み取りに失敗しました: \(testResourcesRootPath) - \(error.localizedDescription)")
            return nil
        }
    }

    var resolvedAnnotationFileName: String!
    var temporaryOutputDirectoryURL: URL!
    var compiledModelURL: URL?
    var trainingResult: MultiLabelClassification.MultiLabelTrainingResult?

    enum TestError: Error {
        case trainingFailed
        case modelFileMissing
        case predictionFailed
        case setupFailed
        case resourcePathError
        case manifestFileError
    }

    override func setUp() async throws {
        try await super.setUp()

        resolvedAnnotationFileName = findAnnotationFileName()
        guard resolvedAnnotationFileName != nil else {
            throw TestError.manifestFileError
        }

        temporaryOutputDirectoryURL = fileManager.temporaryDirectory
            .appendingPathComponent("TestOutput_MultiLabel_\(UUID().uuidString)")
        try fileManager.createDirectory(
            at: temporaryOutputDirectoryURL,
            withIntermediateDirectories: true,
            attributes: nil
        )

        trainer = MultiLabelClassificationTrainer(
            resourcesDirectoryPathOverride: testResourcesRootPath,
            outputDirectoryPathOverride: temporaryOutputDirectoryURL.path,
            annotationFileNameOverride: resolvedAnnotationFileName
        )

        trainingResult = await trainer.train(
            author: authorName,
            modelName: testModelName,
            version: testModelVersion,
            modelParameters: modelParameters // Use class-level computed property
        )

        guard let result = trainingResult else {
            throw TestError.trainingFailed
        }

        let trainedModelURL = URL(fileURLWithPath: result.modelOutputPath)
        do {
            compiledModelURL = try await MLModel.compileModel(at: trainedModelURL)
        } catch {
            print("モデルのコンパイル失敗 (setUp): \(error.localizedDescription)")
            throw TestError.modelFileMissing
        }
    }

    override func tearDownWithError() throws {
        if let tempDir = temporaryOutputDirectoryURL, fileManager.fileExists(atPath: tempDir.path) {
            try? fileManager.removeItem(at: tempDir)
        }
        temporaryOutputDirectoryURL = nil

        if let compiledUrl = compiledModelURL, fileManager.fileExists(atPath: compiledUrl.path) {
            try? fileManager.removeItem(at: compiledUrl)
        }
        compiledModelURL = nil
        trainingResult = nil
        trainer = nil
        try super.tearDownWithError()
    }

    func testTrainerInitialization() throws {
        XCTAssertNotNil(trainer, "MultiLabelClassificationTrainerの初期化失敗")
        XCTAssertEqual(trainer.resourcesDirectoryPath, testResourcesRootPath, "トレーナーのリソースパスが期待値と不一致")
        XCTAssertEqual(trainer.outputDirPath, temporaryOutputDirectoryURL.path, "トレーナーの出力パスが期待値と不一致")
    }

    func testModelTrainingAndArtifactGeneration() throws {
        guard let result = trainingResult else {
            XCTFail("訓練結果がnilです (testModelTrainingAndArtifactGeneration)")
            throw TestError.trainingFailed
        }

        XCTAssertTrue(
            fileManager.fileExists(atPath: result.modelOutputPath),
            "訓練モデルファイルが期待されるパス「\(result.modelOutputPath)」に見つかりません"
        )

        let expectedClassLabels = ["black_and_white", "human_hands_detected", "mouth_open", "sphynx"].sorted()
        XCTAssertEqual(
            Set(result.classLabels.sorted()),
            Set(expectedClassLabels),
            "訓練結果のクラスラベル「\(result.classLabels.sorted())」が期待されるラベル「\(expectedClassLabels)」と一致しません"
        )

        result.saveLog(modelAuthor: authorName, modelName: testModelName, modelVersion: testModelVersion)
        let modelFileDir = URL(fileURLWithPath: result.modelOutputPath).deletingLastPathComponent()
        let expectedLogFileName = "\(testModelName)_\(testModelVersion).md"
        let expectedLogFilePath = modelFileDir.appendingPathComponent(expectedLogFileName).path
        XCTAssertTrue(fileManager.fileExists(atPath: expectedLogFilePath), "ログファイル「\(expectedLogFilePath)」が生成されていません")

        XCTAssertEqual(result.modelName, testModelName)
        XCTAssertFalse(result.trainingDataPath.isEmpty, "訓練データパス(アノテーションファイル)が空です")
        XCTAssertTrue(result.trainingDataPath.contains(resolvedAnnotationFileName), "訓練データパスにアノテーションファイル名が含まれていません")

        XCTAssertNotNil(result.averageRecallAcrossLabels, "ラベル毎の平均再現率がnilです")
        XCTAssertNotNil(result.averagePrecisionAcrossLabels, "ラベル毎の平均適合率がnilです")
        XCTAssertNotNil(result.perLabelMetricsSummary, "ラベル毎の指標サマリーがnilです")
    }

    func testModelCanPerformPrediction() throws {
        guard let finalModelURL = compiledModelURL else {
            XCTFail("コンパイル済みモデルのURLがnil (testModelCanPerformPrediction)")
            throw TestError.modelFileMissing
        }
        guard trainingResult != nil else {
            XCTFail("訓練結果がnil (testModelCanPerformPrediction)")
            throw TestError.trainingFailed
        }

        let annotationFilePath = URL(fileURLWithPath: testResourcesRootPath)
            .appendingPathComponent(resolvedAnnotationFileName)
        guard let annotationData = try? Data(contentsOf: annotationFilePath),
              let entries = try? JSONDecoder().decode(
                  [MultiLabelClassificationTrainer.ManifestEntry].self,
                  from: annotationData
              ),
              let firstEntry = entries.first
        else {
            XCTFail("テスト用アノテーションファイルの読み込み、または最初のエントリーの取得に失敗: \(annotationFilePath.path)")
            throw TestError.manifestFileError
        }

        let imageURL = URL(fileURLWithPath: testResourcesRootPath).appendingPathComponent(firstEntry.filename)
        guard fileManager.fileExists(atPath: imageURL.path) else {
            XCTFail("テスト用画像ファイルが見つかりません: \(imageURL.path)")
            throw TestError.resourcePathError
        }

        let mlModel = try MLModel(contentsOf: finalModelURL)
        let visionModel = try VNCoreMLModel(for: mlModel)
        let request = VNCoreMLRequest(model: visionModel)

        let handler = VNImageRequestHandler(url: imageURL, options: [:])
        try handler.perform([request])

        guard let observations = request.results as? [VNClassificationObservation] else {
            XCTFail("予測結果をVNClassificationObservationにキャストできませんでした。")
            throw TestError.predictionFailed
        }

        XCTAssertFalse(observations.isEmpty, "予測結果(observations)が空でした。モデルは予測を行いませんでした。")

        if !observations.isEmpty {
            let allPredictedLabelsWithConfidence = observations
                .map { "\($0.identifier) (信頼度: \(String(format: "%.2f", $0.confidence)))" }.joined(separator: ", ")
            print("ファイル「\(imageURL.lastPathComponent)」の全予測ラベル: [\(allPredictedLabelsWithConfidence)]")

            let annotationFilePath = URL(fileURLWithPath: testResourcesRootPath)
                .appendingPathComponent(resolvedAnnotationFileName)
            if let annotationData = try? Data(contentsOf: annotationFilePath),
               let entries = try? JSONDecoder().decode(
                   [MultiLabelClassificationTrainer.ManifestEntry].self,
                   from: annotationData
               ),
               let firstEntry = entries
               .first(where: {
                   URL(fileURLWithPath: testResourcesRootPath).appendingPathComponent($0.filename) == imageURL
               })
            {
                print("アノテーションファイル上の期待ラベル (参考): \(firstEntry.annotations.joined(separator: ", "))")
            } else if let firstEntry = entries.first {
                print("アノテーションファイル上の期待ラベル（フォールバック・最初の画像） (参考): \(firstEntry.annotations.joined(separator: ", "))")
            }
        }
    }
}
