import CoreML
import CreateML
import Foundation
@testable import OvOClassification
import Vision
import XCTest

final class OvOClassificationTests: XCTestCase {
    var trainer: OvOClassificationTrainer!
    let fileManager = FileManager.default
    let authorName: String = "Test Author"
    let testModelName: String = "TestCats_OvO_Run"
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

    var temporaryOutputDirectoryURL: URL!
    var compiledModelURL: URL?
    var trainingResult: OvOClassification.OvOTrainingResult?

    enum TestError: Error {
        case trainingFailed
        case modelFileMissing
        case predictionFailed
        case setupFailed
        case resourcePathError
    }

    override func setUp() async throws {
        try await super.setUp()

        temporaryOutputDirectoryURL = fileManager.temporaryDirectory
            .appendingPathComponent("TestOutput_OvO_\(UUID().uuidString)")
        try fileManager.createDirectory(
            at: temporaryOutputDirectoryURL,
            withIntermediateDirectories: true,
            attributes: nil
        )

        trainer = OvOClassificationTrainer(
            resourcesDirectoryPathOverride: testResourcesRootPath,
            outputDirectoryPathOverride: temporaryOutputDirectoryURL.path
        )

        trainingResult = await trainer.train(
            author: authorName,
            modelName: testModelName,
            version: testModelVersion,
            modelParameters: modelParameters
        )

        guard trainingResult != nil else {
            throw TestError.trainingFailed
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
        XCTAssertNotNil(trainer, "OvOClassificationTrainerの初期化失敗")
        XCTAssertEqual(trainer.resourcesDirectoryPath, testResourcesRootPath, "トレーナーのリソースパスが期待値と不一致")
        XCTAssertEqual(trainer.outputDirPath, temporaryOutputDirectoryURL.path, "トレーナーの出力パスが期待値と不一致")
    }

    func testModelTrainingAndArtifactGeneration() throws {
        guard let result = trainingResult else {
            XCTFail("訓練結果がnilです (OvO testModelTrainingAndArtifactGeneration)")
            throw TestError.trainingFailed
        }

        let modelOutputDir = URL(fileURLWithPath: result.modelOutputPath)
        XCTAssertTrue(
            fileManager.fileExists(atPath: modelOutputDir.path),
            "訓練モデル出力ディレクトリが期待されるパス「\(modelOutputDir.path)」に見つかりません"
        )

        let contents = try fileManager.contentsOfDirectory(
            at: modelOutputDir,
            includingPropertiesForKeys: nil,
            options: []
        )
        let mlModelFiles = contents.filter { $0.pathExtension == "mlmodel" }
        XCTAssertFalse(mlModelFiles.isEmpty, "訓練された .mlmodel ファイルが出力ディレクトリ「\(modelOutputDir.path)」に見つかりません")

        result.saveLog(modelAuthor: authorName, modelName: testModelName, modelVersion: testModelVersion)
        let actualLogFileName = "OvO_Run_Report_\(testModelVersion).md"
        let expectedLogFilePath = modelOutputDir.appendingPathComponent(actualLogFileName).path
        XCTAssertTrue(fileManager.fileExists(atPath: expectedLogFilePath), "ログファイル「\(expectedLogFilePath)」が生成されていません")
    }

    func testModelCanPerformPrediction() async throws {
        guard let result = trainingResult else {
            XCTFail("訓練結果がnil (OvO testModelCanPerformPrediction)")
            throw TestError.trainingFailed
        }

        let modelOutputDir = URL(fileURLWithPath: result.modelOutputPath)
        let contents = try fileManager.contentsOfDirectory(
            at: modelOutputDir,
            includingPropertiesForKeys: nil,
            options: []
        )
        guard let firstMlModelURL = contents.first(where: { $0.pathExtension == "mlmodel" }) else {
            XCTFail("出力ディレクトリ「\(modelOutputDir.path)」にコンパイル可能な .mlmodel ファイルが見つかりません")
            throw TestError.modelFileMissing
        }

        print("Attempting to compile model: \(firstMlModelURL.path)")
        let specificCompiledModelURL: URL
        do {
            specificCompiledModelURL = try await MLModel.compileModel(at: firstMlModelURL)
            compiledModelURL = specificCompiledModelURL
        } catch {
            XCTFail("選択されたモデルのコンパイル失敗 (\(firstMlModelURL.path)): \(error.localizedDescription)")
            throw TestError.modelFileMissing
        }

        let imageURL: URL
        do {
            imageURL =
                try getRandomImageURLFromTestResources(inBaseDirectory: URL(fileURLWithPath: testResourcesRootPath))
        } catch {
            XCTFail("テストリソースからのランダム画像取得失敗。エラー: \(error.localizedDescription)")
            throw error
        }

        print("Test image for prediction: \(imageURL.path)")

        guard fileManager.fileExists(atPath: imageURL.path) else {
            XCTFail("OvOテスト用画像ファイルが見つかりません: \(imageURL.path)")
            throw TestError.resourcePathError
        }

        let mlModel = try MLModel(contentsOf: specificCompiledModelURL)
        let visionModel = try VNCoreMLModel(for: mlModel)
        let request = VNCoreMLRequest(model: visionModel) { request, error in
            if let error {
                XCTFail("VNCoreMLRequest failed: \(error.localizedDescription)")
                return
            }
            guard let observations = request.results as? [VNClassificationObservation] else {
                XCTFail("予測結果をVNClassificationObservationにキャストできませんでした。")
                return
            }

            XCTAssertFalse(observations.isEmpty, "予測結果(observations)が空でした。モデルは予測を行いませんでした。")

            if let topResult = observations.first {
                print(
                    "OvO Top prediction for \(imageURL.lastPathComponent) using \(firstMlModelURL.lastPathComponent): \(topResult.identifier) with confidence \(topResult.confidence)"
                )
            } else {
                print(
                    "OvO prediction for \(imageURL.lastPathComponent) using \(firstMlModelURL.lastPathComponent): No observations found, though this should have been caught by XCTAssertFalse."
                )
            }
        }

        let handler = VNImageRequestHandler(url: imageURL, options: [:])
        try handler.perform([request])
    }

    private func getRandomImageURLFromTestResources(inBaseDirectory baseDirectoryURL: URL) throws -> URL {
        let allLabelDirectories = try fileManager.contentsOfDirectory(
            at: baseDirectoryURL,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: .skipsHiddenFiles
        ).filter { url in
            var isDirectory: ObjCBool = false
            fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory)
            return isDirectory.boolValue && !url.lastPathComponent.hasPrefix(".")
        }

        guard !allLabelDirectories.isEmpty else {
            let message = "テストリソースディレクトリ「\(baseDirectoryURL.path)」内にラベルサブディレクトリが見つかりません。"
            XCTFail(message)
            throw TestError.resourcePathError
        }

        guard let randomLabelDirURL = allLabelDirectories.randomElement() else {
            let message = "ランダムなラベルディレクトリの選択に失敗しました。"
            XCTFail(message)
            throw TestError.resourcePathError
        }

        print("Randomly selected label directory for test image: \(randomLabelDirURL.path)")

        let imageFiles = try fileManager.contentsOfDirectory(
            at: randomLabelDirURL,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants]
        ).filter { url in
            let pathExtension = url.pathExtension.lowercased()
            return ["jpg", "jpeg", "png"].contains(pathExtension) && !url.lastPathComponent.hasPrefix(".")
        }

        guard let randomImageURL = imageFiles.randomElement() else {
            let message = "ラベルディレクトリ「\(randomLabelDirURL.path)」内に画像ファイルが見つかりません。"
            XCTFail(message)
            throw TestError.resourcePathError
        }

        return randomImageURL
    }
}
