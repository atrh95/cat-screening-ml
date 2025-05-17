import CoreML
import CreateML
import Foundation
@testable import MultiClassClassification
import Vision
import XCTest

class MultiClassClassificationTests: XCTestCase {
    var trainer: MultiClassClassificationTrainer!
    let fileManager = FileManager.default
    let authorName: String = "Test Author"
    let testModelName: String = "TestCats_Multi_Run"
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

    var temporaryOutputDirectoryURL: URL!
    var compiledModelURL: URL?
    var trainingResult: MultiClassClassification.MultiClassTrainingResult?

    var testResourcesRootPath: String {
        var currentTestFileDir = URL(fileURLWithPath: #filePath)
        currentTestFileDir.deleteLastPathComponent()
        return currentTestFileDir.appendingPathComponent("TestResources").path
    }

    override func setUp() async throws {
        try await super.setUp()

        temporaryOutputDirectoryURL = fileManager.temporaryDirectory
            .appendingPathComponent("TestOutput_\(UUID().uuidString)")
        try fileManager.createDirectory(
            at: temporaryOutputDirectoryURL,
            withIntermediateDirectories: true,
            attributes: nil
        )

        trainer = MultiClassClassificationTrainer(
            resourcesDirectoryPathOverride: testResourcesRootPath,
            outputDirectoryPathOverride: temporaryOutputDirectoryURL.path
        )

        try fileManager.createDirectory(
            at: URL(fileURLWithPath: testResourcesRootPath),
            withIntermediateDirectories: true, attributes: nil
        )

        trainingResult = await trainer.train(
            author: authorName,
            modelName: testModelName,
            version: testModelVersion,
            modelParameters: modelParameters
        )

        guard let result = trainingResult else {
            throw TestError.trainingFailed
        }

        let trainedModelURL = URL(fileURLWithPath: result.modelOutputPath)
        do {
            compiledModelURL = try await MLModel.compileModel(at: trainedModelURL)
        } catch {
            throw error
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

    func testTrainerInitialization() {
        XCTAssertNotNil(trainer, "MultiClassClassificationTrainerの初期化失敗")

        XCTAssertEqual(trainer.resourcesDirectoryPath, testResourcesRootPath, "トレーナーのリソースパスが期待値と不一致")
        XCTAssertEqual(trainer.outputDirPath, temporaryOutputDirectoryURL.path, "トレーナーの出力パスが期待値と不一致")
    }

    enum TestError: Error {
        case testResourceMissing
        case trainingFailed
        case modelFileMissing
        case predictionFailed
        case setupFailed
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

        // Dynamically get expected class labels from the TestResources subdirectories
        let testResourcesURL = URL(fileURLWithPath: testResourcesRootPath)
        var expectedClassLabels: [String] = []
        do {
            let subdirectories = try fileManager.contentsOfDirectory(
                at: testResourcesURL,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants]
            )
            expectedClassLabels = subdirectories.filter { url in
                var isDirectory: ObjCBool = false
                return fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory) && isDirectory.boolValue
            }.map(\.lastPathComponent).sorted()
        } catch {
            XCTFail("テストリソースのサブディレクトリからのクラスラベルの取得に失敗しました: \(error.localizedDescription)")
            throw TestError.setupFailed
        }

        guard !expectedClassLabels.isEmpty else {
            XCTFail("テストリソースディレクトリから期待されるクラスラベルが見つかりませんでした。パス: \(testResourcesRootPath)")
            throw TestError.setupFailed
        }

        XCTAssertEqual(
            Set(result.detectedClassLabelsList.sorted()),
            Set(expectedClassLabels),
            "検出されたクラスラベル「\(result.detectedClassLabelsList.sorted())」が期待されるラベル「\(expectedClassLabels)」と一致しません"
        )

        result.saveLog(modelAuthor: authorName, modelName: testModelName, modelVersion: testModelVersion)
        let modelFileDir = URL(fileURLWithPath: result.modelOutputPath).deletingLastPathComponent()

        let expectedLogFileName = "\(testModelName)_\(testModelVersion).md"
        let expectedLogFilePath = modelFileDir.appendingPathComponent(expectedLogFileName).path
        XCTAssertTrue(
            fileManager.fileExists(atPath: expectedLogFilePath),
            "ログファイルが期待されるパス「\(expectedLogFilePath)」に生成されていません"
        )

        XCTAssertEqual(
            result.modelName,
            testModelName,
            "訓練結果のmodelName「\(result.modelName)」が期待値「\(testModelName)」と一致しません"
        )

        do {
            let logContents = try String(contentsOfFile: expectedLogFilePath, encoding: .utf8)
            XCTAssertFalse(logContents.isEmpty, "ログファイルが空です: \(expectedLogFilePath)")
        } catch {
            XCTFail("ログファイルの読み込みに失敗しました: \(expectedLogFilePath), エラー: \(error.localizedDescription)")
        }

        XCTAssertTrue(
            result.modelOutputPath.contains(testModelName),
            "モデルファイルのパスにモデル名「\(testModelName)」が含まれていません"
        )
        XCTAssertTrue(
            result.modelOutputPath.contains(testModelVersion),
            "モデルファイルのパスにバージョン「\(testModelVersion)」が含まれていません"
        )
        XCTAssertTrue(
            result.modelOutputPath.contains(trainer.classificationMethod),
            "モデルファイルのパスに分類手法「\(trainer.classificationMethod)」が含まれていません"
        )
    }

    func testModelCanPerformPrediction() throws {
        guard let finalModelURL = compiledModelURL else {
            XCTFail("コンパイル済みモデルのURLがnilです (testModelCanPerformPrediction)")
            throw TestError.modelFileMissing
        }
        guard let result = trainingResult else {
            XCTFail("訓練結果がnilです (testModelCanPerformPrediction)")
            throw TestError.setupFailed
        }
        let predictionTestResourcePath = testResourcesRootPath

        let mlModel = try MLModel(contentsOf: finalModelURL)
        let vnCoreMLModel = try VNCoreMLModel(for: mlModel)
        let predictionRequest = VNCoreMLRequest(model: vnCoreMLModel)

        let baseResourceURL = URL(fileURLWithPath: predictionTestResourcePath)

        let classLabelsForPredictionTest = result.detectedClassLabelsList.sorted()

        guard !classLabelsForPredictionTest.isEmpty else {
            XCTFail("予測テストの実行には、訓練結果に最低1つのクラスラベルが必要です。検出されたラベルはありません。")
            throw TestError.setupFailed
        }

        let classLabelToTest = classLabelsForPredictionTest[0]
        let imageURL: URL
        do {
            imageURL = try getRandomImageURL(
                forClassLabel: classLabelToTest,
                inBaseDirectory: baseResourceURL
            )
        } catch {
            XCTFail("クラス「\(classLabelToTest)」のサブディレクトリからのランダム画像取得に失敗しました。エラー: \(error.localizedDescription)")
            throw error
        }

        let imageHandler = VNImageRequestHandler(url: imageURL, options: [:])
        try imageHandler.perform([predictionRequest])
        guard let observations = predictionRequest.results as? [VNClassificationObservation],
              let topResult = observations.first
        else {
            XCTFail("クラス「\(classLabelToTest)」の画像に対する分類結果オブジェクトの取得に失敗しました。")
            throw TestError.predictionFailed
        }
        XCTAssertNotNil(topResult.identifier, "クラス「\(classLabelToTest)」の画像に対する予測結果からクラスラベルが取得できませんでした。")
    }

    private func getRandomImageURL(
        forClassLabel classLabel: String,
        inBaseDirectory baseDirectoryURL: URL
    ) throws -> URL {
        let subdirectoryURL = baseDirectoryURL.appendingPathComponent(classLabel)

        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: subdirectoryURL.path, isDirectory: &isDirectory),
              isDirectory.boolValue
        else {
            let message = "サブディレクトリ「\(classLabel)」が見つからないか、ディレクトリではありません: \(subdirectoryURL.path)"
            XCTFail(message)
            throw TestError.testResourceMissing
        }

        let allFiles = try fileManager.contentsOfDirectory(
            at: subdirectoryURL,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )
        let validExtensions = ["jpg", "jpeg", "png"]
        let imageFiles = allFiles.filter { validExtensions.contains($0.pathExtension.lowercased()) }

        guard let randomImageURL = imageFiles.randomElement() else {
            let message =
                "サブディレクトリ「\(classLabel)」に、指定された拡張子 (\(validExtensions.joined(separator: ", ")) の画像ファイルが見つかりません: \(subdirectoryURL.path)"
            XCTFail(message)
            throw TestError.testResourceMissing
        }

        return randomImageURL
    }
}
