@testable import BinaryClassification
import CoreML
import CreateML
import Foundation
import Vision
import XCTest

final class BinaryClassificationTests: XCTestCase {
    var trainer: BinaryClassificationTrainer!
    let fileManager = FileManager.default
    var authorName: String = "Test Author"
    var testModelName: String = "TestCats_Binary_Run"
    var testModelVersion: String = "v1"

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
    var trainingResult: BinaryClassification.BinaryTrainingResult?

    var testResourcesRootPath: String {
        var sourcesDir = URL(fileURLWithPath: #filePath)
        sourcesDir.deleteLastPathComponent()
        return sourcesDir.appendingPathComponent("TestResources").path
    }

    override func setUp() async throws {
        try await super.setUp()

        temporaryOutputDirectoryURL = fileManager.temporaryDirectory
            .appendingPathComponent("TestOutput_\(UUID().uuidString)")
        print("一時ディレクトリを作成します: \(temporaryOutputDirectoryURL.path)")
        try fileManager.createDirectory(
            at: temporaryOutputDirectoryURL,
            withIntermediateDirectories: true,
            attributes: nil
        )

        trainer = BinaryClassificationTrainer(
            resourcesDirectoryPathOverride: testResourcesRootPath,
            outputDirectoryPathOverride: temporaryOutputDirectoryURL.path
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

        let trainedModelURL = URL(fileURLWithPath: result.trainedModelFilePath)
        do {
            compiledModelURL = try await MLModel.compileModel(at: trainedModelURL)
        } catch {
            print("モデルのコンパイル失敗 in setUp: \(error.localizedDescription)")
            throw error
        }
    }

    override func tearDownWithError() throws {
        if let tempDir = temporaryOutputDirectoryURL, fileManager.fileExists(atPath: tempDir.path) {
            print("一時ディレクトリを削除します: \(tempDir.path)")
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

    // BinaryClassificationTrainer の初期化をテスト
    func testTrainerInitialization() {
        XCTAssertNotNil(trainer, "BinaryClassificationTrainerの初期化失敗")
        XCTAssertEqual(trainer.resourcesDirectoryPath, testResourcesRootPath, "トレーナーのリソースパスがオーバーライド値と不一致")
        XCTAssertEqual(trainer.outputDirPath, temporaryOutputDirectoryURL.path, "トレーナーの出力パスがオーバーライド値と不一致")
    }

    enum TestError: Error {
        case testResourceMissing
        case trainingFailed
        case modelFileMissing
        case predictionFailed
        case setupFailed
    }

    // モデルの訓練と成果物の生成をテスト
    func testModelTrainingAndArtifactGeneration() throws {
        guard let result = trainingResult else {
            XCTFail("訓練結果がnil (testModelTrainingAndArtifacts)")
            throw TestError.trainingFailed
        }

        XCTAssertTrue(
            fileManager.fileExists(atPath: result.trainedModelFilePath),
            "訓練モデルファイルが期待パス「\(result.trainedModelFilePath)」に見つからない"
        )

        let expectedClassLabels = ["NotScary", "Scary"].sorted()
        XCTAssertEqual(
            Set(result.detectedClassLabelsList.sorted()),
            Set(expectedClassLabels),
            "検出クラスラベル「\(result.detectedClassLabelsList.sorted())」が期待ラベル「\(expectedClassLabels)」と不一致"
        )

        result.saveLog(modelAuthor: authorName, modelName: testModelName, modelVersion: testModelVersion)
        let modelFileDir = URL(fileURLWithPath: result.trainedModelFilePath).deletingLastPathComponent()

        let expectedLogFileName = "\(testModelName)_\(testModelVersion).md"
        let expectedLogFilePath = modelFileDir.appendingPathComponent(expectedLogFileName).path
        XCTAssertTrue(fileManager.fileExists(atPath: expectedLogFilePath), "ログファイルが期待パス「\(expectedLogFilePath)」に未生成")

        XCTAssertEqual(result.modelName, testModelName, "訓練結果modelName「\(result.modelName)」が期待値「\(testModelName)」と不一致")

        do {
            let logContents = try String(contentsOfFile: expectedLogFilePath, encoding: .utf8)
            XCTAssertFalse(logContents.isEmpty, "ログファイルが空です: \(expectedLogFilePath)")
        } catch {
            XCTFail("ログファイル内容読込不可: \(expectedLogFilePath), エラー: \(error.localizedDescription)")
        }

        XCTAssertTrue(
            result.trainedModelFilePath.contains(testModelName),
            "モデルファイルパスにモデル名「\(testModelName)」が含まれていません"
        )
        XCTAssertTrue(
            result.trainedModelFilePath.contains(testModelVersion),
            "モデルファイルパスにバージョン「\(testModelVersion)」が含まれていません"
        )
        XCTAssertTrue(
            result.trainedModelFilePath.contains(trainer.classificationMethod),
            "モデルファイルパスに分類法「\(trainer.classificationMethod)」が含まれていません"
        )
    }

    // モデルが予測を実行できるかテスト
    func testModelCanPerformPrediction() throws {
        guard let finalModelURL = compiledModelURL else {
            XCTFail("コンパイル済みモデルURLがnil (testModelPredictionAccuracy)")
            throw TestError.modelFileMissing
        }
        guard let result = trainingResult else {
            XCTFail("訓練結果がnil (testModelPredictionAccuracy)")
            throw TestError.trainingFailed
        }

        let mlModel = try MLModel(contentsOf: finalModelURL)
        let vnCoreMLModel = try VNCoreMLModel(for: mlModel)
        let predictionRequest = VNCoreMLRequest(model: vnCoreMLModel)

        let baseResourceURL = URL(fileURLWithPath: testResourcesRootPath)
        print("テストリソースのベースURL: \(baseResourceURL.path)")

        let classLabels = result.detectedClassLabelsList.sorted()
        guard classLabels.count >= 2 else {
            XCTFail("テストには少なくとも2つのクラスラベルが訓練結果に必要です。検出されたラベル: \(classLabels)")
            throw TestError.setupFailed
        }

        let classLabel1 = classLabels[0]
        let classLabel2 = classLabels[1]

        print("動的に識別されたテスト用クラスラベル: '\(classLabel1)' および '\(classLabel2)'")

        // classLabel1 の画像取得処理
        let imageURL1: URL
        do {
            imageURL1 = try getRandomImageURL(
                forClassLabel: classLabel1,
                inBaseDirectory: baseResourceURL
            )
        } catch {
            XCTFail("'\(classLabel1)' サブディレクトリからのランダム画像取得失敗。エラー: \(error.localizedDescription)")
            throw error
        }

        // classLabel2 の画像取得処理
        let imageURL2: URL
        do {
            imageURL2 = try getRandomImageURL(
                forClassLabel: classLabel2,
                inBaseDirectory: baseResourceURL
            )
        } catch {
            XCTFail("'\(classLabel2)' サブディレクトリからのランダム画像取得失敗。エラー: \(error.localizedDescription)")
            throw error
        }

        let imageHandler1 = VNImageRequestHandler(url: imageURL1, options: [:])
        try imageHandler1.perform([predictionRequest])
        guard let observations1 = predictionRequest.results as? [VNClassificationObservation],
              let topResult1 = observations1.first
        else {
            XCTFail("クラス '\(classLabel1)' 画像: 有効な分類結果オブジェクトを取得できませんでした。")
            throw TestError.predictionFailed
        }
        XCTAssertNotNil(topResult1.identifier, "クラス '\(classLabel1)' 画像: 予測結果からクラスラベルを取得できませんでした。")
        print("クラス '\(classLabel1)' 画像の予測 (正解ラベル): \(topResult1.identifier) (確信度: \(topResult1.confidence))")

        let imageHandler2 = VNImageRequestHandler(url: imageURL2, options: [:])
        try imageHandler2.perform([predictionRequest])
        guard let observations2 = predictionRequest.results as? [VNClassificationObservation],
              let topResult2 = observations2.first
        else {
            XCTFail("クラス '\(classLabel2)' 画像: 有効な分類結果オブジェクトを取得できませんでした。")
            throw TestError.predictionFailed
        }
        XCTAssertNotNil(topResult2.identifier, "クラス '\(classLabel2)' 画像: 予測結果からクラスラベルを取得できませんでした。")
        print("クラス '\(classLabel2)' 画像の予測 (正解ラベル): \(topResult2.identifier) (確信度: \(topResult2.confidence))")
    }

    private func getRandomImageURL(
        forClassLabel classLabel: String,
        inBaseDirectory baseDirectoryURL: URL
    ) throws -> URL {
        let subdirectoryURL = baseDirectoryURL.appendingPathComponent(classLabel)
        print("'\(classLabel)' のサブディレクトリにアクセス試行: \(subdirectoryURL.path)")

        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: subdirectoryURL.path, isDirectory: &isDirectory),
              isDirectory.boolValue
        else {
            let message = "サブディレクトリ '\(classLabel)' が見つからないか、ディレクトリではありません: \(subdirectoryURL.path)"
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
                "サブディレクトリ '\(classLabel)' に指定拡張子 (\(validExtensions.joined(separator: ", ")) の画像ファイルが見つかりません: \(subdirectoryURL.path)"
            XCTFail(message)
            throw TestError.testResourceMissing
        }

        print("クラス '\(classLabel)' のテスト画像URLとして使用: \(randomImageURL.path)")
        return randomImageURL
    }
}
