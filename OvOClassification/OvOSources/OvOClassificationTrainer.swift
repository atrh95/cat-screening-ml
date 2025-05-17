import Combine
import CoreML
import CreateML
import CSInterface
import Foundation
import TabularData

// OvOペアのトレーニング結果を格納する
private struct OvOPairTrainingResult {
    let modelPath: String
    let modelName: String
    let class1Name: String // OvOペアのクラス1
    let class2Name: String // OvOペアのクラス2
    let trainingAccuracyRate: Double
    let validationAccuracyRate: Double
    let trainingErrorRate: Double
    let validationErrorRate: Double
    let trainingTime: TimeInterval
    let trainingDataPath: String // このペアのトレーニングに使用されたデータのパス
    let individualModelDescription: String
}

public class OvOClassificationTrainer: ScreeningTrainerProtocol {
    public typealias TrainingResultType = OvOTrainingResult

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

    public var classificationMethod: String { "OvO" }

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

    static let fileManager = FileManager.default
    static let tempBaseDirName = "TempOvOTrainingData"

    public func train(
        author: String,
        modelName: String,
        version: String,
        modelParameters: CreateML.MLImageClassifier.ModelParameters
    ) async -> OvOTrainingResult? {
        let mainOutputRunURL: URL
        do {
            mainOutputRunURL = try createOutputDirectory(
                modelName: modelName,
                version: version
            )
        } catch {
            print("🛑 エラー: 出力ディレクトリ設定失敗: \(error.localizedDescription)")
            return nil
        }

        let baseProjectURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let tempOvOBaseURL = baseProjectURL.appendingPathComponent(Self.tempBaseDirName) // OvO用一時ディレクトリベースパス
        defer {
            if Self.fileManager.fileExists(atPath: tempOvOBaseURL.path) {
                do {
                    try Self.fileManager.removeItem(at: tempOvOBaseURL)
                    print("🗑️ 一時ディレクトリ \(tempOvOBaseURL.path) クリーンアップ完了")
                } catch {
                    print("⚠️ 一時ディレクトリ \(tempOvOBaseURL.path) クリーンアップ失敗: \(error.localizedDescription)")
                }
            }
        }

        if Self.fileManager.fileExists(atPath: tempOvOBaseURL.path) {
            try? Self.fileManager.removeItem(at: tempOvOBaseURL)
        }
        guard (try? Self.fileManager.createDirectory(at: tempOvOBaseURL, withIntermediateDirectories: true)) != nil
        else {
            print("🛑 エラー: 一時ディレクトリ \(tempOvOBaseURL.path) 作成失敗。処理中止。")
            return nil
        }

        let ovoResourcesURL = URL(fileURLWithPath: resourcesDirectoryPath) // Use the (potentially overridden) property

        print("🚀 OvOトレーニング開始 (バージョン: \(version))...")

        let allLabelSourceDirectories: [URL]
        do {
            allLabelSourceDirectories = try Self.fileManager.contentsOfDirectory(
                at: ovoResourcesURL,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: .skipsHiddenFiles
            ).filter { url in
                var isDirectory: ObjCBool = false
                Self.fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory)
                return isDirectory.boolValue && !url.lastPathComponent.hasPrefix(".") // 隠しファイルを除外
            }
        } catch {
            print("🛑 エラー: リソースディレクトリ内ラベルディレクトリ取得失敗: \(error.localizedDescription)")
            return nil
        }

        // OvOでは最低2つのクラスが必要
        guard allLabelSourceDirectories.count >= 2 else {
            print("🛑 エラー: OvOトレーニングには最低2つのクラスラベルディレクトリが必要です。現在 \(allLabelSourceDirectories.count)個。処理中止。")
            return nil
        }

        print("  検出された総ラベル数: \(allLabelSourceDirectories.count)")

        // クラスペアを生成 (例: [A,B], [A,C], [B,C])
        var classPairs: [(URL, URL)] = []
        for i in 0 ..< allLabelSourceDirectories.count {
            for j in (i + 1) ..< allLabelSourceDirectories.count {
                classPairs.append((allLabelSourceDirectories[i], allLabelSourceDirectories[j]))
            }
        }

        if classPairs.isEmpty {
            print("🛑 エラー: 有効なクラスペアが生成できませんでした。処理中止。")
            return nil
        }

        print("  生成されたOvOペア数: \(classPairs.count)")

        var allPairTrainingResults: [OvOPairTrainingResult] = []
        var pairIndex = 0

        for pair in classPairs {
            let dir1 = pair.0
            let dir2 = pair.1
            print(
                "🔄 OvOペア \(pairIndex + 1)/\(classPairs.count): [\(dir1.lastPathComponent)] vs [\(dir2.lastPathComponent)] トレーニング開始..."
            )
            if let result = await trainSingleOvOPair(
                class1DirURL: dir1,
                class2DirURL: dir2,
                mainRunURL: mainOutputRunURL,
                tempOvOBaseURL: tempOvOBaseURL, // OvO用一時ベースURL
                modelName: modelName, // ベースモデル名
                author: author,
                version: version,
                pairIndex: pairIndex,
                modelParameters: modelParameters
            ) {
                allPairTrainingResults.append(result)
                print("  ✅ OvOペア [\(dir1.lastPathComponent)] vs [\(dir2.lastPathComponent)] トレーニング成功")
            } else {
                print("  ⚠️ OvOペア [\(dir1.lastPathComponent)] vs [\(dir2.lastPathComponent)] トレーニング失敗またはスキップ")
            }
            pairIndex += 1
        }

        guard !allPairTrainingResults.isEmpty else {
            print("🛑 エラー: 有効なOvOペアトレーニングが一つも完了しませんでした。処理中止。")
            return nil
        }

        // IndividualModelReportの作成
        let individualReports: [IndividualModelReport] = allPairTrainingResults.map { result in
            IndividualModelReport(
                modelName: result.modelName,
                // OvOでは「陽性クラス」という概念がOvRと異なるため、ペアの情報を格納する
                positiveClassName: "\(result.class1Name)_vs_\(result.class2Name)",
                trainingAccuracyRate: result.trainingAccuracyRate,
                validationAccuracyPercentage: result.validationAccuracyRate,
                // OvOの再現率・適合率は各クラス視点で計算可能だが、ここではペア全体の精度を重視
                recallRate: 0,
                precisionRate: 0,
                modelDescription: result.individualModelDescription
            )
        }

        let trainingDataPaths = allPairTrainingResults.map(\.trainingDataPath).joined(separator: "; ")
        let finalRunOutputPath = mainOutputRunURL.path

        print("🎉 OvOトレーニング全体完了。結果出力先: \(finalRunOutputPath)")

        let trainingResult = OvOTrainingResult(
            modelOutputPath: finalRunOutputPath,
            trainingDataPaths: trainingDataPaths,
            maxIterations: modelParameters.maxIterations,
            individualReports: individualReports,
            numberOfClasses: allLabelSourceDirectories.count, // 総クラス数を追加
            numberOfPairs: classPairs.count // 総ペア数を追加
        )

        return trainingResult
    }

    // 1つのOvOペアのモデルをトレーニングする関数
    private func trainSingleOvOPair(
        class1DirURL: URL,
        class2DirURL: URL,
        mainRunURL: URL, // 各ペアモデルの保存先ディレクトリの親
        tempOvOBaseURL: URL, // 全ペアの一時データ保存ルート
        modelName: String, // ユーザー指定の基本モデル名
        author: String,
        version: String,
        pairIndex _: Int,
        modelParameters: CreateML.MLImageClassifier.ModelParameters
    ) async -> OvOPairTrainingResult? {
        let class1NameOriginal = class1DirURL.lastPathComponent
        let class2NameOriginal = class2DirURL.lastPathComponent

        // モデル名やディレクトリ名に使用するクラス名 (英数字のみに整形)
        let class1NameForModel = class1NameOriginal.components(separatedBy: CharacterSet(charactersIn: "_-"))
            .map(\.capitalized)
            .joined()
            .replacingOccurrences(of: "[^a-zA-Z0-9]", with: "", options: .regularExpression)

        let class2NameForModel = class2NameOriginal.components(separatedBy: CharacterSet(charactersIn: "_-"))
            .map(\.capitalized)
            .joined()
            .replacingOccurrences(of: "[^a-zA-Z0-9]", with: "", options: .regularExpression)

        // モデルファイル名と一時ディレクトリ名を作成
        // 例: MyCatModel_OvO_Siamese_vs_Persian_v1.0
        let pairModelFileNameBase =
            "\(modelName)_\(classificationMethod)_\(class1NameForModel)_vs_\(class2NameForModel)_\(version)"
        let tempOvOPairRootName = "\(pairModelFileNameBase)_TempData"
        let tempOvOPairRootURL = tempOvOBaseURL.appendingPathComponent(tempOvOPairRootName)

        // CreateMLのImageClassifierに渡すためのディレクトリ
        let tempClass1DataDirForML = tempOvOPairRootURL.appendingPathComponent(class1NameForModel)
        let tempClass2DataDirForML = tempOvOPairRootURL.appendingPathComponent(class2NameForModel)

        // 既存の一時ペアディレクトリがあれば削除
        if Self.fileManager.fileExists(atPath: tempOvOPairRootURL.path) {
            try? Self.fileManager.removeItem(at: tempOvOPairRootURL)
        }
        do {
            try Self.fileManager.createDirectory(at: tempClass1DataDirForML, withIntermediateDirectories: true)
            try Self.fileManager.createDirectory(at: tempClass2DataDirForML, withIntermediateDirectories: true)
        } catch {
            print(
                "🛑 エラー: OvOペア [\(class1NameForModel) vs \(class2NameForModel)] 一時学習ディレクトリ作成失敗: \(error.localizedDescription)"
            )
            return nil
        }

        // class1の画像を一時ディレクトリにコピー
        var class1SamplesCount = 0
        if let class1SourceFiles = try? getFilesInDirectory(class1DirURL) {
            for fileURL in class1SourceFiles {
                try? Self.fileManager.copyItem(
                    at: fileURL,
                    to: tempClass1DataDirForML.appendingPathComponent(fileURL.lastPathComponent)
                )
            }
            class1SamplesCount = (try? getFilesInDirectory(tempClass1DataDirForML).count) ?? 0
        }
        guard class1SamplesCount > 0 else {
            print(
                "⚠️ OvOペア [\(class1NameForModel) vs \(class2NameForModel)]: クラス1 [\(class1NameForModel)] のサンプルなし。学習スキップ。Path: \(tempClass1DataDirForML.path)"
            )
            try? Self.fileManager.removeItem(at: tempOvOPairRootURL) // 不要な一時ディレクトリを削除
            return nil
        }

        // class2の画像を一時ディレクトリにコピー
        var class2SamplesCount = 0
        if let class2SourceFiles = try? getFilesInDirectory(class2DirURL) {
            for fileURL in class2SourceFiles {
                try? Self.fileManager.copyItem(
                    at: fileURL,
                    to: tempClass2DataDirForML.appendingPathComponent(fileURL.lastPathComponent)
                )
            }
            class2SamplesCount = (try? getFilesInDirectory(tempClass2DataDirForML).count) ?? 0
        }
        guard class2SamplesCount > 0 else {
            print(
                "⚠️ OvOペア [\(class1NameForModel) vs \(class2NameForModel)]: クラス2 [\(class2NameForModel)] のサンプルなし。学習スキップ。Path: \(tempClass2DataDirForML.path)"
            )
            try? Self.fileManager.removeItem(at: tempOvOPairRootURL) // 不要な一時ディレクトリを削除
            return nil
        }

        print(
            "  準備完了: [\(class1NameForModel)] (\(class1SamplesCount)枚) vs [\(class2NameForModel)] (\(class2SamplesCount)枚)"
        )

        let startTime = Date()
        var trainingAccuracy: Double = 0
        var validationAccuracy: Double = 0
        var trainingError = 1.0
        var validationError = 1.0

        let modelPath = mainRunURL.appendingPathComponent("\(pairModelFileNameBase).mlmodel")

        do {
            print("  CreateML ImageClassifier トレーニング開始: [\(class1NameForModel)] vs [\(class2NameForModel)]")
            let trainingDataSource = MLImageClassifier.DataSource.labeledDirectories(at: tempOvOPairRootURL)

            let classifier = try MLImageClassifier(trainingData: trainingDataSource, parameters: modelParameters)

            trainingAccuracy = 1.0 - classifier.trainingMetrics.classificationError
            trainingError = classifier.trainingMetrics.classificationError

            validationAccuracy = 1.0 - classifier.validationMetrics.classificationError
            validationError = classifier.validationMetrics.classificationError

            var descriptionParts: [String] = []
            descriptionParts.append(String(
                format: "クラス構成: %@: %d枚; %@: %d枚",
                class1NameForModel,
                class1SamplesCount,
                class2NameForModel,
                class2SamplesCount
            ))
            descriptionParts.append("最大反復回数: \(modelParameters.maxIterations)回")
            descriptionParts.append(String(
                format: "訓練正解率: %.1f%%, 検証正解率: %.1f%%",
                trainingAccuracy * 100,
                validationAccuracy * 100
            ))
            descriptionParts.append("(検証: 自動分割)")
            let individualDesc = descriptionParts.joined(separator: "\n")

            let modelMetadata = MLModelMetadata(author: author, shortDescription: individualDesc, version: version)

            try classifier.write(to: modelPath, metadata: modelMetadata)
            print("  ✅ モデル保存成功: \(modelPath.path)")

            let trainingTime = Date().timeIntervalSince(startTime)
            print(String(format: "  ⏱️ トレーニング時間: %.2f 秒", trainingTime))
            print(String(format: "  📈 トレーニング精度 (代用): %.2f%%", trainingAccuracy * 100))
            print(String(format: "  📊 検証精度: %.2f%%", validationAccuracy * 100))

            return OvOPairTrainingResult(
                modelPath: modelPath.path,
                modelName: pairModelFileNameBase,
                class1Name: class1NameForModel,
                class2Name: class2NameForModel,
                trainingAccuracyRate: trainingAccuracy,
                validationAccuracyRate: validationAccuracy,
                trainingErrorRate: trainingError,
                validationErrorRate: validationError,
                trainingTime: trainingTime,
                trainingDataPath: tempOvOPairRootURL.path,
                individualModelDescription: individualDesc
            )

        } catch let createMLError as CreateML.MLCreateError {
            print(
                "🛑 エラー: CreateML ImageClassifier トレーニングまたはモデル保存失敗 [\(class1NameForModel) vs \(class2NameForModel)]: \(createMLError.localizedDescription)"
            )
            print("  詳細情報: \(createMLError)")
            try? Self.fileManager.removeItem(at: tempOvOPairRootURL)
            return nil
        } catch {
            print(
                "🛑 エラー: CreateML ImageClassifier トレーニングまたはモデル保存失敗 [\(class1NameForModel) vs \(class2NameForModel)]: \(error.localizedDescription)"
            )
            try? Self.fileManager.removeItem(at: tempOvOPairRootURL)
            return nil
        }
    }

    // 指定されたディレクトリ内のファイル一覧を取得する
    private func getFilesInDirectory(_ directoryURL: URL) throws -> [URL] {
        try Self.fileManager.contentsOfDirectory(
            at: directoryURL,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants]
        ).filter { url in
            !url.lastPathComponent
                .hasPrefix(".") && (try? url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true
        }
    }
}
