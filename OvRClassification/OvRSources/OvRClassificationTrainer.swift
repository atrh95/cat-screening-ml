import Combine
import CoreML
import CreateML
import CSInterface
import Foundation
import TabularData

private struct OvRPairTrainingResult {
    let modelPath: String
    let modelName: String
    let positiveClassName: String
    let trainingAccuracyRate: Double
    let validationAccuracyRate: Double
    let trainingErrorRate: Double
    let validationErrorRate: Double
    let trainingTime: TimeInterval
    let trainingDataPath: String
    let recallRate: Double
    let precisionRate: Double
    let individualModelDescription: String
}

public class OvRClassificationTrainer: ScreeningTrainerProtocol {
    public typealias TrainingResultType = OvRTrainingResult
    
    public var outputDirPath: String {
        var dir = URL(fileURLWithPath: #filePath)
        dir.deleteLastPathComponent()
        dir.deleteLastPathComponent()
        return dir.appendingPathComponent("OutputModels").path
    }
    
    public var classificationMethod: String { "OvR" }
    
    public var resourcesDirectoryPath: String {
        var dir = URL(fileURLWithPath: #filePath)
        dir.deleteLastPathComponent()
        dir.deleteLastPathComponent()
        return dir.appendingPathComponent("Resources").path
    }
    
    public init() {}
    
    static let fileManager = FileManager.default
    static let tempBaseDirName = "TempOvRTrainingData"
    
    public func train(
        author: String,
        modelName: String,
        version: String,
        maxIterations: Int
    ) async -> OvRTrainingResult? {
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
        
        let baseProjectURL = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
        let tempOvRBaseURL = baseProjectURL.appendingPathComponent(Self.tempBaseDirName)
        defer {
            if Self.fileManager.fileExists(atPath: tempOvRBaseURL.path) {
                do {
                    try Self.fileManager.removeItem(at: tempOvRBaseURL)
                    print("🗑️ 一時ディレクトリ \(tempOvRBaseURL.path) クリーンアップ完了")
                } catch {
                    print("⚠️ 一時ディレクトリ \(tempOvRBaseURL.path) クリーンアップ失敗: \(error.localizedDescription)")
                }
            }
        }
        
        if Self.fileManager.fileExists(atPath: tempOvRBaseURL.path) {
            try? Self.fileManager.removeItem(at: tempOvRBaseURL)
        }
        guard (try? Self.fileManager.createDirectory(at: tempOvRBaseURL, withIntermediateDirectories: true)) != nil
        else {
            print("🛑 エラー: 一時ディレクトリ \(tempOvRBaseURL.path) 作成失敗。処理中止。")
            return nil
        }
        
        let ovrResourcesURL = URL(fileURLWithPath: resourcesDirectoryPath)
        
        print("🚀 OvRトレーニング開始 (バージョン: \(version))...")
        
        let allLabelSourceDirectories: [URL]
        do {
            allLabelSourceDirectories = try Self.fileManager.contentsOfDirectory(
                at: ovrResourcesURL,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: .skipsHiddenFiles
            ).filter { url in
                var isDirectory: ObjCBool = false
                Self.fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory)
                return isDirectory.boolValue && !url.lastPathComponent.hasPrefix(".")
            }
        } catch {
            print("🛑 エラー: リソースディレクトリ内ラベルディレクトリ取得失敗: \(error.localizedDescription)")
            return nil
        }
        
        let primaryLabelSourceDirs = allLabelSourceDirectories.filter { $0.lastPathComponent.lowercased() != "safe" }
        
        if primaryLabelSourceDirs.isEmpty {
            print("🛑 エラー: プライマリトレーニング対象ディレクトリが見つかりません ('safe'除く)。処理中止。")
            return nil
        }
        
        print("  処理対象主要ラベル数 ('safe'除く): \(primaryLabelSourceDirs.count)")
        
        var allPairTrainingResults: [OvRPairTrainingResult] = []
        
        for (index, dir) in primaryLabelSourceDirs.enumerated() {
            print(
                "🔄 OvRペア \(index + 1)/\(primaryLabelSourceDirs.count): [\(dir.lastPathComponent)] vs Rest トレーニング開始..."
            )
            if let result = await trainSingleOvRPair(
                oneLabelSourceDirURL: dir,
                allLabelSourceDirs: allLabelSourceDirectories,
                mainRunURL: mainOutputRunURL,
                tempOvRBaseURL: tempOvRBaseURL,
                modelName: modelName,
                author: author,
                version: version,
                pairIndex: index,
                maxIterations: maxIterations
            ) {
                allPairTrainingResults.append(result)
                print("  ✅ OvRペア [\(dir.lastPathComponent)] vs Rest トレーニング成功")
            } else {
                print("  ⚠️ OvRペア [\(dir.lastPathComponent)] vs Rest トレーニング失敗またはスキップ")
            }
        }
        
        guard !allPairTrainingResults.isEmpty else {
            print("🛑 エラー: 有効なOvRペアトレーニングが一つも完了しませんでした。処理中止。")
            return nil
        }
        
        let individualReports: [IndividualModelReport] = allPairTrainingResults.map { result in
            IndividualModelReport(
                modelName: result.modelName,
                positiveClassName: result.positiveClassName,
                trainingAccuracyRate: result.trainingAccuracyRate,
                validationAccuracyPercentage: result.validationAccuracyRate,
                recallRate: result.recallRate,
                precisionRate: result.precisionRate,
                modelDescription: result.individualModelDescription
            )
        }
        
        let trainingDataPaths = allPairTrainingResults.map(\.trainingDataPath).joined(separator: "; ")
        
        let finalRunOutputPath = mainOutputRunURL.path
        
        print("🎉 OvRトレーニング全体完了。結果出力先: \(finalRunOutputPath)")
        
        let trainingResult = OvRTrainingResult(
            modelOutputPath: finalRunOutputPath,
            trainingDataPaths: trainingDataPaths,
            maxIterations: maxIterations,
            individualReports: individualReports
        )
        
        return trainingResult
    }
    
    private func trainSingleOvRPair(
        oneLabelSourceDirURL: URL,
        allLabelSourceDirs: [URL],
        mainRunURL: URL,
        tempOvRBaseURL: URL,
        modelName: String,
        author: String,
        version: String,
        pairIndex: Int,
        maxIterations: Int
    ) async -> OvRPairTrainingResult? {
        let originalOneLabelName = oneLabelSourceDirURL.lastPathComponent
        let positiveClassNameForModel = originalOneLabelName.components(separatedBy: CharacterSet(charactersIn: "_-"))
            .map(\.capitalized)
            .joined()
            .replacingOccurrences(of: "[^a-zA-Z0-9]", with: "", options: .regularExpression)
        
        let modelFileNameBase =
        "\(modelName)_\(classificationMethod)_\(positiveClassNameForModel)_vs_Rest_v\(version)_idx\(pairIndex)"
        let tempOvRPairRootName = "\(modelFileNameBase)_TempData"
        let tempOvRPairRootURL = tempOvRBaseURL.appendingPathComponent(tempOvRPairRootName)
        
        let tempPositiveDataDirForML = tempOvRPairRootURL.appendingPathComponent(positiveClassNameForModel)
        let tempRestDataDirForML = tempOvRPairRootURL.appendingPathComponent("Rest")
        
        if Self.fileManager.fileExists(atPath: tempOvRPairRootURL.path) {
            try? Self.fileManager.removeItem(at: tempOvRPairRootURL)
        }
        do {
            try Self.fileManager.createDirectory(at: tempPositiveDataDirForML, withIntermediateDirectories: true)
            try Self.fileManager.createDirectory(at: tempRestDataDirForML, withIntermediateDirectories: true)
        } catch {
            print("🛑 エラー: OvRペア [\(positiveClassNameForModel)] 一時学習ディレクトリ作成失敗: \(error.localizedDescription)")
            return nil
        }
        
        var positiveSamplesCount = 0
        if let positiveSourceFiles = try? getFilesInDirectory(oneLabelSourceDirURL) {
            for fileURL in positiveSourceFiles {
                try? Self.fileManager.copyItem(
                    at: fileURL,
                    to: tempPositiveDataDirForML.appendingPathComponent(fileURL.lastPathComponent)
                )
            }
            positiveSamplesCount = (try? getFilesInDirectory(tempPositiveDataDirForML).count) ?? 0
        }
        
        guard positiveSamplesCount > 0 else {
            print("⚠️ OvRペア [\(positiveClassNameForModel)]: ポジティブサンプルなし。学習スキップ。 Path: \(tempPositiveDataDirForML.path)")
            return nil
        }
        
        let otherDirsForNegativeSampling = allLabelSourceDirs.filter { $0.path != oneLabelSourceDirURL.path }
        
        if otherDirsForNegativeSampling.isEmpty {
            print("ℹ️ OvRペア [\(positiveClassNameForModel)]: ネガティブサンプリング対象の他ディレクトリなし。学習スキップ。")
            return nil
        }
        
        let numFilesToCollectPerOtherDir =
        Int(ceil(Double(positiveSamplesCount) / Double(otherDirsForNegativeSampling.count)))
        var totalNegativeSamplesCollected = 0
        
        for otherDirURL in otherDirsForNegativeSampling {
            guard let filesInOtherDir = try? getFilesInDirectory(otherDirURL), !filesInOtherDir.isEmpty else {
                print(
                    "ℹ️ OvRペア [\(positiveClassNameForModel)]: ディレクトリ \(otherDirURL.lastPathComponent) 空またはアクセス不可。ネガティブサンプル収集からスキップ。"
                )
                continue
            }
            
            let filesToCopy = filesInOtherDir.shuffled().prefix(numFilesToCollectPerOtherDir)
            for fileURL in filesToCopy {
                let sourceDirNamePrefix = otherDirURL.lastPathComponent
                    .replacingOccurrences(of: "[^a-zA-Z0-9_.-]", with: "_", options: .regularExpression)
                let originalFileName = fileURL.lastPathComponent
                    .replacingOccurrences(of: "[^a-zA-Z0-9_.-]", with: "_", options: .regularExpression)
                let newFileName = "\(sourceDirNamePrefix)_\(originalFileName)"
                
                do {
                    try Self.fileManager.copyItem(
                        at: fileURL,
                        to: tempRestDataDirForML.appendingPathComponent(newFileName)
                    )
                    totalNegativeSamplesCollected += 1
                } catch {
                    print(
                        "⚠️ OvRペア [\(positiveClassNameForModel)]: \(fileURL.path) から \(newFileName) へのコピー失敗: \(error.localizedDescription)"
                    )
                }
            }
        }
        
        guard totalNegativeSamplesCollected > 0 else {
            print("⚠️ OvRペア [\(positiveClassNameForModel)]: ネガティブサンプル収集失敗。学習スキップ。")
            return nil
        }
        
        print(
            "  OvRペア [\(positiveClassNameForModel)]: 学習データ準備完了 (ポジティブ: \(positiveSamplesCount)枚, ネガティブ: \(totalNegativeSamplesCollected)枚)"
        )
        
        let trainingDataSource = MLImageClassifier.DataSource.labeledDirectories(at: tempOvRPairRootURL)
        let modelForPairName = "\(modelName)_\(classificationMethod)_\(positiveClassNameForModel)_vs_Rest"
        
        do {
            let trainingStartTime = Date()
            var modelParameters = MLImageClassifier.ModelParameters()
            modelParameters.featureExtractor = .scenePrint(revision: 1)
            modelParameters.maxIterations = maxIterations
            modelParameters.validation = .split(strategy: .automatic)
            
            print("  ⏳ OvRペア [\(positiveClassNameForModel)] モデルトレーニング実行中 (最大反復: \(maxIterations)回)...")
            let imageClassifier = try MLImageClassifier(trainingData: trainingDataSource, parameters: modelParameters)
            print("  ✅ OvRペア [\(positiveClassNameForModel)] モデルトレーニング完了")
            
            let trainingEndTime = Date()
            let trainingDurationSeconds = trainingEndTime.timeIntervalSince(trainingStartTime)
            
            let trainingMetrics = imageClassifier.trainingMetrics
            let validationMetrics = imageClassifier.validationMetrics
            
            let trainingAccuracy = (1.0 - trainingMetrics.classificationError) * 100.0
            let validationAccuracy = (1.0 - validationMetrics.classificationError) * 100.0
            
            var recall = 0.0
            var precision = 0.0
            
            let confusionMatrix = validationMetrics.confusion
            print("  デバッグ [\(positiveClassNameForModel)]: 混同行列の内容: \(confusionMatrix.description)")
            print("  デバッグ [\(positiveClassNameForModel)]: 混同行列の列名: \(confusionMatrix.columnNames)")
            
            var labelSet = Set<String>()
            var rowCount = 0
            for row in confusionMatrix.rows {
                rowCount += 1
                if let actual = row["True Label"]?.stringValue { labelSet.insert(actual) }
                if let predicted = row["Predicted"]?.stringValue { labelSet.insert(predicted) }
            }
            print("  デバッグ [\(positiveClassNameForModel)]: 混同行列から処理された総行数: \(rowCount)")
            print("  デバッグ [\(positiveClassNameForModel)]: 混同行列から抽出されたラベルセット: \(labelSet)")
            
            let classLabelsFromConfusion = Array(labelSet).sorted()
            
            if classLabelsFromConfusion.contains(positiveClassNameForModel), classLabelsFromConfusion.contains("Rest") {
                var truePositives = 0
                var falsePositives = 0
                var falseNegatives = 0
                
                for row in confusionMatrix.rows {
                    guard
                        let actual = row["True Label"]?.stringValue,
                        let predicted = row["Predicted"]?.stringValue,
                        let cnt = row["Count"]?.intValue
                    else { continue }
                    
                    if actual == positiveClassNameForModel, predicted == positiveClassNameForModel {
                        truePositives += cnt
                    } else if actual == "Rest", predicted == positiveClassNameForModel {
                        falsePositives += cnt
                    } else if actual == positiveClassNameForModel, predicted == "Rest" {
                        falseNegatives += cnt
                    }
                }
                if (truePositives + falseNegatives) > 0 {
                    recall = Double(truePositives) / Double(truePositives + falseNegatives)
                }
                if (truePositives + falsePositives) > 0 {
                    precision = Double(truePositives) / Double(truePositives + falsePositives)
                }
            } else {
                print(
                    "  ⚠️ OvRペア [\(positiveClassNameForModel)]: 混同行列から期待されるラベル ('\(positiveClassNameForModel)', 'Rest') が見つからず、再現率/適合率計算スキップ。"
                )
            }
            
            let positiveCountForDesc = (try? getFilesInDirectory(tempPositiveDataDirForML).count) ?? 0
            let restCountForDesc = (try? getFilesInDirectory(tempRestDataDirForML).count) ?? 0
            
            var descriptionParts: [String] = []
            
            // 1. クラス構成
            descriptionParts.append(String(
                format: "クラス構成: %@: %d枚; Rest: %d枚",
                positiveClassNameForModel,
                positiveCountForDesc,
                restCountForDesc
            ))
            
            // 2. 最大反復回数
            descriptionParts.append("最大反復回数: \(maxIterations)回")
            
            // 3. 正解率情報
            descriptionParts.append(String(
                format: "訓練正解率: %.1f%%, 検証正解率: %.1f%%",
                trainingAccuracy,
                validationAccuracy
            ))
            
            // 4. 陽性クラス情報 (再現率・適合率)
            descriptionParts.append(String(
                format: "陽性クラス: %@, 再現率: %.1f%%, 適合率: %.1f%%",
                positiveClassNameForModel,
                recall * 100,
                precision * 100
            ))
            
            // 5. 検証方法
            descriptionParts.append("(検証: 自動分割)")
            
            let individualDesc = descriptionParts.joined(separator: "\n")
            
            let modelMetadata = MLModelMetadata(
                author: author,
                shortDescription: individualDesc,
                version: version
            )
            
            let modelFileName = "\(modelFileNameBase).mlmodel"
            let modelFilePath = mainRunURL.appendingPathComponent(modelFileName).path
            
            print("💾 OvRペア [\(positiveClassNameForModel)] モデル保存中: \(modelFilePath)")
            try imageClassifier.write(to: URL(fileURLWithPath: modelFilePath), metadata: modelMetadata)
            print("✅ OvRペア [\(positiveClassNameForModel)] モデル保存完了")
            
            return OvRPairTrainingResult(
                modelPath: modelFilePath,
                modelName: modelFileNameBase,
                positiveClassName: positiveClassNameForModel,
                trainingAccuracyRate: trainingAccuracy,
                validationAccuracyRate: validationAccuracy,
                trainingErrorRate: trainingMetrics.classificationError,
                validationErrorRate: validationMetrics.classificationError,
                trainingTime: trainingDurationSeconds,
                trainingDataPath: tempOvRPairRootURL.path,
                recallRate: recall,
                precisionRate: precision,
                individualModelDescription: individualDesc
            )
            
        } catch let createMLError as CreateML.MLCreateError {
            print(
                "🛑 エラー: OvRペア [\(positiveClassNameForModel)] トレーニング/保存失敗 (CreateML): \(createMLError.localizedDescription)"
            )
            print("  詳細情報: \(createMLError)")
            return nil
        } catch {
            print("🛑 エラー: OvRペア [\(positiveClassNameForModel)] トレーニング/保存中に予期しないエラー: \(error.localizedDescription)")
            return nil
        }
    }
    
    // 指定されたディレクトリ内のファイル一覧を取得する
    private func getFilesInDirectory(_ directoryURL: URL) throws -> [URL] {
        try Self.fileManager.contentsOfDirectory(
            at: directoryURL,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: .skipsHiddenFiles
        ).filter { url in
            var isDirectory: ObjCBool = false
            Self.fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory)
            return !isDirectory.boolValue && !url.lastPathComponent.hasPrefix(".")
        }
    }
}
