import CreateML
import Foundation

/// 画像分類モデルトレーナー
public protocol ScreeningTrainerProtocol {
    associatedtype TrainingResultType

    var outputDirPath: String { get }
    var resourcesDirectoryPath: String { get }
    var classificationMethod: String { get }

    /// トレーニング実行
    func train(
        author: String,
        modelName: String,
        version: String,
        modelParameters: CreateML.MLImageClassifier.ModelParameters
    ) async
        -> TrainingResultType?
}

public extension ScreeningTrainerProtocol {
    func createOutputDirectory(
        modelName: String,
        version: String
    ) throws -> URL {
        // 1. モジュール固有のベース出力ディレクトリ
        let moduleSpecificBaseOutputDirURL = URL(fileURLWithPath: outputDirPath)

        // 2. モデル名ディレクトリ (例: .../OutputModels/ScaryCatScreeningML)
        let modelSpecificDirURL = moduleSpecificBaseOutputDirURL.appendingPathComponent(modelName)

        // 3. バージョン別ディレクトリ (例: .../ScaryCatScreeningML/v1)
        let versionedOutputDirURL = modelSpecificDirURL.appendingPathComponent(version)

        try FileManager.default.createDirectory(
            at: versionedOutputDirURL,
            withIntermediateDirectories: true,
            attributes: nil
        )
        // print("📂 バージョン別出力ディレクトリ (親): \(versionedOutputDirURL.path)") // This line will be commented out/removed

        // バージョン別ディレクトリ内の既存の実行をリスト
        let existingRuns = (try? FileManager.default.contentsOfDirectory(
            at: versionedOutputDirURL,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: .skipsHiddenFiles
        )) ?? []

        // 実行名のプレフィックス (例: "Binary_Result_") - バージョンはパスに含まれるため、プレフィックスからは削除
        let runNamePrefix = "\(classificationMethod)_Result_"

        // 次の実行インデックスを計算
        let nextIndex = (existingRuns.compactMap { url -> Int? in
            let runName = url.lastPathComponent
            if runName.hasPrefix(runNamePrefix) {
                return Int(runName.replacingOccurrences(of: runNamePrefix, with: ""))
            }
            return nil
        }.max() ?? 0) + 1

        // 最終的な実行出力ディレクトリURLを構築 (例: .../v1/Binary_Result_1)
        let finalOutputRunURL = versionedOutputDirURL.appendingPathComponent("\(runNamePrefix)\(nextIndex)")

        try FileManager.default
            .createDirectory(at: finalOutputRunURL, withIntermediateDirectories: true, attributes: nil)
        print("💾 結果保存ディレクトリ: \(finalOutputRunURL.path)")

        return finalOutputRunURL
    }
}
