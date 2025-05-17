import CSInterface
import Foundation

public struct IndividualModelReport: Codable, Sendable {
    public let modelName: String
    public let positiveClassName: String
    public let trainingAccuracyRate: Double
    public let validationAccuracyPercentage: Double
    public let recallRate: Double
    public let precisionRate: Double
    public let modelDescription: String
}

public struct OvOTrainingResult: TrainingResultProtocol {
    public let modelOutputPath: String
    public let trainingDataPaths: String
    public let maxIterations: Int
    public let individualReports: [IndividualModelReport]
    public let numberOfClasses: Int
    public let numberOfPairs: Int

    public func saveLog(modelAuthor: String, modelName: String, modelVersion: String) {
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd HH:mm:ss Z"
        dateFormatter.timeZone = TimeZone(identifier: "Asia/Tokyo")
        let generatedDateString = dateFormatter.string(from: Date())

        let reportFileName = "OvO_Run_Report_\(modelVersion).md"
        let modelDir = URL(fileURLWithPath: modelOutputPath)
        let reportURL = modelDir.appendingPathComponent(reportFileName)

        var markdownText = """
        # OvO (One-vs-One) トレーニング実行レポート

        ## 実行概要
        モデル群         : OvOモデル群 (One-vs-One)
        モデルベース名   : \(modelName)
        レポート生成日時   : \(generatedDateString)
        総クラス数       : \(numberOfClasses)
        総ペア数         : \(numberOfPairs)
        最大反復回数     : \(maxIterations) (各ペアモデル共通)
        """

        if !individualReports.isEmpty {
            markdownText += """

            ## 個別ペアモデルのパフォーマンス指標
            | ペアモデル名 (Class1 vs Class2) | 検証正解率 |
            |---------------------------------|--------------|
            """
            for report in individualReports {
                let modelNameDisplay = "\(report.modelName) (\(report.positiveClassName))"
                let valAccStr = String(format: "%.2f%%", report.validationAccuracyPercentage)

                markdownText += "\n| \(modelNameDisplay) | \(valAccStr) |"
            }
            markdownText += "\n"
        }

        markdownText += """

        ## 共通メタデータ
        作成者            : \(modelAuthor)
        バージョン        : \(modelVersion)
        """

        do {
            try markdownText.write(to: reportURL, atomically: true, encoding: .utf8)
            print("✅ OvO実行レポートを保存しました: \(reportURL.path)")
        } catch {
            print("❌ OvO実行レポートの保存エラー: \(error.localizedDescription) (Path: \(reportURL.path))")
        }
    }
} 