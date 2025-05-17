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

public struct OvRTrainingResult: TrainingResultProtocol {
    public let modelOutputPath: String
    public let trainingDataPaths: String
    public let maxIterations: Int
    public let individualReports: [IndividualModelReport]

    public func saveLog(modelAuthor: String, modelName _: String, modelVersion: String) {
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd HH:mm:ss Z"
        let generatedDateString = dateFormatter.string(from: Date())

        let reportFileName = "OvR_Run_Report_\(modelVersion).md"
        let modelDir = URL(fileURLWithPath: modelOutputPath)
        let reportURL = modelDir.appendingPathComponent(reportFileName)

        var markdownText = """
        # OvR (One-vs-Rest) トレーニング実行レポート

        ## 実行概要
        モデル群         : OvRモデル群 (One-vs-Rest)
        レポート生成日時   : \(generatedDateString)
        最大反復回数     : \(maxIterations) (各ペアモデル共通)
        """

        if !individualReports.isEmpty {
            markdownText += """

            ## 個別モデルのパフォーマンス指標
            | モデル名 (PositiveClass) | 検証正解率 | 検証再現率 | 検証適合率 |
            |--------------------------|--------------|--------------|--------------|
            """
            for report in individualReports {
                let modelNameDisplay = "\(report.modelName) (\(report.positiveClassName))"
                let valAccStr = String(format: "%.2f%%", report.validationAccuracyPercentage)
                let recallStr = String(format: "%.2f%%", report.recallRate * 100)
                let precisionStr = String(format: "%.2f%%", report.precisionRate * 100)

                markdownText +=
                    "\n| \(modelNameDisplay) | \(valAccStr) | \(recallStr) | \(precisionStr) |"
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
            print("✅ OvR実行レポートを保存しました: \(reportURL.path)")
        } catch {
            print("❌ OvR実行レポートの保存エラー: \(error.localizedDescription) (Path: \(reportURL.path))")
        }
    }
}
