import CSInterface
import Foundation

public struct MultiLabelTrainingResult: TrainingResultProtocol {
    public let modelName: String
    public let trainingDurationInSeconds: TimeInterval
    public let modelOutputPath: String
    public let trainingDataPath: String
    public let classLabels: [String]
    public let maxIterations: Int

    public let meanAveragePrecision: Double?
    public let perLabelMetricsSummary: String?

    public let averageRecallAcrossLabels: Double?
    public let averagePrecisionAcrossLabels: Double?

    public func saveLog(
        modelAuthor: String,
        modelName: String,
        modelVersion: String
    ) {
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd HH:mm:ss Z"
        let generatedDateString = dateFormatter.string(from: Date())

        let classLabelsString = classLabels.isEmpty ? "不明" : classLabels.joined(separator: ", ")
        let durationStr = String(format: "%.2f", trainingDurationInSeconds)

        var metricsLog = ""
        if let map = meanAveragePrecision {
            metricsLog += "平均適合率 (mAP)        : \(String(format: "%.2f", map * 100))%\n"
        }
        if let avgRecall = averageRecallAcrossLabels {
            metricsLog += "平均再現率 (ラベル毎)   : \(String(format: "%.2f", avgRecall * 100))%\n"
        }
        if let avgPrecision = averagePrecisionAcrossLabels {
            metricsLog += "平均適合率 (ラベル毎)   : \(String(format: "%.2f", avgPrecision * 100))%\n"
        }
        if let perLabelSummary = perLabelMetricsSummary, !perLabelSummary.isEmpty {
            metricsLog += "\n## ラベル毎の指標詳細\n\(perLabelSummary.replacingOccurrences(of: "; ", with: "\n"))\n"
        } else if meanAveragePrecision == nil, averageRecallAcrossLabels == nil, averagePrecisionAcrossLabels == nil {
            metricsLog += "詳細なパフォーマンス指標は利用できません。\n"
        }

        let markdownText = """
        # モデルトレーニング情報: \(modelName) (マルチラベル)

        ## モデル詳細
        モデル名           : \(modelName)
        ファイル生成日時   : \(generatedDateString)
        最大反復回数     : \(maxIterations) (注: CreateMLComponentsでは直接使用されません)

        ## トレーニング設定
        元データ(マニフェスト): \(trainingDataPath)
        検出された全ラベル : \(classLabelsString)

        ## パフォーマンス指標
        トレーニング所要時間: \(durationStr) 秒
        \(metricsLog)

        ## モデルメタデータ
        作成者            : \(modelAuthor)
        バージョン          : \(modelVersion)
        """

        let outputDir = URL(fileURLWithPath: modelOutputPath).deletingLastPathComponent()
        let textFileName = "\(modelName)_\(modelVersion).md"
        let textFilePath = outputDir.appendingPathComponent(textFileName).path

        do {
            try markdownText.write(toFile: textFilePath, atomically: true, encoding: String.Encoding.utf8)
            print("✅ [\(modelName) - Multi-Label] モデル情報をMarkdownファイルに保存しました: \(textFilePath)")
        } catch {
            print("❌ [\(modelName) - Multi-Label] Markdownファイルの書き込みに失敗しました: \(error.localizedDescription)")
            print("--- [\(modelName) - Multi-Label] モデル情報 (Markdown) ---:")
            print(markdownText)
            print("--- ここまで --- ")
        }
    }
}
