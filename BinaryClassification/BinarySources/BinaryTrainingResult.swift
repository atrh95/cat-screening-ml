import CSInterface
import Foundation

/// 画像分類モデルのトレーニング結果を格納する構造体
public struct BinaryTrainingResult: TrainingResultProtocol {
    public let modelName: String
    public let trainingDataAccuracyPercentage: Double
    public let validationDataAccuracyPercentage: Double
    public let trainingDataMisclassificationRate: Double
    public let validationDataMisclassificationRate: Double
    public let trainingDurationInSeconds: TimeInterval
    public let trainedModelFilePath: String
    public let sourceTrainingDataDirectoryPath: String
    public let detectedClassLabelsList: [String]
    public let maxIterations: Int

    public func saveLog(
        modelAuthor: String,
        modelName _: String,
        modelVersion: String
    ) {
        // ファイル生成日時フォーマッタ
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd HH:mm:ss Z"
        let generatedDateString = dateFormatter.string(from: Date())

        // resultからクラスラベルを取得 (trainer.classLabelsから変更)
        let classLabelsString = detectedClassLabelsList.isEmpty ? "不明" : detectedClassLabelsList.joined(separator: ", ")

        // 文字列フォーマット
        let trainingAccStr = String(format: "%.2f", trainingDataAccuracyPercentage)
        let validationAccStr = String(format: "%.2f", validationDataAccuracyPercentage)
        let trainingErrStr = String(format: "%.2f", trainingDataMisclassificationRate * 100) // %表示に変更
        let validationErrStr = String(format: "%.2f", validationDataMisclassificationRate * 100) // %表示に変更
        let durationStr = String(format: "%.2f", trainingDurationInSeconds)

        // Markdownの内容
        let infoText = """
        # モデルトレーニング情報

        ## モデル詳細
        モデル名           : \(modelName)
        ファイル生成日時   : \(generatedDateString)
        最大反復回数     : \(maxIterations)

        ## トレーニング設定
        使用されたクラスラベル : \(classLabelsString)

        ## パフォーマンス指標
        トレーニング所要時間: \(durationStr) 秒
        トレーニング誤分類率 : \(trainingErrStr)%
        訓練データ正解率 : \(trainingAccStr)% ※モデルが学習に使用したデータでの正解率
        検証データ正解率 : \(validationAccStr)% ※モデルが学習に使用していない未知のデータでの正解率
        検証誤分類率       : \(validationErrStr)%

        ## モデルメタデータ
        作成者            : \(modelAuthor)
        バージョン          : \(modelVersion)
        """

        // Markdownファイルのパスを作成 (モデルファイルと同じディレクトリ、拡張子を.mdに変更)
        let outputDir = URL(fileURLWithPath: trainedModelFilePath).deletingLastPathComponent()
        let textFileName = "\(modelName)_\(modelVersion).md"
        let textFilePath = outputDir.appendingPathComponent(textFileName).path

        // Markdownファイルに書き込み
        do {
            try infoText.write(toFile: textFilePath, atomically: true, encoding: .utf8)
            print("✅ モデル情報をMarkdownファイルに保存しました: \(textFilePath)")
        } catch {
            print("❌ Markdownファイルの書き込みに失敗しました: \(error.localizedDescription)")
            // 書き込み失敗時はコンソールに出力
            print("--- モデル情報 (Markdown) ---:")
            print(infoText)
            print("--- ここまで --- ")
        }
    }
}
