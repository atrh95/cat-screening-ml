import BinaryClassification
import CreateML
import CreateMLComponents
import CSInterface
import Foundation
import MultiClassClassification
import MultiLabelClassification
import OvRClassification

// --- トレーナータイプの定義 ---
enum TrainerType {
    case binary
    case multiClass
    case multiLabel
    case ovr

    var definedVersion: String {
        switch self {
            case .binary: "v5"
            case .multiClass: "v3"
            case .multiLabel: "v1"
            case .ovr: "v3"
        }
    }
}

// --- トレーニング設定 ---
let currentTrainerType: TrainerType = .binary
let maxTrainingIterations = 11

// --- 共通のログデータ設定 ---
let modelAuthor = "akitora"
let modelShortDescription = "ScaryCatScreener Training"
let modelVersion = currentTrainerType.definedVersion
let modelName = "ScaryCatScreeningML"

print("🚀 トレーニングを開始します... 設定タイプ: \(currentTrainerType), モデル名: \(modelName), バージョン: \(modelVersion)")

// トレーナーの選択と実行
let trainer: any ScreeningTrainerProtocol
var trainingResult: Any?

switch currentTrainerType {
    case .binary:
        let binaryTrainer = BinaryClassificationTrainer()
        trainer = binaryTrainer
    case .multiClass:
        let multiClassTrainer = MultiClassClassificationTrainer()
        trainer = multiClassTrainer
    case .multiLabel:
        let multiLabelTrainer = MultiLabelClassificationTrainer()
        trainer = multiLabelTrainer
    case .ovr:
        let ovrTrainer = OvRClassificationTrainer()
        trainer = ovrTrainer
}

trainingResult = await trainer.train(
    author: modelAuthor,
    modelName: modelName,
    version: modelVersion,
    maxIterations: maxTrainingIterations
)

// 結果の処理
if let result = trainingResult {
    print("🎉 トレーニングが正常に完了しました。")

    // 結果をログに保存 (TrainingResultDataプロトコルのsaveLogメソッドを利用)
    if let resultData = result as? any TrainingResultProtocol {
        resultData.saveLog(
            modelAuthor: modelAuthor,
            modelName: modelName,
            modelDescription: modelShortDescription,
            modelVersion: modelVersion
        )
        print("💾 トレーニング結果をログに保存しました。")
    } else {
        print("⚠️ 結果の型がTrainingResultDataに準拠していません。ログは保存されませんでした。")
    }
} else {
    print("🛑 トレーニングまたはモデルの保存中にエラーが発生しました。")
}

print("✅ すべての処理が完了しました。")
