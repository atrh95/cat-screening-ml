# OvO (One-vs-One) トレーニング実行レポート

## 実行概要
モデル群         : OvOモデル群 (One-vs-One)
モデルベース名   : ScaryCatScreeningML
レポート生成日時   : 2025-05-17 13:57:05 +0900
総クラス数       : 4
総ペア数         : 6
最大反復回数     : 11 (各ペアモデル共通)
## 個別ペアモデルのパフォーマンス指標
| ペアモデル名 (Class1 vs Class2) | 検証正解率 | モデル説明 |
|---------------------------------|--------------|------------|
| ScaryCatScreeningML_OvO_Sphynx_vs_HumanHandsDetected_vv1_idx0 (Sphynx_vs_HumanHandsDetected) | 1.00% | クラス構成: Sphynx: 80枚; HumanHandsDetected: 193枚
最大反復回数: 11回
訓練正解率: 100.0%, 検証正解率: 100.0%
(検証: 自動分割) |
| ScaryCatScreeningML_OvO_Sphynx_vs_BlackAndWhite_vv1_idx1 (Sphynx_vs_BlackAndWhite) | 0.94% | クラス構成: Sphynx: 80枚; BlackAndWhite: 282枚
最大反復回数: 11回
訓練正解率: 98.5%, 検証正解率: 94.4%
(検証: 自動分割) |
| ScaryCatScreeningML_OvO_Sphynx_vs_MouthOpen_vv1_idx2 (Sphynx_vs_MouthOpen) | 0.91% | クラス構成: Sphynx: 80枚; MouthOpen: 141枚
最大反復回数: 11回
訓練正解率: 100.0%, 検証正解率: 90.9%
(検証: 自動分割) |
| ScaryCatScreeningML_OvO_HumanHandsDetected_vs_BlackAndWhite_vv1_idx3 (HumanHandsDetected_vs_BlackAndWhite) | 0.92% | クラス構成: HumanHandsDetected: 193枚; BlackAndWhite: 282枚
最大反復回数: 11回
訓練正解率: 95.8%, 検証正解率: 91.7%
(検証: 自動分割) |
| ScaryCatScreeningML_OvO_HumanHandsDetected_vs_MouthOpen_vv1_idx4 (HumanHandsDetected_vs_MouthOpen) | 0.88% | クラス構成: HumanHandsDetected: 193枚; MouthOpen: 141枚
最大反復回数: 11回
訓練正解率: 98.7%, 検証正解率: 88.2%
(検証: 自動分割) |
| ScaryCatScreeningML_OvO_BlackAndWhite_vs_MouthOpen_vv1_idx5 (BlackAndWhite_vs_MouthOpen) | 0.90% | クラス構成: BlackAndWhite: 282枚; MouthOpen: 141枚
最大反復回数: 11回
訓練正解率: 99.8%, 検証正解率: 90.5%
(検証: 自動分割) |

## 共通メタデータ
作成者            : akitora
バージョン        : v1