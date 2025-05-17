# OvR (One-vs-Rest) トレーニング実行レポート

## 実行概要
モデル群         : OvRモデル群 (One-vs-Rest)
レポート生成日時   : 2025-05-17 12:05:58 +0900
最大反復回数     : 11 (各ペアモデル共通)
## 個別モデルのパフォーマンス指標
| モデル名 (PositiveClass) | 検証正解率 | 検証再現率 | 検証適合率 |
|--------------------------|--------------|--------------|--------------|
| ScaryCatScreeningML_OvR_Sphynx_vs_Rest_vv3_idx0 (Sphynx) | 87.50% | 87.50% | 87.50% |
| ScaryCatScreeningML_OvR_HumanHandsDetected_vs_Rest_vv3_idx1 (HumanHandsDetected) | 93.75% | 100.00% | 88.89% |
| ScaryCatScreeningML_OvR_BlackAndWhite_vs_Rest_vv3_idx2 (BlackAndWhite) | 77.27% | 81.82% | 75.00% |
| ScaryCatScreeningML_OvR_MouthOpen_vs_Rest_vv3_idx3 (MouthOpen) | 75.00% | 83.33% | 71.43% |

## 共通メタデータ
作成者            : akitora
全体説明          : ScaryCatScreener Training (このOvR実行全体に対して)
バージョン        : v3