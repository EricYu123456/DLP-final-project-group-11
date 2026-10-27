# 相關研究與可用比較方法

查證日期：2026-10-28。此文件整理已讀的原始論文與作者程式；沒有把文獻中的數字當成本專題結果。本專題以少量作答估計未見模型在保留題集的表現，並以 EmbedLLM 為起點。

## 與研究問題最接近的工作

| 研究 | 已做的事 | 對本專題的意義 |
| --- | --- | --- |
| [EmbedLLM，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/bf5e4b85d203481d6e37bd32d9600162-Abstract-Conference.html) | 從模型作答與題目文字學模型向量，預測逐題正誤、執行路由、預測 benchmark 表現；主要架構以模型 ID 查表。 | 研究起點；新模型的少量作答適應是對原實作的延伸，不可據此宣稱整個領域第一次冷啟動。 |
| [tinyBenchmarks，ICML 2024](https://proceedings.mlr.press/v235/maia-polo24a.html) | 以代表題與 IRT 等方法估計完整測驗分數，也研究評測樣本少時的分布移動。 | 代表題、少題估分與 IRT 都有先例。[官方程式](https://github.com/felipemaiapolo/tinyBenchmarks)提供估分及建立新子集的工具；必須在本專題的訓練模型上重新建立。 |
| [metabench，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/4ebc26584810a189ef1e4f173aba4319-Abstract-Conference.html) | 建立六項測驗的精簡題庫，重建各項與總分，並討論能力估計與區間。 | 少題預測多項 benchmark 成績及分數區間已有先例。[程式](https://github.com/adkipnis/metabench)的現成題庫與參數不一定符合本研究切分。 |
| [Active Evaluation Acquisition，ICML 2025](https://proceedings.mlr.press/v267/li25bp.html) | 以 Neural Process 利用已回答題目預測其他題目的成績，比較全域／分層隨機、題目分群、不確定性、資訊增益與學得的動態策略。 | 均衡、語意多樣性、主動選題與用集合編碼少量作答都已有直接前例。原文的 stratified sampling 就是讓不同資料集均衡出題；純文字分群或不確定性選題不能假定一定勝過隨機。 |
| [Reliable and Efficient Amortized Model-based Evaluation，ICML 2025](https://proceedings.mlr.press/v267/truong25c.html) | 以題目內容預測 IRT 難度，支援低成本校準與適應式測驗。 | 題目文字到 item parameter 的遷移已有先例；[作者程式](https://github.com/aims-foundations/reeval)可供參考，不等同於本專題的模型側向量適應。 |
| [ATLAS，ICML 2026](https://proceedings.mlr.press/v306/li26fx.html) | 用 IRT 與 Fisher information 逐題選測，並重建測驗分數；[§4.6](https://arxiv.org/html/2511.04689v3#S4.SS6)在 ARC 排除整個 Mixtral 家族後測試。 | Fisher 主動測題與整家族留出都有先例。[作者程式](https://github.com/Peiyu-Georgia-Li/ATLAS)包含 R 的 3PL、CAT 與 p-IRT 流程，依賴 R 及額外資料。 |
| [SkillEval，2026-10-20 預印本](https://arxiv.org/html/2610.28767v1#S4.SS2) | 以題目文字條件化認知診斷模型；凍結共同網路後，用訓練模型能力向量擬合的多變量 Gaussian prior 和少量答題估計未見模型向量；在另留出的題目上評分，並做選題及路由。 | 與「雙重留出、少量作答、empirical prior、能力向量及多 benchmark 誤差」高度重疊；目前只稱預印本。[程式庫](https://github.com/berkearda/skilleval)含新模型 profiling 和 baselines，但 README 稱訓練權重尚待發布。 |
| [Contextual MIRT，2026-08 預印本](https://arxiv.org/abs/2608.22295) | 把題目內容與多維 IRT 結合，預測未見題目的模型表現，並指出跨情境泛化未必可靠。 | 非常接近的文字條件化多維潛在能力模型；本次未核實作者公開可執行程式。 |
| [UniRoute，ICLR 2026](https://arxiv.org/html/2502.08773v2) | 以少量代表題的答題錯誤向量表示新模型，用分群或學得的題目分群 map 做動態路由；公開 arXiv v2 的 EmbedLLM 實驗為隨機模型切分。 | 少量 probes 表示新模型及動態模型池路由已有先例；若只實作題群平均，不應稱完整重現 UniRoute。 |
| [ContextualRouter，EACL 2026 Student Research Workshop](https://aclanthology.org/2026.eacl-srw.22/) | 用語意相似題目的歷史作答預測；其簡單 KNN 可有競爭力。 | 值得加入題目近鄰 baseline，但只能使用新模型已揭露的 support 標籤。 |
| [A Two-Level Adaptive Test Battery，2024](https://doi.org/10.3102/10769986231209447) | 對相關子測驗，在子測驗間與各子測驗內分別做 Bayesian adaptive selection，以估計多維分數。 | 依不確定性分配測題到不同領域、利用領域相關性，已有心理計量學前例，不能當作首創。 |

## 強 baseline 與資訊邊界

**主要設定：同時留出模型與題目。** 訓練時所有模型的 target 題真實標籤都不可用。新模型只能揭露選中的 support 標籤；評分時才讀 target 標籤。選題器只使用訓練／support 池已知的題目文字與領域，不能讀取 target 題目文字或標籤。選題與適應完成後，預測器才使用 target 題目文字，產生逐題預測。

建議在同一支測試模型、同一 target 題、同一 probe budget 下比較：

1. **訓練模型領域平均**：零 probe。目標題未知時，從訓練題與訓練模型估計。
2. **平滑的領域 probe 答對率**：同一批 support 題，向訓練模型的領域平均收縮。非隨機選題或 support／target 分布不同時是 heuristic，不保證無偏。
3. **題目 KNN**：用 target 題文字找新模型已答的相似 support 題，聚合其標籤。相似度與 K 在驗證模型決定；不可引用新模型未揭露的題目標籤。
4. **原型冷啟動 text-MF**：在訓練模型重訓 EmbedLLM 類文字模型，凍結共享參數後，只以相同 support 適應新向量；先驗和最佳化規則在驗證模型決定。
5. **全域／領域偏移，無低秩殘差**：固定適應與選題規則，設 rank 0。此項能檢驗新增 latent 向量是否超過領域分數表。
6. **全域／領域偏移加低秩殘差**：本專題新增模型。每種選題策略應同時餵給可支援的各種預測器；不要只把最佳選題配給主方法。

普通 item-ID IRT 與 model-KNN 要直接預測 target 題時，通常需要舊模型在 target 題的歷史作答或由其估出的 item parameters。它們不能偷偷加入上述嚴格設定。若允許訓練模型看到完整固定題庫的標籤，可以另做**固定題庫設定**，重新擬合 IRT、model-KNN 或群集方法，獨立列一張結果表。文字條件化 IRT 可用於未見題，但其實作與調參資訊必須同樣受限。

[tinyBenchmarks 官方 README](https://raw.githubusercontent.com/felipemaiapolo/tinyBenchmarks/main/README.md)明示公開 tiny 題庫從五個隨機種子中挑選在其 test set 平均表現最好者，且採隨機切分。不能直接把該公開子集和其現成參數視為本專題 family-holdout 的無洩漏比較；應只用本專題訓練／驗證資料重新建立。

## 程式與資料的來源

- [EmbedLLM 資料卡](https://huggingface.co/datasets/RZ412/EmbedLLM/raw/main/README.md)標示資料集為 Apache-2.0；研究應固定資料 revision 並保留來源標示。先前下載紀錄使用 revision `266ae3e180ccb533c86afc9cfb2f5db99d171aea`。
- [EmbedLLM GitHub 根目錄](https://github.com/richardzhuang0412/EmbedLLM)在本次檢查時未顯示 LICENSE 檔或授權聲明；資料集授權不能自動推定為程式碼授權。以論文數學描述獨立撰寫程式並引用來源，避免直接複製未核實授權的程式。
- [tinyBenchmarks](https://github.com/felipemaiapolo/tinyBenchmarks/blob/main/LICENSE) 與 [SkillEval](https://github.com/berkearda/skilleval/blob/main/LICENSE) 程式庫的 LICENSE 為 MIT；[metabench](https://github.com/adkipnis/metabench/blob/main/LICENSE) 程式庫標示 CC BY-NC-SA 4.0。[ATLAS](https://github.com/Peiyu-Georgia-Li/ATLAS) 和 [REEval](https://github.com/aims-foundations/reeval) 根目錄在本次檢查時未顯示 LICENSE。上述是特定程式庫當時的可見標示，引用不代表本專題已複製或執行其程式。

查證時 `git ls-remote` 所見的程式庫 HEAD，可用於追查本文檢查的版本：

| 程式庫 | 2026-10-28 查得的 commit |
| --- | --- |
| EmbedLLM | [`9b276305`](https://github.com/richardzhuang0412/EmbedLLM/tree/9b276305af1342188c2fc066e41d5dc8bdfea895) |
| tinyBenchmarks | [`e9a8b103`](https://github.com/felipemaiapolo/tinyBenchmarks/tree/e9a8b1031b0340571beb6c9ca3a27891be09a8fd) |
| SkillEval | [`319fc54c`](https://github.com/berkearda/skilleval/tree/319fc54c54762879f7219447d5a006699735364e) |
| ATLAS | [`c0bc19b3`](https://github.com/Peiyu-Georgia-Li/ATLAS/tree/c0bc19b33aaa11703b8258377abb367533b8a824) |
| REEval | [`726cfe1c`](https://github.com/aims-foundations/reeval/tree/726cfe1ccedaa5576073665d9cda812d20a910c5) |

## 需要記錄的限制

- 全量原始 CSV 稽核（[資料來源與稽核摘要](data_sources.md)）記錄 4,586,288 列、553,616 個有重複紀錄的模型／prompt 組合，以及其中 77,581 個有 0／1 衝突的組合。112×35,673 個模型／prompt 組合各至少有一列，公開標籤矩陣的每格都等於該組原始標籤的最大值。這是實檔的相等關係，未證實上游形成原因或評分條件。矩陣目標與歧義處理必須隨結果報告，不能把該聚合值說成單次作答真值。
- 若以所有保留模型的紀錄存在與否、列數和類別數先建立共同單列題集，雖不使用正誤值，仍依賴未見模型的完整紀錄結構。這是回溯式完整案例分析；事先不知道真正新模型的完整紀錄狀態，缺漏機制也可能改變保留題集。必須揭露這項篩選、各領域刪除比例和對可部署性的限制。
- 評估的領域是 benchmark 來源與其題集，並不自動等同真正的人類技能。MMLU 子科目與 benchmark 來源的權重應事先定義。
- 題目文字向量若保留了示例答案、標準答案或其他不該提供的資訊，可能讓測試不符合部署條件。需要稽核可取得的文字格式。
- 若題目底題或模型血緣只能部分核實，應列出排除和不確定範圍；不能只憑不同 prompt ID、模型名稱或相似字串保證完全分離。
- 基於少量歷史模型的離線模擬，不能直接推論未來任意 LLM、全新 benchmark 或真實線上成本。
