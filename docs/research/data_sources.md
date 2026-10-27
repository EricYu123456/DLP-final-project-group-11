# EmbedLLM：資料來源、標籤與模型血緣

查證日期：2026-10-28（臺灣時間）。本文件記錄公開檔案與已查證的來源；完整 CSV 稽核結果見 [`reports/data_audit/summary.json`](../../reports/data_audit/summary.json)。本文所稱「模型家族」是為切分資料而定義的保守系列區塊，不是對所有成員共享相同權重的主張。

## 來源與版本

- [EmbedLLM 官方論文](https://proceedings.iclr.cc/paper_files/paper/2025/file/bf5e4b85d203481d6e37bd32d9600162-Paper-Conference.pdf)第 4.2 節說，作者使用 `lm-evaluation-harness` 評估 112 個模型、10 個 benchmark，為模型與題目的配對建立二元 correctness label。論文列出 36,054 題；目前公開檔案的題目數如下。兩者不能假定為同一版前處理。
- 本研究固定 [Hugging Face 資料集 revision `266ae3e180ccb533c86afc9cfb2f5db99d171aea`](https://huggingface.co/datasets/RZ412/EmbedLLM/tree/266ae3e180ccb533c86afc9cfb2f5db99d171aea)。[資料卡](https://huggingface.co/datasets/RZ412/EmbedLLM/blob/266ae3e180ccb533c86afc9cfb2f5db99d171aea/README.md)將三份 CSV 描述為 correctness matrix，並列出模型、題目及向量檔。程式碼參考 [官方 repository commit `9b276305af1342188c2fc066e41d5dc8bdfea895`](https://github.com/richardzhuang0412/EmbedLLM/tree/9b276305af1342188c2fc066e41d5dc8bdfea895)。
- 公開 `model_order.csv` 有 112 個模型；`question_order.csv` 有 35,673 個 prompt ID；`question_embeddings.pth` 的 shape 為 `(35673, 768)`。公開 `train_y.pth` 為 `(112, 29673)`，`val_y.pth`、`test_y.pth` 各為 `(112, 3000)`。這些數字來自本地已下載檔案的觀察，原始 label tensor 值為 0/1。三個公開 split 的 prompt ID 不重疊，但下文說明底題仍會重複。
- 論文使用 `all-mpnet-base-v2` 產生 768 維題目向量。可快取公開題目向量，但不得把包含全部模型監督資訊的公開模型向量當作未見模型的初始化。[資料卡](https://huggingface.co/datasets/RZ412/EmbedLLM/blob/266ae3e180ccb533c86afc9cfb2f5db99d171aea/README.md)還說 `optimal_mf_model_embeddings.pth` 是按測試準確率選出的模型；它不適合當本專案的無洩漏預訓練結果。

## 標籤實際代表什麼

論文將 `1`、`0` 定義為評測得到的正確與否；公開 CSV 包含 `prompt_id`、`model_id`、`category_id`、`label`、`prompt`、`model_name`、`category`。公開欄位沒有模型輸出文字、原 benchmark item ID、metric/filter、解碼設定、harness revision 或推論失敗狀態。不同 task 可採不同答案評分流程，不能將所有標籤稱為同一種自由文字回覆的判分，也不能只憑 `0` 排除上游缺失被填零的可能。

因此，離線實驗以**公開二元標籤作為代理目標**。給新模型 50、100、200 個標籤是遮住既有資料後的模擬；若尚未實際呼叫新 LLM 與實作同條件評分，不能宣稱完成線上端到端推論。各 benchmark 的報告值是本研究保留題目的平均代理標籤，不自動等於官方 leaderboard 分數。

### 完整 CSV 稽核與尚待確認的原因

完整掃描三份 CSV 共 4,586,288 列，涵蓋全部 112 × 35,673 個模型與 prompt 配對；依 CSV 列存在性檢查，沒有缺少的配對。共有 553,616 個配對有重複列，77,581 個配對出現不同標籤，其中 75,904 個在同一 category 內也有衝突。這裡的「CSV 無缺列」不能證明每次上游模型推論都成功，因為未公開推論狀態。

完整稽核也顯示：每個已觀察配對的公開 y 值，都等於該配對原始列標籤的最大值，總計 0 個不一致。這是對此發布版本的可重現觀察；公開檔案沒有足夠資訊證明這些列是否代表多次作答、不同評分 filter，或原始生成流程為何。

第一個模型 `model_id=0` 的細分數字如下，可與先前僅下載 CSV 前綴時的檢查對照：

| 原 split | CSV 列數 | 不重複 prompt ID | 同模型、同 prompt 出現不同標籤的組數 |
| --- | ---: | ---: | ---: |
| train | 34,056 | 29,673 | 784 |
| validation | 3,424 | 3,000 | 69 |
| test | 3,469 | 3,000 | 85 |

GPQA 有跨 category 的相同 prompt，也有同 category、同 prompt 的 0/1 衝突；全部模型合計 71,517 個 GPQA 配對有不同標籤。GSM8K 的每個模型／prompt 配對有兩列，共 4,403 個配對有不同標籤。[目前的 harness GSM8K 設定](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k.yaml)有多種答案擷取 filter；它提供一種可能的解釋，但資料未列使用的版本和 filter，不能證實原因。

另須區分官方程式的 router 資料處理與已發布 y tensor 的生成方式。[官方 `mf.py`](https://github.com/richardzhuang0412/EmbedLLM/blob/9b276305af1342188c2fc066e41d5dc8bdfea895/algorithm/mf.py#L192-L213) 的 router 整理程序會以後一列覆寫同一 `(prompt, model)` 的前一列，並對缺少的模型使用 `0`；這不能證明公開 y tensor 是如何製作的。本專案應保留存在性遮罩，避免把缺失自動視為答錯。

## 題目範圍與切分

以 MMLU、MedMCQA、MathQA、TruthfulQA 和 LogiQA 當首批候選來源。完整 CSV 稽核顯示，這五者合計 22,651 個公開 prompt ID；要求所有 112 個模型各有一列後為 22,624 個，包括 MMLU 13,988、MedMCQA 4,183、MathQA 2,985、TruthfulQA 817、LogiQA 651。相同的單列結構也適用於表內納入的 88 個模型。依版本 2 的保守題組規則，每組取最小 prompt ID 作代表後剩 22,475 題；這是**代表題子集**，其分數的母體不等於全部公開 prompt 或原 benchmark。規則應先於檢視預測成效固定：

1. 對選定的模型，逐模型逐 prompt 核對原始 CSV 的存在性、category 和列數。主子集只保留每個納入模型都恰有一筆、且 category 可唯一識別的 prompt；重複列即使標籤相同也排除。此規則依列結構選題，不根據 held-out label 值決定是否納入。
2. 題組依內容而非原 split 重切。MMLU 移除科目提示前綴，並以正規化後的完整題幹進行保守分組；MedMCQA 和 LogiQA 同樣保留完整題幹及文章。完整選項內容另可用於檢查候選匹配。TruthfulQA 取最後一個目標問題，避免將固定示例當成目標；MathQA 公開 prompt 未包含選項，故只能以題幹保守分組，不能重建完整原始題目。相同題幹可能搭配不同選項，也可能只是通用問句；強制同組會過度合併部分無關題目，但可降低把同一底題分到兩側的風險。
3. 每種方法都使用相同的 grouped split、同一批保留模型及 benchmark 題目。保存 ID、分組結果、缺失與排除原因，並報告選入比例。讀取 row existence 本身也會影響目標分布，因此應公開此選取規則與各模型覆蓋率。
4. 暫不將 GPQA 和 GSM8K 放入主子集：前者存在大量提示及 category 變體，後者評分 filter 來源尚不可核對。它們可以作為清楚標示目標定義的敏感度分析；不能把不同條件的 label 平均或合併成單次作答真值。

模型 0 的 prompt 文本與 category 索引提供了具體洩漏例子：移除 MMLU 科目標頭後，以完整題幹及排序選項比對，14,015 個 MMLU prompt 形成 79 個重複候選組，其中 24 組跨越公開 train/validation/test。例如 prompt 3538（train，college medicine）與 20614（test，clinical knowledge）的題幹及選項相同。僅比對題幹也可找出 prompt 12185（train）與 12293（test），兩者選項僅差句尾標點；因此精確選項鍵仍會漏掉它們。MMLU、MedMCQA、LogiQA 有 51 個同題幹但選項字串不同的候選組，其中 20 組跨越公開 split；部分同題幹只是通用問句，不能全部當成確認的重複題。探索性解析器在取最後四個選項標記後可解析所有 14,015 筆 MMLU prompt；但字串規則仍需抽查，也無法排除改寫或相同來源題的其他未匹配版本。探索記錄保留於本機；可重現的正式分組實作見 [data.py](../../src/coldstart/data.py)。

額外的來源對齊顯示，公開版本的 4,183 個 MedMCQA prompt，按正規化題幹與四個選項比對，都能唯一對上 [MedMCQA validation split](https://huggingface.co/datasets/openlifescienceai/medmcqa/tree/91c6572c454088bf71b679ad90aa8dffcd0d5868) 的 4,183 個 item；並非單靠題數相等推論。映射與來源版本在 [`medmcqa_source_ids.csv`](../../data/metadata/medmcqa_source_ids.csv)，不包含題目文字。論文對所有 benchmark 一般性描述為 test sets；這項特定公開版本的觀察應另外報告，不推斷其他來源的 split。

14,015 個 MMLU prompt 依科目、正規化題幹和原順序的四個選項，也全部對上 [`cais/mmlu` 的 `all/test` 14,042 列](https://huggingface.co/datasets/cais/mmlu/tree/c30699e8356da336a370243923dbaf21066bb9fe)：其中 27 個 prompt 各對到兩筆相同內容的來源列。這些重複來源列的 answer key 彼此相同，無法單靠它們解釋 CSV 中兩個 MMLU 標籤衝突。對齊表 [`mmlu_source_rows.csv`](../../data/metadata/mmlu_source_rows.csv) 不包含題目文字；其中的來源列號是固定 revision、config 和 split 內的 viewer row index，不是原資料提供的永久 item ID。

準備好本專案資料後，可執行 [`scripts/research/verify_source_items.py`](../../scripts/research/verify_source_items.py) 重建兩份映射。程式先核對來源 dataset revision 和固定 Parquet 的 SHA-256，再讀取 viewer 資料；若來源 revision 已變更，會停止而非默默使用新資料。

## 模型家族表

[`data/metadata/model_families.csv`](../../data/metadata/model_families.csv) 對 112 個模型保留原 ID、原名稱、系列區塊、納入狀態、已知 parent、來源和判斷。當前 88 個納入、24 個排除；排除的模型在家族留出實驗的訓練、驗證和測試均不可使用。

| 納入的系列區塊 | 模型數 | 證據與邊界 |
| --- | ---: | --- |
| Llama | 36 | 包含 Llama 1/2/3、Vicuna、Code Llama 及其可核對衍生模型。三代合併是保守切分，並非宣稱它們共用同一 checkpoint。 |
| Mistral | 18 | 包含可核對的 OpenChat、Starling、BioMistral 和 SOLAR；[SOLAR 模型卡](https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0)說明使用 Mistral 7B 權重。 |
| Yi | 9 | 包含 Yi、Yi 微調及兩個 parent 均可追到 Yi 的 HermesBagel merge。 |
| Qwen | 6 | 包含 Qwen1.5、Sailor 及可追到 Qwen-72B 的 Smaug 衍生模型；不同代數作保守合組。 |
| DeepSeek | 5 | 將 DeepSeek Coder、Math 和 LLM 置於同一保守區塊，不宣稱 Coder 與獨立預訓練的 LLM 共用權重。 |
| Pythia | 3 | 包含 Dolly 及 OpenAssistant 的 Pythia 衍生模型；deduplicated 與非 deduplicated 版本同組。 |
| Gemma、MPT、Phi | 各 2 | 同一開發者模型系列；不同大小或版本不必共享權重。 |
| BLOOM、Falcon、Palmyra、StableLM、StarCoder | 各 1 | 單模型系列；若用於家族測試，必須報告樣本數。 |

例如，[Llemma](https://huggingface.co/EleutherAI/llemma_7b)明確以 Code Llama 權重初始化，而 [Meta 說明 Code Llama 建立於 Llama 2](https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/)；[Sailor 模型卡](https://huggingface.co/sail/Sailor-7B)列出 Qwen1.5 base。不同 parent 皆屬同一已查證區塊的 merge 可納入；parent 不完整、跨系列 merge 尚未查清，或卡片不足以辨認權重來源者，先排除。模型名稱、上傳者與 `LlamaForCausalLM` 架構本身都不足以判定權重血緣。

`evidence_kind` 區分 `direct_parent`、`multi_step_chain`、`same_family_merge`、`series` 和 `unresolved`。`series` 僅代表模型由開發者歸為該系列，不代表已證明所有組員來自同一 checkpoint。[`family_evidence.csv`](../../data/metadata/family_evidence.csv) 保存每個來源的固定 URL、可取得的 repository revision／時間、抓取時間和原始位元組 SHA-256；非 repository 網頁無 commit 時間。六個模型的最新卡片在資料集發布之後，現已改用可公開取得、且仍支持相同血緣的較早版本。兩個 Meta Llama 3 instruct 模型的歷史紀錄 API 回傳 401，保留已取得的新版官方卡片。這些是**查證用模型卡的版本**，不能證明 EmbedLLM 當初測試的權重 revision。家族留出只能避免表內已知血緣混入，不能保證不同系列的預訓練資料互不重疊。

## 來源授權與再散布

資料集頁面標記 `apache-2.0`，但資料含多個 benchmark 的 prompt；此標記本身不能核實每筆上游素材均可按同一條款再散布。下列是查證時的**來源頁面聲明**，不是對個別題目權利歸屬的判定。程式和文件應保留來源引用；不把原始 CSV、題目文字或公開衍生向量提交到 git。如下若日後再散布，須先逐項確認條件。

| 來源 | 可核對的聲明 |
| --- | --- |
| MMLU | [`cais/mmlu`](https://huggingface.co/datasets/cais/mmlu) 的 Hub metadata 標記 MIT；[原 repository](https://github.com/hendrycks/test/blob/master/LICENSE) 亦有 MIT 授權檔。 |
| MedMCQA | [資料集頁](https://huggingface.co/datasets/openlifescienceai/medmcqa)標記 Apache-2.0；[原 repository](https://github.com/medmcqa/medmcqa)另有 MIT repository 授權，兩者適用範圍不可混為一談。 |
| MathQA | [資料卡](https://huggingface.co/datasets/allenai/math_qa/blob/main/README.md)標記資料為 Apache-2.0。 |
| TruthfulQA | [資料集頁](https://huggingface.co/datasets/truthfulqa/truthful_qa)標記 Apache-2.0；[原 repository](https://github.com/sylinrl/TruthfulQA)有 Apache-2.0 授權檔。 |
| LogiQA | 檢查的 [Hub 資料集 metadata](https://huggingface.co/datasets/lucasmccabe/logiqa) 未列授權；[原 repository](https://github.com/lgw863/LogiQA-dataset) 的 license API 未回傳授權檔。資料再散布條件尚待確認。 |
| PIQA | [Hub 資料集頁](https://huggingface.co/datasets/ybisk/piqa)的授權欄標為 `unknown`，須另行確認。 |
| SocialIQA | 檢查的 [Hub 資料集 metadata](https://huggingface.co/datasets/allenai/social_i_qa) 未列資料授權，須另行確認。 |
| GSM8K | [OpenAI 資料卡](https://huggingface.co/datasets/openai/gsm8k/blob/main/README.md)標記 MIT。 |
| GPQA | [作者的資料集頁](https://huggingface.co/datasets/Idavidrein/gpqa)標記 CC-BY-4.0，並要求不要在網路以文字或圖片公開題目例子；[程式 repository](https://github.com/idavidrein/gpqa)的 MIT 標記不應代替資料條件。 |
| ASDiv | [EleutherAI 資料集頁](https://huggingface.co/datasets/EleutherAI/asdiv)標記 CC-BY-NC-4.0。 |

[all-mpnet-base-v2 模型卡](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)標記模型 Apache-2.0；這不會改變輸入題目的上游條款。各 LLM 權重也有個別條款；本專案如只使用既有結果，不等於已取得所有模型權重的再散布授權。
