# 方法與新意審查

查證日期：2026-10-28。這是可實作的課程專題設計，不是新方法或效果已成立的證明。既有研究已分別涵蓋少題適應、empirical prior、文字條件化能力向量、動態選題、領域分配與家族留出；最接近者見 [related_work.md](related_work.md)。

## 可主張什麼

可主張：**以 EmbedLLM 為基礎，實作可適應未見模型的分層文字條件化評分器，並在模型與題目雙重留出的條件下，比較領域配額及參考題池分數不確定性導向的測題策略；同時測試全域／領域偏移與低秩殘差各自的作用。** 這些是具體程式增量與受控實驗。結果出來以前不可宣稱任何方法勝出，也不能宣稱本研究首次提出這些單項元件。

| 候選設計 | 已有的直接前例 | 建議定位 |
| --- | --- | --- |
| 凍結共同網路，以少量答題最佳化新模型向量 | [SkillEval §4.2](https://arxiv.org/html/2610.28767v1#S4.SS2) | 可重現的 EmbedLLM 延伸；與原固定模型 ID 架構相比有工程增量。 |
| 訓練模型學出的 Gaussian 能力先驗 | 同上；其先驗是訓練模型 raw profile 的多變量 Gaussian | 對照固定先驗和資料估計的 prior；不要宣稱首次 empirical Bayes。若只手選 shrinkage，稱正則化或固定 Gaussian prior。 |
| 全域＋領域殘差＋文字條件化低秩 interaction | 多維 IRT、分層 IRT、[Contextual MIRT](https://arxiv.org/abs/2608.22295) | 可解釋的架構選擇。藉 rank 0、去除領域偏移等消融檢驗增益；領域偏移是統計量，不代表可證明的獨立人類技能。 |
| 均衡配額＋不確定性／Fisher 主動選題 | [AEA](https://proceedings.mlr.press/v267/li25bp.html)、[ATLAS](https://proceedings.mlr.press/v306/li26fx.html)、[two-level test battery](https://doi.org/10.3102/10769986231209447) | 將現有設計原則用於參考題池中各 benchmark 的預測分數，並用同預算的消融評估。不要把「平衡＋uncertainty」稱新理論。 |
| 由 support 集合直接編碼新模型 | AEA 的 Neural Process 已用觀察到的題目與成績集合，預測剩餘題；[Conditional Neural Processes](https://arxiv.org/abs/1807.01613)早已有一般方法 | 不優先投入；如果實作，應比較與 MAP 的準確度、適應時間與模型數敏感度，不可說首次以集合做冷啟動。 |
| 以新模型能力向量做路由 | [UniRoute](https://arxiv.org/html/2502.08773v2)、[SkillEval](https://arxiv.org/html/2610.28767v1)、[ContextualRouter](https://aclanthology.org/2026.eacl-srw.22/) | 作可選下游示範；沒有經核實價格就只做正確率／預設相對代價，不稱真實成本節省。 |

## 與 SkillEval 的具體差別

[SkillEval §4.2 和附錄 G](https://arxiv.org/html/2610.28767v1#S4.SS2)已從訓練模型向量估計 Gaussian 先驗，凍結 text-conditioned cognitive diagnosis network，再以 sum likelihood 加 Gaussian prior 適應新模型；它也用未見模型與保留題目計算 benchmark macro MAE。這些共同點必須明寫。

差異在於研究設計和結構：SkillEval 將題目映射到有名稱的技能，使用神經認知診斷模型與多變量相關先驗；本專題使用 benchmark 偏移與未命名的低秩互動，凍結網路後的 logit 對新模型向量是線性的，MAP 為凸最佳化。SkillEval 的選題規則是 `p_q(1-p_q)` 乘該題對應的技能數；此處提出的近似規則則使用 MAP 的曲率與訓練題池中的領域分數梯度。這些差異不保證效果，也不等於首次應用 Bayesian experimental design。

SkillEval 說明其技能分類依完整題庫文字建構，包含最終測試題文字，且未用測試題作答擬合 item parameters；其作者稱此設定對技能表示是 transductive（[附錄 O](https://arxiv.org/html/2610.28767v1)）。本專題選題階段只看訓練／support 池文字，預測階段才讀最終 target 題文字。兩個專案也使用不同資料來源與模型數；不能把各自論文數字當成直接比較。

## 與原 EmbedLLM 的數學對照

原作者的[模型程式](https://raw.githubusercontent.com/richardzhuang0412/EmbedLLM/main/algorithm/mf.py)將題目文字向量 `x` 經單層線性 `Wx+b`，與模型向量 `z` 逐元素相乘，再輸出兩類 logits。兩類 logits 的差等價於：

```text
original_logit = [(w_1 − w_0) ⊙ (W x + b)]ᵀ z + (b_1 − b_0)
```

此原式沒有獨立的題目文字難度 head，也沒有明確的模型全域與 benchmark 偏移。若比較器使用 GELU 非線性題目網路、獨立題目難度 head 或 benchmark bias，應稱 **EmbedLLM-inspired low-rank control**，不能稱為逐式重現。若另做忠實原式的冷啟動 baseline，仍需在各訓練模型 split 上重訓，凍結共享層，再以相同 support 答題適應新向量；它是原模型的冷啟動延伸而非原論文完整實驗的直接重現。

## 一個可測量的選題規則

以下是依一般 Bayesian optimal design 與 CAT 推出的**局部近似**，不是前述論文的逐字重現。設訓練好的題目網路及 decoder 已凍結，對新模型有參數向量 `η`，題目 `q` 的 logit 可寫成

```text
ℓ_q(η) = b_q + c_qᵀη
p_q(η) = sigmoid(ℓ_q(η))
```

例如 `η = [全域偏移, 各 benchmark 偏移, 低秩向量]`，`c_q = [1, benchmark one-hot, v(q)/sqrt(rank)]`；`b_q` 和 `v(q)` 由凍結的題目網路給出。rank 為 0 時省略最後一段，不除以零。

對已揭露的 support 集合 `S`，以固定 Gaussian prior `N(μ, Λ⁻¹)` 求 MAP：

```text
η̂ = argmin_η  Σ_{q ∈ S} BCEWithLogits(ℓ_q(η), y_q)
                   + ½(η−μ)ᵀΛ(η−μ)
H = Λ + Σ_{q ∈ S} w_q c_q c_qᵀ,   w_q = p_q(1−p_q)
Σ = H⁻¹
```

此處是 **sum BCE**，因此 `Λ` 是一個固定 prior precision。若用 mean BCE，正則項必須除以 probe 數才會和同一個固定 prior 對應；直接在不同 `k` 下維持相同 mean-BCE penalty 代表改變 prior 相對於資料的強度。`Λ` 應對稱正定，可用對角／isotropic 基線；若由訓練模型向量估協方差，使用 shrinkage 和數值下限，並只在訓練／驗證資料選設定。向量尺度、旋轉與冗餘的全域／領域偏移可能影響 raw covariance；`rank 0` 和先驗敏感度應分開報告。

對每個保留 benchmark `d`，事先以訓練／support 題池文字選定不看新模型正誤標籤的參考題集合 `R_d`。定義參考題池預測分數 `s_d(η) = mean_{t ∈ R_d} p_t(η)`；在目前 `η̂` 處的梯度為

```text
g_d = mean_{t ∈ R_d} [p_t(1−p_t)c_t]
Var(s_d | S) ≈ g_dᵀΣg_d
```

如果下一題 `q` 的 Bernoulli outcome 依模型產生，以其 Fisher information `w_q c_q c_qᵀ` 近似新增訊息，且暫時不重新線性化，Sherman–Morrison 更新給出估計的分數方差下降：

```text
Δ_d(q) ≈ w_q (g_dᵀΣc_q)² / (1 + w_q c_qᵀΣc_q)
score(q) = Σ_d α_d Δ_d(q),   α_d = 1 / D  # 各 benchmark 等權
```

這個 `score` 針對各 benchmark **參考題池預測分數**的不確定性，而不是只找 `p_q ≈ 0.5` 的題目。參考題池與最終 target 題的分布接近時，這個量才有機會成為 target score uncertainty 的有效代理；兩者不等同。它以等權 domain 方差為近似目標；主要指標若是 macro MAE，兩者並不完全相同，選題也不保證降低實測 MAE。要最小化加權總分的方差，應先將 `g = Σ_d α_d g_d`，再計算 `gᵀΣg`，而不是把各 domain 的方差直接相加。

執行方式：

1. 依允許的資料訓練共同題目網路、決定先驗與固定 train／validation／test 切分。
2. 對新模型以先驗初始化 `η̂`；評估可使用預先固定的 warm-start probes。
3. 計算 `H` 與 `g_d`。用 Cholesky 線性求解，不必顯式求逆；僅在候選 support 題上算 `score(q)`。
4. 如需均衡，依事先固定的 benchmark 配額限制 eligible 題目，再選最高分；全域版本不加配額。候選子抽樣、平手規則與每次更新頻率應事先固定。
5. 揭露被選中題目的單一標籤，再重新適應 `η̂` 並更新 `H`、`g_d`；重複到 budget。完整記錄序列、缺失與運算時間。若採 batch 更新，要說明這是近似批次適應，而非逐題重算。

這個近似把 decoder 與 item network 當作完全已知，假設目前參數附近可線性化，且把題目作答視為條件獨立；它忽略共享網路、題目分布、資料處理和模型錯誤的不確定性。選題以訓練／support 池的參考題文字和 benchmark 標籤計算梯度；選題與適應完成之前不能使用最終 target 題文字或 correctness。適應完成後，預測器才讀取 target 題目文字產生逐題預測；最後僅評分程序讀 target correctness。若參考題與目標題分布有差，選題指標可能無法代表真正的目標誤差。

## 必要消融與檢查

- 同一評分器、切分與 probe budget 比較：全域隨機、領域均衡隨機、全域 Fisher score、領域均衡 Fisher score。隨機與 adaptive 選題可能得到不同 support；每個策略內的預測器必須共用同一批已選題，且跨策略對同一模型／target 配對評分。
- 固定選題比較 rank 0 與 rank > 0；再比較移除領域偏移或不同 prior。不要同時換選題、decoder 和 split，然後把差距歸因於單一元件。
- 主指標為每模型每 benchmark 的 score MAE 再宏平均；同報逐 benchmark、最差領域、micro MAE、逐題 Brier、探測數及時間。用模型或家族作獨立分析單位；家族太少時呈現逐家族結果，避免虛假的窄區間。
- 在合成的正定 `H` 上，比較公式的方差差值與直接計算 `(H+wccᵀ)⁻¹` 的結果；檢查非負性、候選不得重複、query 标签不可被選題器讀到，以及配額恰好滿足。
- `p(1−p)` 反映 Bernoulli 響應的局部資訊權重；其大不等同完整的 epistemic uncertainty。上述 Laplace covariance 也不能自動當成經校準的置信區間；若報區間，須在獨立驗證與各留出家族檢查覆蓋率和寬度。
- 評估家族與驗證家族不得同源；若原始資料不足以證實 lineage 或 prompt 的底題分組，限制結論並列出證據缺口。
