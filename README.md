<div align="center">

# WattBot · HERO

### Hybrid Evidence Retrieval Optimization

**Evidence-grounded answers about the environmental impact of AI.**<br>
**從文字、表格與圖像中，找出 AI 環境影響的可追溯證據。**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Notebooks](https://img.shields.io/badge/Jupyter-Research_Notebooks-F37626?style=flat-square&logo=jupyter&logoColor=white)
![Retrieval](https://img.shields.io/badge/Retrieval-BM25_%2B_Dense-0F766E?style=flat-square)
![Team](https://img.shields.io/badge/Team-Attention_Please-334155?style=flat-square)

[English](#english) · [繁體中文](#traditional-chinese)<br>
[Technical Report](docs/WattBot2025_AttentionPlease_Technical_Report.pdf) · [Presentation](docs/WattBot2025_AttentionPlease_Presentation.pdf) · [Baseline Notebook](notebook/v1_gemini_2_5pro.ipynb) · [HERO Notebook](notebook/Gemini_2_5_pro_新pipeline_0_821.ipynb)

</div>

> **Reported best WattBot Score: 0.886 — HERO + Gemini 3 Pro.**<br>
> The public HERO notebook uses Gemini 2.5 Pro; the report records 0.821 for that configuration.<br>
> **報告最佳成績為 0.886（HERO + Gemini 3 Pro）；公開 HERO notebook 使用 Gemini 2.5 Pro，報告對應成績為 0.821。**

<a id="english"></a>

## English

[Overview](#en-overview) · [Results](#en-results) · [Baseline](#en-baseline) · [HERO](#en-hero) · [Comparison](#en-comparison) · [Data & Files](#en-data) · [Getting Started](#en-setup) · [Notes](#en-notes) · [Team & Citation](#en-team)

<a id="en-overview"></a>

### Overview

Developed by **Team Attention Please** for the WattBot 2025 challenge and an NLP term project, HERO retrieves evidence from academic papers and industry reports to answer questions about AI energy consumption, emissions, water use, and sustainability.

The task goes beyond producing fluent text: answers must contain normalized values and units, document references, and supporting evidence. When evidence is insufficient, the system must explicitly abstain.

**The central challenge — the table paradox.** Tables contain valuable numerical evidence, but their formatting can interfere with sparse retrieval. Removing tables reduces noise while discarding useful information. Our progression from a BM25 baseline to HERO addresses this tension through layout-aware parsing, OCR, hybrid retrieval, and content-aware weighting.

<a id="en-results"></a>

### Reported Results

| Model | Baseline | HERO | Gain |
| :--- | ---: | ---: | ---: |
| GPT-4o-mini | 0.753 | 0.789 | +0.036 |
| GPT-4o | 0.768 | 0.811 | +0.043 |
| GPT-5.1 | 0.778 | 0.807 | +0.029 |
| Gemini 2.5 Pro | 0.807 | 0.821 | +0.014 |
| **Gemini 3 Pro** | **0.853** | **0.886** | **+0.033** |

Source: [technical report, Figure 5](docs/WattBot2025_AttentionPlease_Technical_Report.pdf) and [presentation, slide 24](docs/WattBot2025_AttentionPlease_Presentation.pdf). These are reported competition scores, not new measurements from this documentation update. The presentation states “Top 1”; the repository does not provide separate evidence establishing both public and private first-place rankings.

The WattBot Score weights **answer value (75%)**, **reference IDs (15%)**, and **unanswerable handling (10%)**. The report describes ±0.1% relative tolerance for numeric answers and Jaccard overlap for citation sets.

<a id="en-baseline"></a>

### Baseline Architecture — Dual BM25 Retrieval

[Explore the Baseline notebook](notebook/v1_gemini_2_5pro.ipynb)

![Baseline BM25 architecture from technical report Figure 2](docs/images/baseline-architecture.png)

*Source: [technical report, Figure 2 (page 5)](docs/WattBot2025_AttentionPlease_Technical_Report.pdf#page=5). The report illustrates text-only and hybrid BM25 indexes; the notebook routing implementation described below uses separate text and table indexes.*

1. **Extract and prepare evidence.** PDF parsing and chunking produce text and table content. The notebook contains several experiments; the walkthrough below highlights its `hybrid_search` text/table routing implementation.
2. **Build two sparse indexes.** Separate BM25 indexes support text and table search. Here, “hybrid” refers to combining content types, rather than combining sparse and dense retrieval.
3. **Route the question.** Ordinary queries retrieve text chunks. Table-related queries combine seven text chunks with three table chunks, multiplying table scores by **0.8** before sorting.
4. **Generate in two stages.** The generation design first uses a small context, then expands context if the initial answer is unavailable. Outputs include values, units, evidence, and citations.

**Why move beyond it?** Exact keyword matching struggles with synonyms, noisy table representations, and evidence inside figures. The baseline notebook also includes alternative retrieval and generation paths; it is an experimental record rather than one packaged executable pipeline.

<a id="en-hero"></a>

### HERO Architecture — Visual-Aware Hybrid Retrieval

[Explore the HERO notebook](notebook/Gemini_2_5_pro_新pipeline_0_821.ipynb)

![HERO architecture from technical report Figure 4](docs/images/hero-architecture.png)

*Source: [technical report, Figure 4 (page 7)](docs/WattBot2025_AttentionPlease_Technical_Report.pdf#page=7). The original figure shows the report configuration (BGE-M3); the public notebook uses BGE-large and the settings documented below.*

| Step | Implementation | Purpose |
| :--- | :--- | :--- |
| Ingestion | Docling + EasyOCR + OpenCV | Extract layout-aware text, tables, and text from figures |
| Sparse retrieval | BM25 + enhanced tokenization | Match technical terms, numbers, and years |
| Dense retrieval | `BAAI/bge-large-en-v1.5` + FAISS `IndexFlatIP` | Retrieve semantically related evidence using normalized embeddings |
| Query enhancement | Unit expansion + author/year hints | Connect abbreviated units with domain vocabulary |
| Fusion | Weighted reciprocal rank fusion | Combine rankings; boost tables, OCR, and metadata matches |
| Reranking | `BAAI/bge-reranker-v2-m3` | Score query–chunk pairs jointly |
| Generation | Gemini 2.5 Pro + question-type prompts | Produce structured answers with adaptive context |
| Export | pandas → CSV | Assemble answers, references, evidence, and explanations |

**How evidence flows.** Offline processing turns PDFs into typed chunks and builds sparse and dense indexes. Online retrieval searches both indexes, fuses their rankings, and reranks candidates before generation. The generator uses question-specific instructions for tables, model-specific values, scores, terms, and numeric extraction.

**Weighting matters.** RRF operates on ranks rather than directly mixing incompatible BM25 and vector scores. Table chunks receive **2.0×**, image OCR chunks **1.8×**, and metadata matches **2.0×** weighting. Metadata weighting can compound with content-type weighting.

**Context is bounded by retrieval.** `adaptive_search_final` defaults to 100 fused candidates and eight final chunks. Generation requests up to seven or ten chunks in its fallback stage, but can only use the chunks actually supplied. The query-based reranking helper currently enables reranking for all queries.

**Model fallbacks.** If BGE-large fails to load, the notebook attempts `all-MiniLM-L6-v2`; the reranker similarly falls back to `ms-marco-MiniLM-L-12-v2`.

<a id="en-comparison"></a>

### Baseline vs. HERO

| Aspect | Baseline routing path | HERO public notebook |
| :--- | :--- | :--- |
| PDF processing | PyMuPDF / pdfplumber experiments | Docling + EasyOCR |
| Evidence | Text and tables | Text, tables, and figure OCR |
| Retrieval | Text/table BM25 indexes | BM25 + BGE-large + FAISS |
| Table treatment | Score penalty ×0.8 | Rank-fusion boost ×2.0 |
| Result combination | Sort weighted BM25 scores | Weighted RRF + CrossEncoder |
| Query adaptation | Table-related keyword routing | Units, metadata hints, and question-type prompts |
| Generation | Two-stage context expansion | Type-aware two-stage context expansion |

<a id="en-data"></a>

### Dataset & Repository

| File | Records | Contents |
| :--- | ---: | :--- |
| `metadata.csv` | 32 | Document IDs, titles, years, citations, and source URLs |
| `train_QA.csv` | 41 | Questions, gold answers, units, citations, and supporting evidence |
| `test_Q.csv` | 282 | Test questions and expected units; answer fields are empty |

```text
WattBot-RAG-System/
├── README.md
├── notebook/
│   ├── v1_gemini_2_5pro.ipynb
│   └── Gemini_2_5_pro_新pipeline_0_821.ipynb
├── data/
│   ├── metadata.csv
│   ├── train_QA.csv
│   └── test_Q.csv
└── docs/
    ├── WattBot2025_AttentionPlease_Technical_Report.pdf
    └── WattBot2025_AttentionPlease_Presentation.pdf
```

The PDFs referenced by `metadata.csv` are downloaded separately; the two PDFs in `docs/` are the project report and presentation.

<a id="en-setup"></a>

### Getting Started

The notebooks target a **Google Colab-style environment** with `/content/` paths. They require manual setup and have not been converted into a standalone command-line application.

1. Open the [HERO notebook](notebook/Gemini_2_5_pro_新pipeline_0_821.ipynb) in Colab, or Jupyter with paths adapted to your environment.
2. Upload the three CSV files from `data/` to `/content/`. Review `DATA_DIR`, `PDF_DIR`, and `CHUNK_DIR` before execution.
3. Install the dependencies used by the HERO workflow:

```python
%pip install docling easyocr opencv-python-headless rank-bm25 sentence-transformers faiss-cpu pandas numpy requests pdfminer.six pillow tqdm google-genai
```

4. Configure your Gemini API key in the generation cell. Prefer a secret store or environment variable over saving a key in the notebook:

```python
import os
from google import genai

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
```

   This snippet replaces the notebook's manual key/client setup; merely setting the environment variable does not change its existing cell.
5. Run the ingestion, OCR, indexing, retrieval, and generation cells in order. Review optional evaluation and cache-cleanup cells separately; the notebooks contain exploratory code and environment-specific assumptions.
6. The HERO generation cell writes `/content/submission_enhanced.csv`.

The submission contains `id`, `question`, `answer`, `answer_value`, `answer_unit`, `ref_id`, `ref_url`, `supporting_materials`, and `explanation`. Generation first produces structured JSON and then maps it to these CSV columns.

To explore the baseline, open its notebook separately. It additionally uses Kaggle authentication/downloads and packages such as `pdfplumber`, `pdf2image`, `pytesseract`, `fuzzywuzzy`, and `python-Levenshtein`; OCR cells use the Poppler and Tesseract system tools.

<a id="en-notes"></a>

### Reproducibility & Evaluation Notes

- **Report versus code.** The report describes BGE-M3 with FAISS L2 and a different reranker configuration. The public HERO notebook uses BGE-large, normalized embeddings with FAISS inner-product search, and BGE-reranker-v2-m3. This README documents that public implementation.
- **Published scores versus available artifact.** The best 0.886 result is attributed to Gemini 3 Pro in the report. The checked-in HERO notebook initializes Gemini 2.5 Pro; its filename is not evidence that the 0.886 configuration is included.
- **Retrieval metrics need context.** The baseline comparison uses 39 queries, while the final HERO saved evaluation uses 41, including unanswerable examples. The saved HERO output reports Recall@10 = 95.12% and MRR = 0.8519; some report values differ. These runs should not be presented as a controlled comparison on an identical evaluation set.
- **Recall definition.** The notebook's Recall@K checks whether any gold reference appears among the top-K results; it is a hit-rate measure. Multiple chunks can come from one document, so chunk ranks and unique-document ranks are not interchangeable.
- **Abstention behavior.** The notebook uses keyword checks for some out-of-domain questions and an LLM fallback response. These heuristics do not establish calibrated confidence. Its fallback preserves the requested unit, which differs from the report's description of all unavailable fields becoming `is_blank`.
- **Execution status.** This update checks documentation against the stored artifacts; it does not rerun PDF downloads, model inference, or paid Gemini requests. Dependencies are not pinned and API/model availability must be checked when reproducing.

<a id="en-team"></a>

### Team & Citation

**Team Attention Please · National Chengchi University**

Authors as listed in the technical report: **Shao-Hua Wu, Pei-Ju Hsieh, Bo-Hao Chen, Yi-Chen Hsiao, and Yi-Yang Xue**. The report marks the first three authors as equal contributors.

For this repository and its accompanying report:

```bibtex
@misc{wattbot2025_hero,
  title = {WattBot 2025: Retrieval-Augmented Generation for ESG Intelligence},
  author = {Wu, Shao-Hua and Hsieh, Pei-Ju and Chen, Bo-Hao and Hsiao, Yi-Chen and Xue, Yi-Yang},
  year = {2025},
  howpublished = {GitHub repository and accompanying technical report},
  url = {https://github.com/AnnieHsieh-12/WattBot-RAG-System}
}
```

Thanks to the WattBot organizers and the open-source tools supporting this project. See the [technical report](docs/WattBot2025_AttentionPlease_Technical_Report.pdf) for related work, including the KohakuRAG case study that informed the team's design. The checked-in repository does not contain a `LICENSE` file; no license grant is asserted here.

[Back to top](#wattbot--hero) · [繁體中文](#traditional-chinese)

---

<a id="traditional-chinese"></a>

## 繁體中文

[專案介紹](#zh-overview) · [實驗成果](#zh-results) · [Baseline 架構](#zh-baseline) · [HERO 架構](#zh-hero) · [架構比較](#zh-comparison) · [資料與目錄](#zh-data) · [開始使用](#zh-setup) · [重現說明](#zh-notes) · [團隊與引用](#zh-team)

<a id="zh-overview"></a>

### 專案介紹

本專案由 **Attention Please 團隊**為 WattBot 2025 競賽與自然語言處理期末專題開發，透過檢索增強生成（RAG），從學術論文與產業報告中回答 AI 能耗、碳排、水資源與永續相關問題。

任務不只是產生流暢回答，還需要提供標準化數值、單位、文件引用與支持證據；證據不足時，必須明確表示無法作答。

**核心問題：表格悖論（The Table Paradox）。** 表格包含重要數值，但格式雜訊可能干擾 BM25。刪除表格雖然降低雜訊，也會失去關鍵證據。我們從 BM25 baseline 出發，透過版面解析、OCR、混合檢索與內容加權，逐步發展為 HERO。

<a id="zh-results"></a>

### 實驗成果

| Model | Baseline | HERO | Gain |
| :--- | ---: | ---: | ---: |
| GPT-4o-mini | 0.753 | 0.789 | +0.036 |
| GPT-4o | 0.768 | 0.811 | +0.043 |
| GPT-5.1 | 0.778 | 0.807 | +0.029 |
| Gemini 2.5 Pro | 0.807 | 0.821 | +0.014 |
| **Gemini 3 Pro** | **0.853** | **0.886** | **+0.033** |

資料來源：[技術報告 Figure 5](docs/WattBot2025_AttentionPlease_Technical_Report.pdf) 與[簡報第 24 頁](docs/WattBot2025_AttentionPlease_Presentation.pdf)。上表是文件記錄的競賽成績，並非本次修改重新實測的結果。簡報標示「Top 1」，但 repo 沒有分別證明 public 與 private leaderboard 都是第一名的資料。

WattBot Score 包含 **答案值 75%**、**引用文件 15%**、**無法作答判定 10%**。報告說明數值答案允許 ±0.1% 相對誤差，引用集合以 Jaccard overlap 計分。

<a id="zh-baseline"></a>

### Baseline 架構：雙 BM25 檢索

[開啟 Baseline notebook](notebook/v1_gemini_2_5pro.ipynb)

![技術報告 Figure 2：Baseline BM25 架構](docs/images/baseline-architecture.png)

*來源：[技術報告 Figure 2，第 5 頁](docs/WattBot2025_AttentionPlease_Technical_Report.pdf#page=5)。原圖以純文字與混合 BM25 索引表示；下方說明的 notebook 路由實作則分開建立文字與表格索引。*

1. **整理文件證據。** 解析 PDF 並產生文字與表格區塊。Notebook 保留多個實驗版本，下方流程說明以 `hybrid_search` 的文字／表格路由設計為主。
2. **建立雙索引。** 分別建立文字與表格的 BM25 索引。此處的 hybrid 指「混合內容類型」，不是稀疏與稠密檢索的結合。
3. **依問題路由。** 一般問題取文字區塊；表格相關問題混合七個文字與三個表格區塊，表格分數乘上 **0.8** 後排序。
4. **兩階段生成。** 先以較小上下文作答，無法回答時再擴大證據範圍，輸出答案值、單位、支持內容與引用。

**為什麼需要升級？** 關鍵字檢索較難處理同義詞、表格格式雜訊與圖像中的資訊。Baseline notebook 也保留其他檢索與生成路徑，因此應視為實驗紀錄，而非已封裝的單一執行程式。

<a id="zh-hero"></a>

### HERO 架構：結合視覺證據的混合檢索

[開啟 HERO notebook](notebook/Gemini_2_5_pro_新pipeline_0_821.ipynb)

![技術報告 Figure 4：HERO 架構](docs/images/hero-architecture.png)

*來源：[技術報告 Figure 4，第 7 頁](docs/WattBot2025_AttentionPlease_Technical_Report.pdf#page=7)。原圖呈現報告中的 BGE-M3 設定；公開 notebook 使用 BGE-large，實際設定見下方說明。*

| 階段 | 公開實作 | 目的 |
| :--- | :--- | :--- |
| 文件處理 | Docling + EasyOCR + OpenCV | 擷取文字、結構化表格與圖像內文字 |
| 稀疏檢索 | BM25 + 強化斷詞 | 比對技術名詞、數值與年份 |
| 稠密檢索 | `BAAI/bge-large-en-v1.5` + FAISS `IndexFlatIP` | 使用正規化向量找出語意相關證據 |
| 查詢增強 | 單位擴展 + 作者／年份提示 | 連結縮寫單位與領域詞彙 |
| 排名融合 | 加權倒數排名融合（RRF） | 合併兩種排名，提高表格、OCR 與元資料匹配權重 |
| 重排序 | `BAAI/bge-reranker-v2-m3` | 聯合評估問題與候選區塊的相關性 |
| 答案生成 | Gemini 2.5 Pro + 題型專屬提示詞 | 根據問題類型調整上下文並生成結構化答案 |
| 結果匯出 | pandas → CSV | 組合答案、引用、支持證據與解釋 |

**證據如何流動？** 離線階段先將 PDF 轉為帶有類型標籤的區塊，建立稀疏與稠密索引；線上階段搜尋兩個索引、融合排名，再以 CrossEncoder 重排序。生成器依表格、模型數值、分數、術語與數值擷取等題型套用提示詞。

**為什麼使用加權 RRF？** BM25 與向量分數的尺度不同，RRF 以排名合併結果。表格權重為 **2.0 倍**、圖像 OCR 為 **1.8 倍**、作者／年份匹配為 **2.0 倍**；元資料權重可與內容類型權重相乘。

**上下文受檢索結果限制。** `adaptive_search_final` 預設保留 100 個融合候選，再取八個重排序結果。第二階段生成雖可要求七個或十個區塊，實際仍只能使用傳入的證據數量。目前 query-based helper 對所有問題都啟用重排序。

**備援模型。** BGE-large 載入失敗時嘗試 `all-MiniLM-L6-v2`；重排序模型則備援為 `ms-marco-MiniLM-L-12-v2`。

<a id="zh-comparison"></a>

### Baseline 與 HERO 比較

| 面向 | Baseline 路由版本 | HERO 公開 notebook |
| :--- | :--- | :--- |
| PDF 處理 | PyMuPDF／pdfplumber 實驗 | Docling + EasyOCR |
| 證據類型 | 文字與表格 | 文字、表格與圖像 OCR |
| 檢索方式 | 文字／表格雙 BM25 索引 | BM25 + BGE-large + FAISS |
| 表格處理 | 分數降權 ×0.8 | 排名融合加權 ×2.0 |
| 結果整合 | 依加權 BM25 分數排序 | 加權 RRF + CrossEncoder |
| 查詢調整 | 表格關鍵字路由 | 單位、元資料與題型提示詞 |
| 答案生成 | 兩階段擴大上下文 | 依題型調整的兩階段上下文 |

<a id="zh-data"></a>

### 資料與專案目錄

| 檔案 | 筆數 | 內容 |
| :--- | ---: | :--- |
| `metadata.csv` | 32 | 文件 ID、標題、年份、引用與來源網址 |
| `train_QA.csv` | 41 | 問題、標準答案、單位、引用與支持證據 |
| `test_Q.csv` | 282 | 測試問題與預期單位，答案欄位留空 |

```text
WattBot-RAG-System/
├── README.md
├── notebook/
│   ├── v1_gemini_2_5pro.ipynb
│   └── Gemini_2_5_pro_新pipeline_0_821.ipynb
├── data/
│   ├── metadata.csv
│   ├── train_QA.csv
│   └── test_Q.csv
└── docs/
    ├── WattBot2025_AttentionPlease_Technical_Report.pdf
    └── WattBot2025_AttentionPlease_Presentation.pdf
```

原始文獻 PDF 需依 `metadata.csv` 另行下載；`docs/` 中的兩份 PDF 是本專案的技術報告與簡報。

<a id="zh-setup"></a>

### 開始使用

Notebook 以 **Google Colab 類型環境**為主，使用 `/content/` 路徑，需要手動設定，尚未整理為獨立命令列應用程式。

1. 在 Colab 開啟 [HERO notebook](notebook/Gemini_2_5_pro_新pipeline_0_821.ipynb)；若使用本機 Jupyter，先調整路徑。
2. 將 `data/` 的三份 CSV 上傳至 `/content/`，檢查 `DATA_DIR`、`PDF_DIR` 與 `CHUNK_DIR`。
3. 安裝 HERO 流程使用的套件：

```python
%pip install docling easyocr opencv-python-headless rank-bm25 sentence-transformers faiss-cpu pandas numpy requests pdfminer.six pillow tqdm google-genai
```

4. 在生成階段設定 Gemini API key。建議透過秘密管理工具或環境變數提供，不要將金鑰儲存在 notebook：

```python
import os
from google import genai

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
```

   上述程式用來替換原本手動設定 key 與 client 的部分；只設定環境變數，不會自動改變現有 cell。
5. 依序執行文件下載、解析、OCR、索引、檢索與生成。評估與清除快取 cell 請另外檢查；notebook 保留探索性程式與環境假設。
6. HERO 的生成 cell 會輸出 `/content/submission_enhanced.csv`。

輸出欄位包含 `id`、`question`、`answer`、`answer_value`、`answer_unit`、`ref_id`、`ref_url`、`supporting_materials` 與 `explanation`。生成器先產生結構化 JSON，再轉換為 CSV 欄位。

若要研究 Baseline，請另外開啟其 notebook。它還使用 Kaggle 登入／下載，以及 `pdfplumber`、`pdf2image`、`pytesseract`、`fuzzywuzzy`、`python-Levenshtein` 等套件；OCR cell 使用 Poppler 與 Tesseract 系統工具。

<a id="zh-notes"></a>

### 重現與評估說明

- **報告與程式設定不同。** 報告描述 BGE-M3、FAISS L2 與不同的 reranker 設定；公開 HERO notebook 實際使用 BGE-large、正規化向量的 FAISS 內積搜尋，以及 BGE-reranker-v2-m3。本 README 的實作說明以公開程式為準。
- **最佳成績與公開版本分開看。** 報告將 0.886 歸於 Gemini 3 Pro；公開 HERO notebook 初始化 Gemini 2.5 Pro，檔名不代表包含 0.886 的完整設定。
- **檢索指標需標明評估範圍。** Baseline 比較使用 39 題，HERO 最後留存的評估使用 41 題，包含不可回答問題。HERO 輸出記錄 Recall@10 為 95.12%、MRR 為 0.8519，部分數值與報告不同，不能直接當成同一評估集上的控制實驗。
- **Recall 的定義。** Notebook 的 Recall@K 判斷前 K 個結果是否命中任一標準引用，實際上是命中率。多個區塊可能來自同一文件，因此區塊排名與去重後的文件排名不同。
- **拒答的實際行為。** 部分領域外問題以關鍵字阻擋，其餘交由 LLM 判斷；這些規則不代表已校準的信心分數。程式的預設拒答保留預期單位，與報告描述的不可用欄位全部設為 `is_blank` 有差異。
- **本次驗證範圍。** 文件更新依現有資料核對，沒有重新下載全部文獻、執行模型推論或呼叫付費 Gemini API。套件未固定版本，重現時需確認模型與 API 可用性。

<a id="zh-team"></a>

### 團隊與引用

**Attention Please 團隊 · 國立政治大學**

依技術報告列名：**Shao-Hua Wu、Pei-Ju Hsieh、Bo-Hao Chen、Yi-Chen Hsiao、Yi-Yang Xue**。報告將前三位作者標示為共同貢獻。

引用資訊請使用[英文區的 BibTeX](#en-team)，網址已指向本專案。感謝 WattBot 主辦單位與開源工具社群；相關研究與啟發設計的 KohakuRAG 案例，可參閱[技術報告](docs/WattBot2025_AttentionPlease_Technical_Report.pdf)。目前 repo 未包含 `LICENSE` 檔案，此處不宣告授權條款。

[回到英文版](#english) · [回到頂端](#wattbot--hero)
