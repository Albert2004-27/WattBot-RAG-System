# WattBot 2025 - Team Attention Please 🔋

<div align="center">

[![Competition](https://img.shields.io/badge/Kaggle-WattBot%202025-20BEFF?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/competitions/WattBot2025)
![Team](https://img.shields.io/badge/Team-Attention%20Please-FF6B6B?style=for-the-badge)
![System](https://img.shields.io/badge/System-HERO-9B59B6?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

### HERO: Hybrid Evidence Retrieval Optimization

*Evidence-based Energy Estimation for AI Workloads using Retrieval-Augmented Generation*

[📄 Technical Report](docs/WattBot2025_AttentionPlease_Technical_Report.pdf) | [📊 Presentation](docs/WattBot2025_AttentionPlease_Presentation.pdf)

[Baseline Notebook](notebook/v1_gemini_2_5pro.ipynb) · [HERO Notebook](notebook/Gemini_2_5_pro_新pipeline_0_821.ipynb) · [繁體中文](#中文版說明)

</div>

## Table of Contents

- [Introduction](#introduction)
- [Technology Stack](#technology-stack)
- [System Architecture](#system-architecture)
  - [Baseline: BM25 Retrieval](#baseline-bm25-retrieval)
  - [HERO: Visual-Aware Hybrid Retrieval](#hero-visual-aware-hybrid-retrieval)
  - [Architecture Comparison](#architecture-comparison)
- [Project Structure](#project-structure)
- [System Workflow](#system-workflow)
- [Dataset and Evaluation](#dataset-and-evaluation)
- [Experimental Results](#experimental-results)
- [Quick Start](#quick-start)
- [Team Members](#team-members)
- [Citation](#citation)
- [中文版說明](#中文版說明)

## Introduction

WattBot asks a practical question: **can a system find the evidence behind an AI sustainability claim?** Answers often depend on a single value inside a table or figure, expressed in units that differ from the question.

Our system retrieves the relevant evidence and returns a normalized answer with document citations and supporting text. It also handles questions that cannot be answered from the available documents.

The project develops two approaches:

- **BM25 Baseline** — text and table retrieval with query routing and two-stage answer generation.
- **HERO** — layout-aware parsing and OCR, sparse–dense retrieval, weighted rank fusion, CrossEncoder reranking, and question-specific generation.

The key finding is the **table paradox**: adding raw tables can introduce noise into keyword retrieval, while removing them discards important numerical evidence. HERO addresses this by preserving content types and giving table and OCR evidence explicit weights during retrieval.

## Technology Stack

```text
Language:          Python
Environment:       Jupyter Notebook / Google Colab
Baseline Parsing:  PyMuPDF + pdfplumber
HERO Parsing:      Docling + EasyOCR + OpenCV
Sparse Retrieval:  BM25 (rank-bm25)
Dense Retrieval:   Sentence-Transformers + BGE-large + FAISS
Reranking:         BGE-reranker-v2-m3
Answer Generation: Google Gemini API
Data Processing:   pandas + NumPy
```

The stack above follows the public notebooks. The technical report describes additional experimental configurations.

## System Architecture

### Baseline: BM25 Retrieval

![Baseline BM25 architecture](docs/images/baseline-architecture.png)

*Figure 2 from the [technical report, page 5](docs/WattBot2025_AttentionPlease_Technical_Report.pdf#page=5).*

The baseline separates document processing from question answering:

1. **Document extraction.** Parse PDFs and prepare text and table chunks.
2. **Sparse indexing.** Build BM25 indexes for lexical matching.
3. **Query routing.** Identify table-related questions using keywords. The notebook's `hybrid_search` combines **Top-7 text + Top-3 table** results for these questions, applying a **0.8× table-score weight**. Other questions use text retrieval.
4. **Answer generation.** First answer from a small evidence set; if unsuccessful, expand the context and try again.
5. **Structured output.** Return the answer, value, unit, references, supporting evidence, and explanation.

The report diagram presents text-only and hybrid indexes; the notebook routing implementation separates text and table indexes. Here, “hybrid” refers to content types. This baseline routing path does not combine BM25 with dense retrieval.

### HERO: Visual-Aware Hybrid Retrieval

![HERO architecture](docs/images/hero-architecture.png)

*Figure 4 from the [technical report, page 7](docs/WattBot2025_AttentionPlease_Technical_Report.pdf#page=7).*

HERO extends the baseline in five stages:

1. **Visual-aware ingestion.** Docling extracts document elements; EasyOCR and OpenCV turn text inside figures into searchable evidence. Chunks retain their content type and document ID.
2. **Dual retrieval.** BM25 handles exact terms and numbers, while dense embeddings retrieve semantically related passages.
3. **Weighted reciprocal rank fusion.** Combine sparse and dense rankings, boosting **tables ×2.0**, **OCR ×1.8**, and **author/year matches ×2.0**.
4. **CrossEncoder reranking.** Jointly score each query–chunk pair to prioritize relevant evidence.
5. **Adaptive generation.** Choose prompts by question type, answer from Top-3 context first, and expand the supplied context when a second attempt is needed.

The original report figure labels the embedding model as **BGE-M3**. The public notebook uses **BGE-large-en-v1.5**, normalized embeddings with FAISS `IndexFlatIP`, and **BGE-reranker-v2-m3**. It includes MiniLM fallbacks for model-loading failures.

### Architecture Comparison

<table width="100%">
  <thead>
    <tr>
      <th width="500" align="left">Component</th>
      <th width="750" align="left">BM25 Baseline</th>
      <th width="750" align="left">HERO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left">Document parsing</td>
      <td align="left">PyMuPDF / pdfplumber</td>
      <td align="left">Docling + EasyOCR</td>
    </tr>
    <tr>
      <td align="left">Evidence sources</td>
      <td align="left">Text and tables</td>
      <td align="left">Text, tables, and figure OCR</td>
    </tr>
    <tr>
      <td align="left">Retrieval</td>
      <td align="left">Text/table BM25 indexes</td>
      <td align="left">BM25 + dense vector search</td>
    </tr>
    <tr>
      <td align="left">Table handling</td>
      <td align="left">BM25 score weight ×0.8</td>
      <td align="left">RRF weight ×2.0</td>
    </tr>
    <tr>
      <td align="left">Ranking</td>
      <td align="left">Weighted BM25 scores</td>
      <td align="left">Weighted RRF + CrossEncoder</td>
    </tr>
    <tr>
      <td align="left">Query adaptation</td>
      <td align="left">Table-related routing</td>
      <td align="left">Unit expansion and metadata hints</td>
    </tr>
    <tr>
      <td align="left">Generation</td>
      <td align="left">Two-stage fallback</td>
      <td align="left">Type-specific prompts and two-stage fallback</td>
    </tr>
  </tbody>
</table>

## Project Structure

```text
WattBot-RAG-System/
├── README.md
├── notebook/
│   ├── v1_gemini_2_5pro.ipynb                 # BM25 baseline experiments
│   └── Gemini_2_5_pro_新pipeline_0_821.ipynb   # HERO pipeline
├── data/
│   ├── metadata.csv                         # 32 source documents
│   ├── train_QA.csv                         # 41 training questions
│   └── test_Q.csv                           # 282 test questions
└── docs/
    ├── images/                              # Architecture figures
    ├── WattBot2025_AttentionPlease_Technical_Report.pdf
    └── WattBot2025_AttentionPlease_Presentation.pdf
```

## System Workflow

```text
Document preparation
    ↓
Download PDFs from metadata.csv
    ↓
Parse documents and extract text / tables / figure OCR
    ↓
Build retrieval indexes
    ↓
Read question and expected answer unit
    ↓
Retrieve and rank supporting evidence
    ↓
Generate structured answer with citations
    ↓
Expand context if the first attempt cannot answer
    ↓
Export submission CSV
```

Generation produces structured JSON, which is mapped to these submission fields:

<table width="100%">
  <thead>
    <tr>
      <th width="1000" align="left">Field</th>
      <th width="1000" align="left">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left"><code>id</code>, <code>question</code></td>
      <td align="left">Question identifier and text</td>
    </tr>
    <tr>
      <td align="left"><code>answer</code></td>
      <td align="left">Natural-language answer</td>
    </tr>
    <tr>
      <td align="left"><code>answer_value</code>, <code>answer_unit</code></td>
      <td align="left">Normalized value and unit</td>
    </tr>
    <tr>
      <td align="left"><code>ref_id</code>, <code>ref_url</code></td>
      <td align="left">Supporting document IDs and URLs</td>
    </tr>
    <tr>
      <td align="left"><code>supporting_materials</code></td>
      <td align="left">Evidence excerpt</td>
    </tr>
    <tr>
      <td align="left"><code>explanation</code></td>
      <td align="left">Explanation connecting evidence to the answer</td>
    </tr>
  </tbody>
</table>

## Dataset and Evaluation

<table width="100%">
  <thead>
    <tr>
      <th width="500" align="left">File</th>
      <th width="750" align="left">Records</th>
      <th width="750" align="left">Content</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left"><code>metadata.csv</code></td>
      <td align="right">32</td>
      <td align="left">Document metadata and source URLs</td>
    </tr>
    <tr>
      <td align="left"><code>train_QA.csv</code></td>
      <td align="right">41</td>
      <td align="left">Questions with answers, citations, and evidence</td>
    </tr>
    <tr>
      <td align="left"><code>test_Q.csv</code></td>
      <td align="right">282</td>
      <td align="left">Test questions and expected units</td>
    </tr>
  </tbody>
</table>

Source PDFs are downloaded separately using the metadata URLs. The PDFs in `docs/` are the project report and presentation.

<table width="100%">
  <thead>
    <tr>
      <th width="500" align="left">WattBot Score component</th>
      <th width="750" align="left">Weight</th>
      <th width="750" align="left">Criterion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left">Answer value</td>
      <td align="right">75%</td>
      <td align="left">Numeric tolerance or normalized categorical match</td>
    </tr>
    <tr>
      <td align="left">Reference IDs</td>
      <td align="right">15%</td>
      <td align="left">Jaccard overlap of citation sets</td>
    </tr>
    <tr>
      <td align="left">Unanswerable handling</td>
      <td align="right">10%</td>
      <td align="left">Correct recognition of unavailable evidence</td>
    </tr>
  </tbody>
</table>

The report specifies **±0.1% relative tolerance** for numeric answers.

## Experimental Results

### Pipeline and Model Comparison

<table width="100%">
  <thead>
    <tr>
      <th width="500" align="left">Model</th>
      <th width="750" align="left">BM25 Baseline</th>
      <th width="750" align="left">HERO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left">GPT-4o-mini</td>
      <td align="right">0.753</td>
      <td align="right">0.789</td>
    </tr>
    <tr>
      <td align="left">GPT-4o</td>
      <td align="right">0.768</td>
      <td align="right">0.811</td>
    </tr>
    <tr>
      <td align="left">GPT-5.1</td>
      <td align="right">0.778</td>
      <td align="right">0.807</td>
    </tr>
    <tr>
      <td align="left">Gemini 2.5 Pro</td>
      <td align="right">0.807</td>
      <td align="right">0.821</td>
    </tr>
    <tr>
      <td align="left"><strong>Gemini 3 Pro</strong></td>
      <td align="right"><strong>0.853</strong></td>
      <td align="right"><strong>0.886</strong></td>
    </tr>
  </tbody>
</table>

Reported WattBot Scores from [technical report Figure 5](docs/WattBot2025_AttentionPlease_Technical_Report.pdf#page=10). The best reported configuration is **HERO + Gemini 3 Pro (0.886)**; the public HERO notebook uses **Gemini 2.5 Pro (0.821 in the report)**.

### What We Learned

- **Table formatting affects retrieval.** More document content does not automatically produce better evidence ranking.
- **Visual evidence needs its own ingestion path.** OCR makes information inside figures available to text retrieval.
- **Retrieval and generation improve together.** HERO improves reported scores across all five evaluated models.
- **Context expansion is a trade-off.** A second attempt can recover missed evidence, but requires careful handling of unsupported answers.

<details>
<summary>Retrieval evaluation and reproducibility details</summary>

The baseline comparison uses 39 questions; the final saved HERO evaluation uses 41, including unanswerable examples. The HERO saved output reports Recall@10 of 95.12% and MRR of 0.8519. Some report values differ, so these runs should not be treated as a controlled comparison on identical samples.

The notebooks label a top-K reference hit rate as Recall@K. Multiple chunks may come from the same document, so chunk ranking and unique-document ranking differ.

The public HERO retrieval defaults to eight final chunks. Generation can request up to ten fallback chunks, but only uses the evidence supplied. Its query-based helper currently enables reranking for every query. Abstention uses heuristics and LLM responses rather than calibrated confidence.

</details>

## Quick Start

### 1. Open a Notebook

Use Google Colab for the `/content/` paths used by the notebooks, or adapt the paths for local Jupyter:

- [Baseline notebook](notebook/v1_gemini_2_5pro.ipynb)
- [HERO notebook](notebook/Gemini_2_5_pro_新pipeline_0_821.ipynb)

### 2. Prepare the Data

Upload `metadata.csv`, `train_QA.csv`, and `test_Q.csv` from `data/` to `/content/`. Check the notebook's data, PDF, and chunk paths before running cells.

### 3. Install HERO Dependencies

```python
%pip install docling easyocr opencv-python-headless rank-bm25 sentence-transformers faiss-cpu pandas numpy requests pdfminer.six pillow tqdm google-genai
```

The baseline additionally uses Kaggle authentication, `pdfplumber`, `pdf2image`, `pytesseract`, `fuzzywuzzy`, and `python-Levenshtein`. Its OCR cells require Poppler and Tesseract system tools.

### 4. Configure Gemini

Replace the notebook's manual API-key/client initialization with your preferred secret source. For an environment variable:

```python
import os
from google import genai

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
```

### 5. Run the Pipeline

Run ingestion, indexing, retrieval, and generation in sequence. Review experimental, evaluation, and cache-cleanup cells before executing them.

The HERO generation cell saves:

```text
/content/submission_enhanced.csv
```

These are research notebooks with manual configuration, not a packaged command-line application. Dependencies are not pinned; model/API availability and source-document downloads may affect reproduction.

## Team Members

**Attention Please · National Chengchi University**

<table width="100%">
  <thead>
    <tr>
      <th width="1000" align="left">Member</th>
      <th width="1000" align="left">Affiliation in the technical report</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left">Shao-Hua Wu‡</td>
      <td align="left">Department of Money and Banking</td>
    </tr>
    <tr>
      <td align="left">Pei-Ju Hsieh‡</td>
      <td align="left">Department of Money and Banking</td>
    </tr>
    <tr>
      <td align="left">Bo-Hao Chen‡</td>
      <td align="left">Department of Money and Banking</td>
    </tr>
    <tr>
      <td align="left">Yi-Chen Hsiao</td>
      <td align="left">Department of Finance</td>
    </tr>
    <tr>
      <td align="left">Yi-Yang Xue</td>
      <td align="left">Department of Computer Science</td>
    </tr>
  </tbody>
</table>

‡ Equal contribution, as marked in the technical report.

## Citation

```bibtex
@misc{wattbot2025_hero,
  title = {WattBot 2025: Retrieval-Augmented Generation for ESG Intelligence},
  author = {Wu, Shao-Hua and Hsieh, Pei-Ju and Chen, Bo-Hao and Hsiao, Yi-Chen and Xue, Yi-Yang},
  year = {2025},
  url = {https://github.com/AnnieHsieh-12/WattBot-RAG-System}
}
```

Related work and acknowledgments are included in the technical report. The repository currently does not include a license file.

---

## 中文版說明

## WattBot 2025：AI 環境影響的證據檢索與問答系統

**HERO（Hybrid Evidence Retrieval Optimization）** 是 Attention Please 團隊為 WattBot 2025 與政大自然語言處理期末專題開發的 RAG 系統，從學術論文與產業報告中擷取 AI 能耗、碳排、水資源與永續相關資訊。

[技術報告](docs/WattBot2025_AttentionPlease_Technical_Report.pdf) · [專案簡報](docs/WattBot2025_AttentionPlease_Presentation.pdf) · [Baseline notebook](notebook/v1_gemini_2_5pro.ipynb) · [HERO notebook](notebook/Gemini_2_5_pro_新pipeline_0_821.ipynb) · [English](#table-of-contents)

## 中文目錄

- [專案介紹](#專案介紹)
- [技術架構](#技術架構)
- [系統架構](#系統架構)
  - [Baseline：BM25 檢索](#baselinebm25-檢索)
  - [HERO：結合視覺證據的混合檢索](#hero結合視覺證據的混合檢索)
  - [架構比較](#架構比較)
- [專案目錄](#專案目錄)
- [系統流程](#系統流程)
- [資料集與評分方式](#資料集與評分方式)
- [實驗成果](#實驗成果)
- [快速啟動](#快速啟動)
- [團隊成員與引用](#團隊成員與引用)

## 專案介紹

WattBot 的核心任務是：**找出支持 AI 永續資訊的實際證據。** 答案可能藏在表格的一個數值、圖像中的標籤，或使用不同單位的文獻段落中。

系統除了回答問題，還需要提供標準化數值、單位、文件引用與支持內容；若資料不足，也必須辨識無法回答的情況。

本專案包含兩套方法：

- **BM25 Baseline**：結合文字與表格檢索、問題路由，以及兩階段答案生成。
- **HERO**：加入版面解析、OCR、稀疏與稠密檢索、加權排名融合、CrossEncoder 重排序及題型專屬提示詞。

研究的關鍵是**表格悖論（Table Paradox）**：加入原始表格可能干擾關鍵字檢索，刪除表格又會失去重要數值。HERO 保留內容類型，透過加權融合提高表格與 OCR 證據的可見度。

## 技術架構

```text
程式語言：     Python
執行環境：     Jupyter Notebook / Google Colab
Baseline 解析：PyMuPDF + pdfplumber
HERO 解析：    Docling + EasyOCR + OpenCV
稀疏檢索：     BM25 (rank-bm25)
稠密檢索：     Sentence-Transformers + BGE-large + FAISS
重排序：       BGE-reranker-v2-m3
答案生成：     Google Gemini API
資料處理：     pandas + NumPy
```

上述設定以公開 notebook 為準；技術報告另包含其他實驗配置。

## 系統架構

### Baseline：BM25 檢索

![Baseline BM25 系統架構](docs/images/baseline-architecture.png)

*擷取自[技術報告第 5 頁 Figure 2](docs/WattBot2025_AttentionPlease_Technical_Report.pdf#page=5)。*

Baseline 將文件處理與問答分成以下步驟：

1. **文件擷取**：解析 PDF，整理文字與表格區塊。
2. **建立索引**：使用 BM25 進行關鍵字與數值比對。
3. **問題路由**：判斷是否為表格相關問題。Notebook 的 `hybrid_search` 對這類問題合併 **Top-7 文字 + Top-3 表格**，表格分數乘上 **0.8** 後排序；其他問題使用文字檢索。
4. **答案生成**：先以較少證據作答，無法回答時再擴大上下文。
5. **結構化輸出**：產生答案、數值、單位、引用、支持內容與解釋。

報告原圖以純文字與混合索引呈現，notebook 的路由版本則分開建立文字與表格索引。此處的 hybrid 指內容類型混合，這條 baseline 路徑尚未結合向量檢索。

### HERO：結合視覺證據的混合檢索

![HERO 系統架構](docs/images/hero-architecture.png)

*擷取自[技術報告第 7 頁 Figure 4](docs/WattBot2025_AttentionPlease_Technical_Report.pdf#page=7)。*

HERO 在 Baseline 基礎上加入五個階段：

1. **視覺證據擷取**：Docling 解析文件元素，EasyOCR 與 OpenCV 擷取圖像中的文字，保留區塊類型與文件 ID。
2. **雙路檢索**：BM25 比對精確詞彙與數值，稠密向量補足語意相近但用詞不同的證據。
3. **加權倒數排名融合**：透過 RRF 合併排名，設定**表格 ×2.0、OCR ×1.8、作者／年份匹配 ×2.0**。
4. **CrossEncoder 重排序**：聯合評估問題與候選區塊，優先保留相關證據。
5. **自適應生成**：依題型選擇提示詞，先用 Top-3 上下文回答，需要第二次嘗試時再擴大傳入的證據範圍。

原圖標示的模型為 **BGE-M3**；公開 notebook 實際使用 **BGE-large-en-v1.5**、正規化向量的 FAISS `IndexFlatIP` 與 **BGE-reranker-v2-m3**，並提供 MiniLM 載入失敗備援。

### 架構比較

<table width="100%">
  <thead>
    <tr>
      <th width="500" align="left">元件</th>
      <th width="750" align="left">BM25 Baseline</th>
      <th width="750" align="left">HERO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left">文件解析</td>
      <td align="left">PyMuPDF / pdfplumber</td>
      <td align="left">Docling + EasyOCR</td>
    </tr>
    <tr>
      <td align="left">證據來源</td>
      <td align="left">文字與表格</td>
      <td align="left">文字、表格與圖像 OCR</td>
    </tr>
    <tr>
      <td align="left">檢索方式</td>
      <td align="left">文字／表格 BM25 索引</td>
      <td align="left">BM25 + 稠密向量搜尋</td>
    </tr>
    <tr>
      <td align="left">表格處理</td>
      <td align="left">BM25 分數權重 ×0.8</td>
      <td align="left">RRF 權重 ×2.0</td>
    </tr>
    <tr>
      <td align="left">結果排序</td>
      <td align="left">加權 BM25 分數</td>
      <td align="left">加權 RRF + CrossEncoder</td>
    </tr>
    <tr>
      <td align="left">查詢調整</td>
      <td align="left">表格相關問題路由</td>
      <td align="left">單位擴展與元資料提示</td>
    </tr>
    <tr>
      <td align="left">答案生成</td>
      <td align="left">兩階段 fallback</td>
      <td align="left">題型提示詞與兩階段 fallback</td>
    </tr>
  </tbody>
</table>

## 專案目錄

```text
WattBot-RAG-System/
├── README.md
├── notebook/
│   ├── v1_gemini_2_5pro.ipynb                 # BM25 baseline experiments
│   └── Gemini_2_5_pro_新pipeline_0_821.ipynb   # HERO pipeline
├── data/
│   ├── metadata.csv                         # 32 source documents
│   ├── train_QA.csv                         # 41 training questions
│   └── test_Q.csv                           # 282 test questions
└── docs/
    ├── images/                              # Architecture figures
    ├── WattBot2025_AttentionPlease_Technical_Report.pdf
    └── WattBot2025_AttentionPlease_Presentation.pdf
```

## 系統流程

```text
準備文件
    ↓
依 metadata.csv 下載 PDF
    ↓
解析文字、表格與圖像 OCR
    ↓
建立檢索索引
    ↓
讀取問題與預期單位
    ↓
檢索並排序支持證據
    ↓
生成結構化答案與引用
    ↓
第一次無法作答時擴大上下文
    ↓
匯出 submission CSV
```

生成器先產生結構化 JSON，再轉換成競賽欄位：

<table width="100%">
  <thead>
    <tr>
      <th width="1000" align="left">欄位</th>
      <th width="1000" align="left">說明</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left"><code>id</code>、<code>question</code></td>
      <td align="left">問題編號與內容</td>
    </tr>
    <tr>
      <td align="left"><code>answer</code></td>
      <td align="left">自然語言答案</td>
    </tr>
    <tr>
      <td align="left"><code>answer_value</code>、<code>answer_unit</code></td>
      <td align="left">標準化答案值與單位</td>
    </tr>
    <tr>
      <td align="left"><code>ref_id</code>、<code>ref_url</code></td>
      <td align="left">支持文件 ID 與網址</td>
    </tr>
    <tr>
      <td align="left"><code>supporting_materials</code></td>
      <td align="left">證據摘錄</td>
    </tr>
    <tr>
      <td align="left"><code>explanation</code></td>
      <td align="left">證據如何支持答案</td>
    </tr>
  </tbody>
</table>

## 資料集與評分方式

<table width="100%">
  <thead>
    <tr>
      <th width="500" align="left">檔案</th>
      <th width="750" align="left">筆數</th>
      <th width="750" align="left">內容</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left"><code>metadata.csv</code></td>
      <td align="right">32</td>
      <td align="left">文件資訊與來源網址</td>
    </tr>
    <tr>
      <td align="left"><code>train_QA.csv</code></td>
      <td align="right">41</td>
      <td align="left">問題、標準答案、引用與證據</td>
    </tr>
    <tr>
      <td align="left"><code>test_Q.csv</code></td>
      <td align="right">282</td>
      <td align="left">測試問題與預期單位</td>
    </tr>
  </tbody>
</table>

原始文獻需依 metadata 網址另外下載；`docs/` 的 PDF 是專案報告與簡報。

<table width="100%">
  <thead>
    <tr>
      <th width="500" align="left">WattBot Score 項目</th>
      <th width="750" align="left">權重</th>
      <th width="750" align="left">評分依據</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left">答案值</td>
      <td align="right">75%</td>
      <td align="left">數值誤差或標準化類別比對</td>
    </tr>
    <tr>
      <td align="left">引用文件</td>
      <td align="right">15%</td>
      <td align="left">引用集合的 Jaccard overlap</td>
    </tr>
    <tr>
      <td align="left">無法作答判定</td>
      <td align="right">10%</td>
      <td align="left">正確辨識缺乏證據的問題</td>
    </tr>
  </tbody>
</table>

報告設定數值答案的容許相對誤差為 **±0.1%**。

## 實驗成果

### 不同模型與 Pipeline 比較

<table width="100%">
  <thead>
    <tr>
      <th width="500" align="left">Model</th>
      <th width="750" align="left">BM25 Baseline</th>
      <th width="750" align="left">HERO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left">GPT-4o-mini</td>
      <td align="right">0.753</td>
      <td align="right">0.789</td>
    </tr>
    <tr>
      <td align="left">GPT-4o</td>
      <td align="right">0.768</td>
      <td align="right">0.811</td>
    </tr>
    <tr>
      <td align="left">GPT-5.1</td>
      <td align="right">0.778</td>
      <td align="right">0.807</td>
    </tr>
    <tr>
      <td align="left">Gemini 2.5 Pro</td>
      <td align="right">0.807</td>
      <td align="right">0.821</td>
    </tr>
    <tr>
      <td align="left"><strong>Gemini 3 Pro</strong></td>
      <td align="right"><strong>0.853</strong></td>
      <td align="right"><strong>0.886</strong></td>
    </tr>
  </tbody>
</table>

成績來自[技術報告 Figure 5](docs/WattBot2025_AttentionPlease_Technical_Report.pdf#page=10)。最佳配置為 **HERO + Gemini 3 Pro（0.886）**；公開 HERO notebook 使用 **Gemini 2.5 Pro（報告成績 0.821）**。

### 主要發現

- **表格格式會影響檢索。** 增加內容不一定能提升證據排序品質。
- **視覺資訊需要獨立擷取。** OCR 讓圖像中的資訊進入文字檢索流程。
- **檢索與生成需要一起優化。** 報告中五種模型搭配 HERO 都有成績提升。
- **擴大上下文需要取捨。** 第二次生成能補回遺漏證據，也需要避免無證據推論。

<details>
<summary>檢索評估與重現細節</summary>

Baseline 比較使用 39 題，HERO 最後留存的評估使用 41 題，包含不可回答問題。HERO 輸出記錄 Recall@10 為 95.12%、MRR 為 0.8519，部分報告數值不同，因此不能視為同一樣本上的控制比較。

Notebook 將前 K 個結果命中任一標準引用的比例稱為 Recall@K；多個區塊可能來自同一文件，區塊排名與去重後的文件排名不同。

公開 HERO 檢索預設保留八個最終區塊。生成器雖可要求最多十個 fallback 區塊，仍受實際傳入證據數量限制。目前 query-based helper 對所有問題啟用重排序。拒答使用規則與 LLM 回應，不是經過校準的信心估計。

</details>

## 快速啟動

### 1. 開啟 Notebook

建議使用 Google Colab，配合 notebook 的 `/content/` 路徑；使用本機 Jupyter 時請先調整路徑。

- [Baseline notebook](notebook/v1_gemini_2_5pro.ipynb)
- [HERO notebook](notebook/Gemini_2_5_pro_新pipeline_0_821.ipynb)

### 2. 準備資料

將 `data/` 中的 `metadata.csv`、`train_QA.csv`、`test_Q.csv` 上傳至 `/content/`，確認資料、PDF 與區塊輸出路徑。

### 3. 安裝 HERO 套件

```python
%pip install docling easyocr opencv-python-headless rank-bm25 sentence-transformers faiss-cpu pandas numpy requests pdfminer.six pillow tqdm google-genai
```

Baseline 另外使用 Kaggle 登入、`pdfplumber`、`pdf2image`、`pytesseract`、`fuzzywuzzy` 與 `python-Levenshtein`；OCR cell 需要 Poppler 與 Tesseract 系統工具。

### 4. 設定 Gemini

將 notebook 手動設定 API key 與 client 的部分改為你的秘密管理方式。例如從環境變數讀取：

```python
import os
from google import genai

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
```

### 5. 執行流程

依序執行文件處理、索引、檢索與生成。探索性實驗、評估與清除快取 cell 請先檢查再執行。

HERO 生成階段的輸出位置：

```text
/content/submission_enhanced.csv
```

目前為需要手動設定的研究 notebook。套件未固定版本，模型／API 可用性與原始文獻下載狀態可能影響重現。

## 團隊成員與引用

**Attention Please · 國立政治大學**

<table width="100%">
  <thead>
    <tr>
      <th width="1000" align="left">成員</th>
      <th width="1000" align="left">技術報告所列系所</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left">Shao-Hua Wu‡</td>
      <td align="left">金融學系</td>
    </tr>
    <tr>
      <td align="left">Pei-Ju Hsieh‡</td>
      <td align="left">金融學系</td>
    </tr>
    <tr>
      <td align="left">Bo-Hao Chen‡</td>
      <td align="left">金融學系</td>
    </tr>
    <tr>
      <td align="left">Yi-Chen Hsiao</td>
      <td align="left">財務管理學系</td>
    </tr>
    <tr>
      <td align="left">Yi-Yang Xue</td>
      <td align="left">資訊科學系</td>
    </tr>
  </tbody>
</table>

‡ 技術報告標示共同貢獻。引用格式請見 [Citation](#citation)；相關研究與致謝收錄於技術報告。目前 repo 未提供授權檔案。

[回到英文版](#table-of-contents) · [回到中文目錄](#中文目錄)
