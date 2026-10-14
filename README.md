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
- [Dataset and Evaluation](#dataset-and-evaluation)
- [Output Format](#output-format)
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

## Output Format

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

## Experimental Results

The experiments follow three questions: **does including tables help sparse retrieval, does HERO improve evidence retrieval, and do those improvements carry across language models?** Retrieval is evaluated on internal question sets; end-to-end WattBot Scores are reported for the 282-question competition test set.

### 1. The Table Paradox: Text-Only vs. Hybrid BM25

The baseline experiment compares a hybrid corpus containing text and raw Markdown tables with a text-only corpus. On the **39-query evaluation**, text-only retrieval matches or exceeds the hybrid version on every reported metric.

<table width="100%">
  <thead><tr>
    <th width="666" align="left">Metric</th>
    <th width="666" align="left">Hybrid: Text + Tables</th>
    <th width="666" align="left">Text-Only</th>
  </tr></thead>
  <tbody>
    <tr><td align="left">Recall@1</td><td align="right">79.49%</td><td align="right">87.18%</td></tr>
    <tr><td align="left">Recall@3</td><td align="right">89.74%</td><td align="right">92.31%</td></tr>
    <tr><td align="left">Recall@5</td><td align="right">92.31%</td><td align="right">92.31%</td></tr>
    <tr><td align="left">Recall@10</td><td align="right">92.31%</td><td align="right">94.87%</td></tr>
    <tr><td align="left">nDCG@1</td><td align="right">0.7949</td><td align="right">0.8718</td></tr>
    <tr><td align="left">nDCG@3</td><td align="right">0.8497</td><td align="right">0.8859</td></tr>
    <tr><td align="left">nDCG@5</td><td align="right">0.8564</td><td align="right">0.8859</td></tr>
    <tr><td align="left">nDCG@10</td><td align="right">0.8617</td><td align="right">0.8950</td></tr>
    <tr><td align="left">MRR</td><td align="right">0.8526</td><td align="right">0.8974</td></tr>
  </tbody>
</table>

Source: [report Table 1, page 4](docs/WattBot2025_AttentionPlease_Technical_Report.pdf#page=4). Here, “hybrid” means text plus tables, not sparse–dense retrieval.

Removing raw tables raises Recall@1 by **7.69 percentage points** and MRR by **0.0448**. The report attributes this to table-formatting noise: delimiters and disrupted structure dilute meaningful terms in BM25 scoring. Yet removing tables also removes numerical evidence. This trade-off motivates HERO's structured parsing, dense retrieval, and table-aware weighting.

### 2. HERO Retrieval Performance

The report compares the BM25 baseline with the complete HERO pipeline using coverage, ranking quality, and first-relevant-result position.

<table width="100%">
  <thead><tr>
    <th width="500" align="left">Metric</th>
    <th width="500" align="left">Baseline</th>
    <th width="500" align="left">HERO</th>
    <th width="500" align="left">Difference</th>
  </tr></thead>
  <tbody>
    <tr><td align="left">Recall@1</td><td align="right">79.49%</td><td align="right">80.49%</td><td align="right">+1.00%</td></tr>
    <tr><td align="left">Recall@3</td><td align="right">89.74%</td><td align="right">89.80%</td><td align="right">+0.06%</td></tr>
    <tr><td align="left">Recall@5</td><td align="right">92.31%</td><td align="right">92.68%</td><td align="right">+0.37%</td></tr>
    <tr><td align="left">Recall@10</td><td align="right">92.31%</td><td align="right">95.12%</td><td align="right">+2.81%</td></tr>
    <tr><td align="left">nDCG@1</td><td align="right">0.7949</td><td align="right">0.8049</td><td align="right">+0.0100</td></tr>
    <tr><td align="left">nDCG@3</td><td align="right">0.8497</td><td align="right">0.8502</td><td align="right">+0.0005</td></tr>
    <tr><td align="left">nDCG@5</td><td align="right">0.8564</td><td align="right">0.8584</td><td align="right">+0.0020</td></tr>
    <tr><td align="left">nDCG@10</td><td align="right">0.8617</td><td align="right">0.8665</td><td align="right">+0.0048</td></tr>
    <tr><td align="left">MRR</td><td align="right">0.8526</td><td align="right">0.8560</td><td align="right">+0.0034</td></tr>
  </tbody>
</table>

Source: [report Table 16, page 9](docs/WattBot2025_AttentionPlease_Technical_Report.pdf#page=9). Recall differences use `%` to show direct subtraction of the displayed percentages (percentage-point changes), not relative percentage growth. Values above reproduce the report; the evaluation caveat below explains differences from the saved notebook run.

- **Coverage:** reported Recall@10 increases from 92.31% to 95.12%, a gain of **2.81 percentage points**.
- **Early evidence:** Recall@1 increases by **1.00 percentage point**, indicating more questions have a relevant reference at the first retrieved position.
- **Ranking:** nDCG@10 increases by **0.0048**, while MRR increases by **0.0034** in the report.

These are results for the combined pipeline. They do not separately establish the causal contribution of OCR, fusion weights, or reranking; component-level attribution would require additional ablations.

### 3. Evidence Coverage After Visual-Aware Ingestion

Docling and EasyOCR expand the searchable corpus beyond prose. The report's corpus inventory contains **8,407 chunks**:

<table width="100%">
  <thead><tr>
    <th width="666" align="left">Chunk Type</th>
    <th width="666" align="left">Count</th>
    <th width="666" align="left">Share</th>
  </tr></thead>
  <tbody>
    <tr><td align="left">Text</td><td align="right">7,817</td><td align="right">93.0%</td></tr>
    <tr><td align="left">Table</td><td align="right">153</td><td align="right">1.8%</td></tr>
    <tr><td align="left">Figure OCR</td><td align="right">437</td><td align="right">5.2%</td></tr>
    <tr><td align="left">Total</td><td align="right">8,407</td><td align="right">100%</td></tr>
  </tbody>
</table>

Source: [report Table 8, page 6](docs/WattBot2025_AttentionPlease_Technical_Report.pdf#page=6). OCR contributes 437 figure-derived chunks, while 153 chunks preserve table content. Content labels allow the retrieval system to weight these sources differently from prose.

The report conclusion gives a different visual-chunk total; this inventory consistently follows Table 8. Corpus counts describe evidence availability, not a measured standalone OCR accuracy gain.

### 4. Pipeline and Model Comparison

<table width="100%">
  <thead><tr>
    <th width="500" align="left">Model</th>
    <th width="500" align="left">BM25 Baseline</th>
    <th width="500" align="left">HERO</th>
    <th width="500" align="left">Score Gain</th>
  </tr></thead>
  <tbody>
    <tr><td align="left">GPT-4o-mini</td><td align="right">0.753</td><td align="right">0.789</td><td align="right">+0.036</td></tr>
    <tr><td align="left">GPT-4o</td><td align="right">0.768</td><td align="right">0.811</td><td align="right">+0.043</td></tr>
    <tr><td align="left">GPT-5.1</td><td align="right">0.778</td><td align="right">0.807</td><td align="right">+0.029</td></tr>
    <tr><td align="left">Gemini 2.5 Pro</td><td align="right">0.807</td><td align="right">0.821</td><td align="right">+0.014</td></tr>
    <tr><td align="left">Gemini 3 Pro</td><td align="right">0.853</td><td align="right">0.886</td><td align="right">+0.033</td></tr>
  </tbody>
</table>

Source: [report Figure 5, page 10](docs/WattBot2025_AttentionPlease_Technical_Report.pdf#page=10). All five tested models improve with HERO.

**GPT-4o has the largest absolute gain (+0.043)**, while **Gemini 3 Pro achieves the highest final score (0.886)**. Gemini 3 Pro rises from 0.853 to 0.886, showing that retrieval improvements remain useful with a stronger answer model. Gemini 2.5 Pro rises from 0.807 to 0.821; this is the model initialized in the public HERO notebook.

### 5. Conclusions and Future Work

The experiments connect the table-noise problem to a practical retrieval redesign: preserve text, tables, and OCR as distinct evidence types; combine lexical and semantic search; rerank candidates; and adapt the answer context to the question.

The report proposes **learned fusion weights**, **multilingual corpora**, and **hierarchical indexing for larger collections** as future directions. The current weights and question routing remain heuristic. The two-stage generation design balances an initial narrow context against a broader fallback, but its independent benefit is not quantified in these tables.

<details>
<summary>Evaluation scope and reproducibility details</summary>

The baseline table experiment uses 39 queries. The final saved HERO evaluation uses 41, including unanswerable examples. Its saved values include Recall@3 = 87.80%, nDCG@3 = 0.8384, and MRR = 0.8519, differing from report Table 16. The report comparison should therefore be read as reported results rather than a newly verified same-sample evaluation; the text-only baseline also performs better on some metrics than the HERO figures shown above.

The notebook's Recall@K is an any-reference hit rate. Multiple chunks can come from the same document; chunk ranks and unique-document ranks are not interchangeable. The saved evaluation includes unanswerable entries, which also affects interpretation.

The public HERO pipeline defaults to eight final retrieval chunks; a generation request for ten fallback chunks can only use those supplied. Reranking is currently enabled for every query. Abstention uses heuristics and LLM responses rather than calibrated confidence. These figures are taken from the report and saved outputs, not a new execution of the models or competition evaluator.

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
- [資料集與評分方式](#資料集與評分方式)
- [輸出格式](#輸出格式)
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

## 輸出格式

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

## 實驗成果

實驗圍繞三個問題：**加入表格是否有利於 BM25、HERO 是否改善證據檢索，以及不同語言模型是否都能受益？** 檢索評估使用內部問題集，最終 WattBot Score 則來自 282 題競賽測試集的報告結果。

### 1. 表格悖論：純文字與混合 BM25 比較

Baseline 比較「文字加原始 Markdown 表格」與「純文字」兩種語料。在 **39 題評估**中，純文字版本的所有指標都優於或等於混合版本。

<table width="100%">
  <thead><tr>
    <th width="666" align="left">指標</th>
    <th width="666" align="left">混合：文字＋表格</th>
    <th width="666" align="left">純文字</th>
  </tr></thead>
  <tbody>
    <tr><td align="left">Recall@1</td><td align="right">79.49%</td><td align="right">87.18%</td></tr>
    <tr><td align="left">Recall@3</td><td align="right">89.74%</td><td align="right">92.31%</td></tr>
    <tr><td align="left">Recall@5</td><td align="right">92.31%</td><td align="right">92.31%</td></tr>
    <tr><td align="left">Recall@10</td><td align="right">92.31%</td><td align="right">94.87%</td></tr>
    <tr><td align="left">nDCG@1</td><td align="right">0.7949</td><td align="right">0.8718</td></tr>
    <tr><td align="left">nDCG@3</td><td align="right">0.8497</td><td align="right">0.8859</td></tr>
    <tr><td align="left">nDCG@5</td><td align="right">0.8564</td><td align="right">0.8859</td></tr>
    <tr><td align="left">nDCG@10</td><td align="right">0.8617</td><td align="right">0.8950</td></tr>
    <tr><td align="left">MRR</td><td align="right">0.8526</td><td align="right">0.8974</td></tr>
  </tbody>
</table>

來源：[報告第 4 頁 Table 1](docs/WattBot2025_AttentionPlease_Technical_Report.pdf#page=4)。這裡的 hybrid 指文字與表格混合，不是稀疏與稠密檢索。

移除原始表格後，Recall@1 提高 **7.69 個百分點**，MRR 增加 **0.0448**。報告將原因歸於表格格式雜訊：分隔符號與破碎結構稀釋關鍵詞的 BM25 訊號。然而，刪除表格也會失去數值證據。這個取捨促成 HERO 的結構化解析、語意檢索與表格加權設計。

### 2. HERO 檢索品質

報告以覆蓋率、排序品質與第一個相關結果的位置，比較 BM25 Baseline 與完整 HERO pipeline。

<table width="100%">
  <thead><tr>
    <th width="500" align="left">指標</th>
    <th width="500" align="left">Baseline</th>
    <th width="500" align="left">HERO</th>
    <th width="500" align="left">差異</th>
  </tr></thead>
  <tbody>
    <tr><td align="left">Recall@1</td><td align="right">79.49%</td><td align="right">80.49%</td><td align="right">+1.00%</td></tr>
    <tr><td align="left">Recall@3</td><td align="right">89.74%</td><td align="right">89.80%</td><td align="right">+0.06%</td></tr>
    <tr><td align="left">Recall@5</td><td align="right">92.31%</td><td align="right">92.68%</td><td align="right">+0.37%</td></tr>
    <tr><td align="left">Recall@10</td><td align="right">92.31%</td><td align="right">95.12%</td><td align="right">+2.81%</td></tr>
    <tr><td align="left">nDCG@1</td><td align="right">0.7949</td><td align="right">0.8049</td><td align="right">+0.0100</td></tr>
    <tr><td align="left">nDCG@3</td><td align="right">0.8497</td><td align="right">0.8502</td><td align="right">+0.0005</td></tr>
    <tr><td align="left">nDCG@5</td><td align="right">0.8564</td><td align="right">0.8584</td><td align="right">+0.0020</td></tr>
    <tr><td align="left">nDCG@10</td><td align="right">0.8617</td><td align="right">0.8665</td><td align="right">+0.0048</td></tr>
    <tr><td align="left">MRR</td><td align="right">0.8526</td><td align="right">0.8560</td><td align="right">+0.0034</td></tr>
  </tbody>
</table>

來源：[報告第 9 頁 Table 16](docs/WattBot2025_AttentionPlease_Technical_Report.pdf#page=9)。Recall 差異以 `%` 顯示兩個百分比直接相減的結果（百分點變化），並非相對成長率。上表忠實呈現報告值，與 notebook 留存輸出的差異另列於下方評估說明。

- **證據覆蓋率**：Recall@10 從 92.31% 提升至 95.12%，增加 **2.81 個百分點**。
- **首位命中**：Recall@1 增加 **1.00 個百分點**，代表更多問題能在第一個檢索位置找到相關引用。
- **排序品質**：報告中的 nDCG@10 增加 **0.0048**，MRR 增加 **0.0034**。

這些是完整 pipeline 的結果，不能單獨證明 OCR、融合權重或重排序各自造成多少提升；若要分離元件貢獻，仍需要額外消融實驗。

### 3. 視覺證據擷取後的語料組成

Docling 與 EasyOCR 將可搜尋內容從一般段落擴展至表格和圖像文字。報告的語料統計共 **8,407 個區塊**：

<table width="100%">
  <thead><tr>
    <th width="666" align="left">區塊類型</th>
    <th width="666" align="left">數量</th>
    <th width="666" align="left">占比</th>
  </tr></thead>
  <tbody>
    <tr><td align="left">文字</td><td align="right">7,817</td><td align="right">93.0%</td></tr>
    <tr><td align="left">表格</td><td align="right">153</td><td align="right">1.8%</td></tr>
    <tr><td align="left">圖像 OCR</td><td align="right">437</td><td align="right">5.2%</td></tr>
    <tr><td align="left">總計</td><td align="right">8,407</td><td align="right">100%</td></tr>
  </tbody>
</table>

來源：[報告第 6 頁 Table 8](docs/WattBot2025_AttentionPlease_Technical_Report.pdf#page=6)。其中 437 個區塊來自圖像 OCR，153 個區塊保留表格內容。類型標籤讓檢索階段能為不同證據來源配置權重。

報告結論另列不同的視覺區塊總數；此處統一採用 Table 8。區塊數反映可用證據規模，並不等於已量測的 OCR 獨立準確率提升。

### 4. 不同模型與 Pipeline 比較

<table width="100%">
  <thead><tr>
    <th width="500" align="left">模型</th>
    <th width="500" align="left">BM25 Baseline</th>
    <th width="500" align="left">HERO</th>
    <th width="500" align="left">分數提升</th>
  </tr></thead>
  <tbody>
    <tr><td align="left">GPT-4o-mini</td><td align="right">0.753</td><td align="right">0.789</td><td align="right">+0.036</td></tr>
    <tr><td align="left">GPT-4o</td><td align="right">0.768</td><td align="right">0.811</td><td align="right">+0.043</td></tr>
    <tr><td align="left">GPT-5.1</td><td align="right">0.778</td><td align="right">0.807</td><td align="right">+0.029</td></tr>
    <tr><td align="left">Gemini 2.5 Pro</td><td align="right">0.807</td><td align="right">0.821</td><td align="right">+0.014</td></tr>
    <tr><td align="left">Gemini 3 Pro</td><td align="right">0.853</td><td align="right">0.886</td><td align="right">+0.033</td></tr>
  </tbody>
</table>

來源：[報告第 10 頁 Figure 5](docs/WattBot2025_AttentionPlease_Technical_Report.pdf#page=10)。五種模型搭配 HERO 均有成績提升。

**GPT-4o 的絕對提升最大（+0.043）**，**Gemini 3 Pro 的最終分數最高（0.886）**。Gemini 3 Pro 從 0.853 提升至 0.886，說明即使搭配較強的生成模型，改善證據檢索仍有幫助。Gemini 2.5 Pro 則從 0.807 提升至 0.821，也是公開 HERO notebook 初始化使用的模型。

### 5. 結論與未來方向

實驗呈現從表格雜訊到檢索改造的完整脈絡：保留文字、表格與 OCR 類型，結合詞彙與語意搜尋，再重排序候選證據，依問題調整生成上下文。

報告提出未來可研究**學習式融合權重**、**多語言語料**與**大型文件集合的階層索引**。目前權重與問題路由仍是啟發式規則；兩階段生成在較小上下文與較廣 fallback 之間取捨，但上表未量化其獨立效果。

<details>
<summary>評估範圍與重現細節</summary>

Baseline 表格實驗使用 39 題；HERO 最後留存的評估使用 41 題，包含不可回答問題。其輸出記錄 Recall@3 為 87.80%、nDCG@3 為 0.8384、MRR 為 0.8519，與報告 Table 16 不同。因此報告比較應視為文件記錄的結果，而非本次驗證的同樣本實驗；純文字 baseline 在部分指標上也高於上述 HERO 數值。

Notebook 的 Recall@K 判斷前 K 個結果是否命中任一標準引用，實際上是命中率。多個區塊可來自同一文件，區塊排名與去重文件排名不能混用。留存評估也包含不可回答問題，會影響指標解讀。

公開 HERO 檢索預設保留八個最終區塊；生成器要求十個 fallback 區塊時，仍只能使用實際傳入的內容。目前所有問題均啟用重排序，拒答由規則與 LLM 回應決定，不是校準後的信心估計。本節數據來自報告與留存輸出，沒有重新執行模型或競賽評分。

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
