# Legal Contract Clause Extraction and Summarization

This project extracts key clauses from legal contracts (PDFs) and generates a summary of each contract. It focuses on three main clause types: **Termination**, **Confidentiality**, and **Liability**. The system uses a local LLM (LM Studio) for clause extraction and optional semantic search to prefilter relevant content. 

---

## 📂 Directory Structure

```

project/
│
├─ data/
│  └─ full_contract_pdf/       # Put your PDF contracts here
│
├─ outputs/
│  ├─ raw/                     # Raw LLM outputs
│  ├─ results.json             # Aggregated results in JSON
│  └─ results.csv              # Aggregated results in CSV
│
├─ main.py                     # Main script (the code)
└─ README.md

````

---

## ⚙️ Requirements

- Python 3.9+
- Install dependencies:

```bash
pip install requests pdfplumber pandas sentence-transformers numpy
````

* **LM Studio** running locally or on your network (`LM_STUDIO_HOST` environment variable).

Optional:

* `sentence-transformers` for semantic prefiltering.

---

## 🏃 How to Run

1. Place PDF contracts in `data/full_contract_pdf/`.
2. Set the LM Studio host (if different from default):

```bash
export LM_STUDIO_HOST="http://<your-host>:1234"
```

3. Run the main script:

```bash
python main.py
```

4. Outputs will be saved in `outputs/`:

   * `results.json`: structured JSON with clauses and offsets.
   * `results.csv`: table format for easy viewing.
   * `raw/`: raw LLM outputs for debugging.

---

## 📝 Approach / Workflow

The solution follows these steps:

```
PDF Contracts
      │
      ▼
  Extract Text (pdfplumber)
      │
      ▼
  Normalize Text (clean formatting)
      │
      ▼
  Chunk Text into overlapping segments
      │
      ▼
  Semantic Prefilter (optional)
      │
      ▼
  LLM Clause Extraction
      │
      ├─ Few-shot examples for accuracy
      ├─ Fallbacks:
      │     ├─ Regex extraction
      │     └─ Full-document LLM extraction
      ▼
  Aggregate & Deduplicate Clauses
      │
      ▼
  Pick Best Clause per Type (highest confidence)
      │
      ▼
  Generate Contract Summary (120-150 words)
      │
      ▼
  Save Outputs (JSON + CSV + Raw LLM text)
```

---

## 🧩 Key Features

* Extracts **Termination**, **Confidentiality**, and **Liability** clauses.
* Uses **exact text** with character offsets from PDFs.
* **Semantic prefilter** optionally improves efficiency.
* **Few-shot LLM examples** improve extraction accuracy.
* **Fallbacks**: regex + full-document LLM to ensure nothing is missed.
* Generates **contract summaries** strictly from document content.

---

## ⚠️ Notes / Limitations

* Currently, a **small LLM model** is used due to limited memory. This may limit accuracy for very long or complex contracts.
* If you have a **GPU**, you can:

  * Load a **larger GGUF model** in LM Studio for better clause extraction.
  * Increase `CHUNK_SIZE` and `MAX_DOCS` to process longer contracts more efficiently.
  * Enable full-document LLM extraction for improved accuracy without worrying about memory.
* Works best with clear, text-based PDFs.
* Semantic search requires `sentence-transformers`; otherwise, all chunks are processed.

---

## 🔧 Customization

* Change `PDF_ROOT` to point to your folder of PDFs.
* Adjust `MAX_DOCS` to limit number of PDFs processed.
* Modify `CLAUSE_TYPES` or add few-shot examples for other clause types.
* Adjust `CHUNK_SIZE` / `CHUNK_OVERLAP` for larger models and GPU usage.

---

## Author

Developed by Nidhi Sahani.
