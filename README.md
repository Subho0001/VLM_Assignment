# Intelligent VLM Document Extraction Pipeline

This repository contains an end-to-end Vision-Language Model (VLM) pipeline designed for intelligent document understanding. The system processes raw PDF files, analyzes their layout and text, and extracts structured data without relying on paid APIs or traditional OCR engines.

The pipeline performs three core tasks zero-shot:
1. **Key-Value Extraction:** Parsing receipts/invoices into structured JSON format.
2. **Signature Detection:** Identifying the presence of handwritten signatures (Binary classification).
3. **Form Field Analysis:** Detecting form fields/checkboxes and determining if they are filled or empty.

## ⚙️ Setup Instructions

### Prerequisites
* **Python Version:** Python 3.10 or higher is recommended.
* **System Dependencies:** You must install `poppler` on your machine, as it is required by `pdf2image` to render PDFs.
  * **Ubuntu/Debian (Colab):** `sudo apt-get install poppler-utils`
  * **macOS:** `brew install poppler`
  * **Windows:** Download Poppler for Windows and add it to your PATH.

### 1. Install Python Packages
Clone the repository and install the required dependencies:
```bash
pip install torch transformers datasets pdf2image pillow accelerate
```

## 🏗️ Pipeline Architecture

The system is built on a modular, three-stage architecture:

### Module 1: PDF Processing (`pdf_to_image`)
Multi-page PDFs are ingested and converted into a list of high-quality RGB PIL Images. To prevent CUDA Out-of-Memory (OOM) errors during the VLM forward pass, a dynamic resizing algorithm caps the maximum dimension of any page to 800 pixels while perfectly preserving the aspect ratio. This maintains OCR fidelity while drastically shrinking the token payload.

### Module 2: Intelligent Routing & Inference (`vlm_engine`)
This is the core VLM integration. The module acts as an intelligent router, mapping specific document tasks to highly constrained prompt templates:
* **JSON Extraction Prompt:** Forces the model to ignore conversational filler and return raw nested JSON.
* **Signature Prompt:** Constrains the output space to a strict "Yes" or "No" binary.
* **Form Fields Prompt:** Instructs the model to output a status array for all detected input fields.

**Inference Parameters:** The generation engine explicitly uses **Greedy Decoding** (`do_sample=False`). By disabling temperature and creative sampling, the model is forced into deterministic extraction, virtually eliminating the "hallucinations" (invented items or prices) common in generative models.

### Module 3: Output Sanitization & Evaluation (`eval_metrics`)
Because LLMs occasionally omit syntax (like closing brackets) which causes standard `json.loads()` to crash, this module introduces a **Regex Fallback Parser**. If strict JSON parsing fails, the regex sweeps the raw model string to rescue all valid `"key": "value"` pairs. The extracted pairs are then flattened into sets and evaluated against the CORD-v2 ground truth using normal strict accuracy scores, Precision, Recall, and F1 scoring.

---

## 🧠 VLM Selection & Justification

**Selected Model:** `HuggingFaceTB/SmolVLM-500M-Instruct`

**Why this model?**
1. **Architectural Efficiency:** At 500 million parameters, this model represents the ideal balance between capability and least model complexity. It can be loaded in standard precision or 4-bit quantization, easily fitting into the VRAM of standard consumer GPUs (or free Google Colab tiers) without requiring API calls or heavy compute clusters.
2. **Instruction Fine-Tuning:** By choosing the `-Instruct` variant rather than a base completion model, the VLM is already mathematically aligned to follow complex behavioral instructions. It responds significantly better to formatting constraints (like "Output strictly as JSON" or "Answer only Yes or No") compared to base models of similar size.
3. **Zero-Shot Generalization:** Despite its small size, it demonstrates robust spatial reasoning and OCR capabilities out-of-the-box, allowing it to navigate the highly varied, inconsistent layouts of datasets like CORD-v2 without requiring immediate task-specific fine-tuning.

