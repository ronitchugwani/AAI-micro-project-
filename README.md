# Document Intelligence & Extraction Agent

A multi-agent Python system that reads vendor invoices (PDF, scanned image or
phone photo), extracts their fields, validates them against business rules and
reference data, and decides whether each document can be **auto-approved**,
**needs human review**, or should be **rejected**, with a reason for every flag.

## Project structure

```
doc_intel_agent/
├── agents/
│   ├── ingestion.py        Agent 1  load file, PDF -> images, greyscale, deskew
│   ├── ocr.py              Agent 2  Tesseract OCR, line boxes, confidence, weak-line re-read
│   ├── classification.py   Agent 3  invoice vs. other (keyword evidence)
│   ├── extraction.py       Agent 4  header fields, line items, tax rows, totals
│   ├── validation.py       Agent 5  rules R1-R11 (arithmetic, tax, dates, vendor, PO, threshold)
│   └── output.py           Agent 6  status, priority, JSON records, results.csv
├── orchestrator.py         runs the agents in order, retries, audit trail (main entry point)
├── evaluate.py             compares output with ground truth, draws charts
├── app.py                  Streamlit demo page (single document + batch dashboard)
├── generate_dataset.py     creates the synthetic invoice dataset
├── dataset/                24 documents + vendors.csv, purchase_orders.csv,
│                           business_rules.json, ground_truth.json/.csv
├── output/                 results on dataset/ (records, results.csv, evaluation, charts)
└── output_test/            results on the unseen test set (seed 11)
```

## Setup

**1. Python packages** (Python 3.11 or newer), in a virtual environment

```
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Tesseract OCR and Poppler** (system programs, not pip packages)

| OS | Tesseract | Poppler (for PDFs) |
|---|---|---|
| Windows | Installer from https://github.com/UB-Mannheim/tesseract/wiki | Zip from https://github.com/oschwartz10612/poppler-windows/releases, extract it |
| macOS | `brew install tesseract` (and `brew install python@3.11`) | `brew install poppler` |
| Ubuntu | `sudo apt install tesseract-ocr` | `sudo apt install poppler-utils` |

On **Windows**, if they are not on your PATH, tell the project where they are
(Command Prompt, before running):

```
set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
set POPPLER_PATH=C:\poppler-24.08.0\Library\bin
```

Check the install with `tesseract --version`.

## Running it

```
python orchestrator.py                          # process all 24 documents in dataset/documents
python orchestrator.py dataset/documents/invoice_012_scan.png   # one file
python evaluate.py                              # accuracy report + charts in output/
streamlit run app.py                            # demo web page
```

Test on documents the system has never seen:

```
python generate_dataset.py --seed 11 --out test_set
python orchestrator.py test_set/documents -r test_set -o output_test
python evaluate.py --dataset test_set --output output_test
```

## How a document flows

1. **Ingestion**: PDFs are rendered at 300 DPI; every page is greyscaled,
   contrast-stretched and deskewed (projection-profile method, ±6°).
2. **OCR**: Tesseract reads the page line by line, keeping each line's bounding
   box and confidence. Lines below 60% confidence are re-read from an enlarged,
   denoised crop; both readings are kept. If the page average is below 80%, the
   orchestrator re-runs ingestion and OCR at 400 DPI.
3. **Classification**: keyword scoring decides invoice vs. other (delivery
   challan, quotation). Non-invoices go straight to `rejected`.
4. **Extraction**: regular expressions keyed on printed labels pull out every
   field and record which line it came from. A quantity that cannot be read is
   derived from amount ÷ rate and noted; a field that cannot be found is left
   empty, never guessed.
5. **Validation**: rules R1-R11 (see table). An ambiguous OCR'd GSTIN is resolved
   against the vendor master.
6. **Output**: any High/Medium flag → `needs_review` (routed to Finance),
   otherwise `auto_approved`. One JSON record per document plus `results.csv`.

| Rule | Check |
|---|---|
| R1 | Required fields present |
| R2 | Each line: qty × rate = amount |
| R3 | Line items add up to subtotal |
| R4 | CGST/SGST = 9% of subtotal |
| R5 | No duplicated tax row |
| R6 | Subtotal + tax = printed total |
| R7 | Due date not before invoice date |
| R8 | Vendor and GSTIN in `vendors.csv` |
| R9 | PO number in `purchase_orders.csv` |
| R10 | Total ≤ Rs. 50,000 (from `business_rules.json`) |
| R11 | OCR confidence acceptable after retry |

Thresholds live in `dataset/business_rules.json` and can be changed without code changes.

## Results

| Metric | Development set (`dataset/`, seed 42) | Unseen test set (seed 11) |
|---|---|---|
| Status accuracy | 24/24 (100%) | 24/24 (100%) |
| Classification accuracy | 24/24 | 24/24 |
| Planted errors sent to review | 11/11 | 11/11 |
| Planted errors caught by the expected rule | 11/11 | 10/11 |
| Clean invoices wrongly flagged | 0/11 | 0/11 |
| Field extraction accuracy (incl. tax rows) | 218/219 (99.5%) | 218/219 (99.5%) |
| Average time per document | about 0.6 s | about 0.6 s |

Measured with Tesseract 5.5.4. The rules were tuned on the development set,
so the **unseen test set is the fair measure**. The remaining misses are OCR
misreads that the system handles safely rather than guessing:

- `dataset/invoice_007_photo.jpg`: the due date `08/11/2026` is read as
  `08/1 4/2026` (month 14), so the field is left empty and flagged (R1).
- `test_set/invoice_020_photo.jpg`: the tax amounts on this photo cannot be
  read. The planted 12% CGST error is therefore caught through R1 (unreadable
  tax row) and R6 (total does not match subtotal + 18% GST) instead of R4, and
  the invoice still goes to review.

## Limitations

- Rule-based extraction is tuned to this GST invoice layout; very different
  layouts would need new patterns or a learned model (e.g. LayoutLM, Donut).
- English, printed, single-page, intra-state GST (CGST + SGST) invoices only.
- Synthetic data: real scans may be noisier than the simulated ones.
