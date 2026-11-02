"""Agent 3 - Document Classification Agent.

Decides whether a document is an invoice using keyword evidence, and names
the other document type when it can recognise one.
"""
import re

INVOICE_MARKERS = {
    r"tax\s+invoice": 3,
    r"invoice\s*no": 2,
    r"invoice\s*date": 2,
    r"gstin": 1,
    r"subtotal": 1,
    r"\bcgst\b|\bsgst\b|\bigst\b": 2,
    r"^\s*total\b": 1,
}
OTHER_MARKERS = {
    "delivery_challan": r"delivery\s+challan|challan\s+no",
    "quotation": r"\bquotation\b|pleased\s+to\s+quote",
    "purchase_order": r"purchase\s+order\b(?!.*number)",
}
NEGATIONS = r"not\s+a\s+tax\s+invoice"


class ClassificationAgent:
    name = "ClassificationAgent"

    def __init__(self, threshold=6):
        self.threshold = threshold

    def run(self, record):
        text = record["raw_text"].lower()
        score = sum(w for pat, w in INVOICE_MARKERS.items()
                    if re.search(pat, text, re.MULTILINE))
        other = next((name for name, pat in OTHER_MARKERS.items() if re.search(pat, text)), None)
        if re.search(NEGATIONS, text):
            score -= 5

        max_score = sum(INVOICE_MARKERS.values())
        if score >= self.threshold and other is None:
            record["doc_type"] = "invoice"
        else:
            record["doc_type"] = "other"
            record["other_type"] = other or "unknown"
        record["classification_confidence"] = round(max(score, 0) / max_score, 2)
        return record
