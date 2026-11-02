"""Agent 6 - Output Agent (scoring, review routing and structured output).

Turns the flags into a status and priority, then writes one JSON record per
document plus a combined CSV for the whole batch.
"""
import csv
import json
from pathlib import Path

SEVERITY_ORDER = {"High": 3, "Medium": 2, "Low": 1}
REVIEW_QUEUE = {"invoice": "Finance - Accounts Payable"}


class OutputAgent:
    name = "OutputAgent"

    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        (self.out_dir / "records").mkdir(parents=True, exist_ok=True)

    def run(self, record):
        blocking = [f for f in record.get("flags", []) if f["severity"] in ("High", "Medium")]

        if record["doc_type"] != "invoice":
            record["status"] = "rejected"
            record["priority"] = None
            record["reason"] = f"Document is not an invoice ({record.get('other_type', 'unknown')} detected)"
        elif blocking:
            record["status"] = "needs_review"
            top = max(blocking, key=lambda f: SEVERITY_ORDER[f["severity"]])
            record["priority"] = top["severity"]
            record["review_queue"] = REVIEW_QUEUE["invoice"]
            record["reason"] = "; ".join(f"[{f['rule']}] {f['message']}" for f in blocking)
        else:
            record["status"] = "auto_approved"
            record["priority"] = None
            record["reason"] = "All checks passed"

        out = self.export(record)
        with open(self.out_dir / "records" / f"{Path(record['file_name']).stem}.json", "w") as fh:
            json.dump(out, fh, indent=2)
        return record

    @staticmethod
    def export(record):
        """The downstream-facing record: no images, no raw OCR lines."""
        keep = ["file_name", "doc_type", "status", "priority", "review_queue", "reason",
                "ocr_confidence", "deskew_applied", "dpi_used", "page_count", "fields",
                "flags", "sources", "audit_trail", "processing_seconds"]
        out = {k: record.get(k) for k in keep if k in record}
        out["source_ref"] = f"{record['file_name']}#p1"
        return out

    def write_batch_csv(self, records):
        cols = ["file_name", "doc_type", "status", "priority", "vendor_name", "vendor_gstin",
                "invoice_no", "invoice_date", "due_date", "po_number", "subtotal", "tax_total",
                "printed_total", "ocr_confidence", "flag_rules", "reason", "processing_seconds"]
        path = self.out_dir / "results.csv"
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in records:
                f = r.get("fields", {})
                w.writerow({
                    "file_name": r["file_name"], "doc_type": r["doc_type"], "status": r["status"],
                    "priority": r.get("priority"), "vendor_name": f.get("vendor_name"),
                    "vendor_gstin": f.get("vendor_gstin"), "invoice_no": f.get("invoice_no"),
                    "invoice_date": f.get("invoice_date"), "due_date": f.get("due_date"),
                    "po_number": f.get("po_number"), "subtotal": f.get("subtotal"),
                    "tax_total": round(sum(t["amount"] for t in f.get("tax_rows", [])), 2) if f else None,
                    "printed_total": f.get("printed_total"), "ocr_confidence": r.get("ocr_confidence"),
                    "flag_rules": " ".join(sorted({x["rule"] for x in r.get("flags", []) if x["rule"] != "NOTE"})),
                    "reason": r.get("reason"), "processing_seconds": r.get("processing_seconds"),
                })
        return path
