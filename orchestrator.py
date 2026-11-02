"""Orchestrator - runs the six agents in order for every document.

Usage:
    python orchestrator.py                       # process dataset/documents
    python orchestrator.py path/to/invoice.pdf   # process one file
    python orchestrator.py some/folder -o output
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

from agents import (ClassificationAgent, ExtractionAgent, IngestionAgent,
                    OCRAgent, OutputAgent, ValidationAgent)

BASE = Path(__file__).parent


class Orchestrator:
    def __init__(self, reference_dir=BASE / "dataset", out_dir=BASE / "output", retry_dpi=400):
        self.ingestion = IngestionAgent(dpi=300)
        self.ocr = OCRAgent(min_confidence=80.0)
        self.classifier = ClassificationAgent()
        self.extractor = ExtractionAgent()
        self.validator = ValidationAgent(reference_dir)
        self.output = OutputAgent(out_dir)
        self.retry_dpi = retry_dpi

    def process(self, file_path, keep_pages=False):
        start = time.perf_counter()
        record = {"file_path": str(file_path), "file_name": Path(file_path).name, "audit_trail": []}

        def step(agent, **kwargs):
            t0 = time.perf_counter()
            agent.run(record, **kwargs)
            record["audit_trail"].append({
                "agent": agent.name, "at": datetime.now().isoformat(timespec="seconds"),
                "ms": round((time.perf_counter() - t0) * 1000), **kwargs})

        try:
            step(self.ingestion)
            step(self.ocr)
            # Retry once at higher resolution instead of trusting a poor read
            if record["ocr_low_confidence"]:
                first_conf = record["ocr_confidence"]
                step(self.ingestion, dpi=self.retry_dpi)
                step(self.ocr)
                record["audit_trail"].append({"agent": "Orchestrator", "event": "ocr_retry",
                                              "confidence_before": first_conf,
                                              "confidence_after": record["ocr_confidence"]})
            step(self.classifier)
            if record["doc_type"] == "invoice":
                step(self.extractor)
                step(self.validator)
            else:
                record["audit_trail"].append({"agent": "Orchestrator", "event": "skipped_extraction",
                                              "reason": "not an invoice"})
        except Exception as exc:                         # never lose a document silently
            record.setdefault("doc_type", "invoice")
            record.setdefault("fields", {})
            record.setdefault("sources", {})
            record["flags"] = [{"rule": "ERROR", "severity": "High",
                                "message": f"Processing failed: {exc}", "field": None, "source": None}]

        record["processing_seconds"] = round(time.perf_counter() - start, 2)
        step(self.output)
        if not keep_pages:
            record.pop("pages", None)                    # free the images
        return record

    def process_batch(self, paths):
        records = []
        for p in paths:
            r = self.process(p)
            records.append(r)
            print(f"{r['file_name']:<24} {r['status']:<14} {r.get('priority') or '':<7} "
                  f"{r['processing_seconds']:>5}s  {r['reason'][:70]}")
        csv_path = self.output.write_batch_csv(records)
        self.print_summary(records, csv_path)
        return records

    @staticmethod
    def print_summary(records, csv_path):
        counts = {}
        for r in records:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        avg = sum(r["processing_seconds"] for r in records) / len(records)
        print("\n" + "=" * 60)
        print(f"Processed {len(records)} documents, average {avg:.1f}s each")
        for status in ("auto_approved", "needs_review", "rejected"):
            print(f"  {status:<14} {counts.get(status, 0)}")
        print(f"Batch CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Document Intelligence & Extraction Agent")
    parser.add_argument("input", nargs="?", default=str(BASE / "dataset" / "documents"),
                        help="a document file or a folder of documents")
    parser.add_argument("-o", "--out", default=str(BASE / "output"), help="output folder")
    parser.add_argument("-r", "--reference", default=str(BASE / "dataset"),
                        help="folder with vendors.csv, purchase_orders.csv, business_rules.json")
    args = parser.parse_args()

    target = Path(args.input)
    paths = sorted(p for p in target.iterdir()
                   if p.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg"}) if target.is_dir() else [target]
    Orchestrator(reference_dir=args.reference, out_dir=args.out).process_batch(paths)


if __name__ == "__main__":
    main()
