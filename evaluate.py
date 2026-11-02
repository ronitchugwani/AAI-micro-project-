"""Compare pipeline output with the ground-truth answer key.

Usage:  python evaluate.py                                   (dataset/ vs output/)
        python evaluate.py --dataset test_set --output output_test
Writes: <output>/evaluation.json, status_confusion.png, field_accuracy.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).parent
OUT = BASE / "output"

EXPECTED_RULE = {
    "total_mismatch": "R6", "duplicate_tax_row": "R5", "tax_rate_error": "R4",
    "due_before_invoice_date": "R7", "missing_invoice_number": "R1",
    "unregistered_vendor": "R8", "invalid_po_number": "R9", "high_value_over_50000": "R10",
}
FIELDS = ["vendor_name", "vendor_gstin", "invoice_no", "invoice_date", "due_date",
          "po_number", "subtotal", "printed_total", "line_items", "tax_rows"]
STATUSES = ["auto_approved", "needs_review", "rejected"]


def same(field, got, want):
    if field == "line_items":
        return len(got or []) == len(want) and all(
            g["qty"] == w["qty"] and abs(g["amount"] - w["amount"]) < 0.01 for g, w in zip(got, want))
    if field == "tax_rows":
        return len(got or []) == len(want) and all(
            g["label"] == w["label"] and abs(g["amount"] - w["amount"]) < 0.01 for g, w in zip(got, want))
    if field in ("subtotal", "printed_total"):
        return got is not None and abs(got - want) < 0.01
    if field == "vendor_name":
        return (got or "").lower() == (want or "").lower()
    return got == want


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(BASE / "dataset"))
    ap.add_argument("--output", default=str(OUT))
    args = ap.parse_args()
    OUT = Path(args.output)
    truth = json.loads((Path(args.dataset) / "ground_truth.json").read_text())
    results = {p.stem: json.loads(p.read_text()) for p in (OUT / "records").glob("*.json")}

    rows, field_hits, field_total = [], {f: 0 for f in FIELDS}, {f: 0 for f in FIELDS}
    confusion = {a: {p: 0 for p in STATUSES} for a in STATUSES}
    detected = caught = clean_total = clean_flagged = class_ok = 0

    for t in truth:
        r = results[Path(t["file_name"]).stem]
        rules = {f["rule"] for f in r.get("flags", [])}
        confusion[t["expected_status"]][r["status"]] += 1
        class_ok += (r["doc_type"] == "invoice") == (t["doc_type"] == "invoice")

        error = t["planted_error"]
        if error in EXPECTED_RULE:
            detected += 1
            caught += EXPECTED_RULE[error] in rules
        elif error == "none":
            clean_total += 1
            clean_flagged += r["status"] != "auto_approved"

        wrong_fields = []
        if t["doc_type"] == "invoice":
            for fld in FIELDS:
                if fld == "invoice_no" and t[fld] is None:
                    continue                      # deliberately blank, covered by R1
                field_total[fld] += 1
                if same(fld, r["fields"].get(fld), t[fld]):
                    field_hits[fld] += 1
                else:
                    wrong_fields.append(fld)

        rows.append({"file": t["file_name"], "expected": t["expected_status"], "got": r["status"],
                     "planted_error": error, "rules": sorted(rules - {"NOTE"}), "wrong_fields": wrong_fields})

    n = len(truth)
    status_ok = sum(confusion[s][s] for s in STATUSES)
    total_hits, total_checks = sum(field_hits.values()), sum(field_total.values())
    summary = {
        "documents": n,
        "status_accuracy": round(status_ok / n, 3),
        "classification_accuracy": round(class_ok / n, 3),
        "planted_errors_caught": f"{caught}/{detected}",
        "clean_invoices_wrongly_flagged": f"{clean_flagged}/{clean_total}",
        "field_accuracy_overall": round(total_hits / total_checks, 3),
        "field_accuracy": {f: round(field_hits[f] / field_total[f], 3) for f in FIELDS},
        "confusion_matrix": confusion,
        "per_document": rows,
    }
    (OUT / "evaluation.json").write_text(json.dumps(summary, indent=2))

    # ---- console report
    print(f"Documents evaluated:            {n}")
    print(f"Status accuracy:                {status_ok}/{n} ({summary['status_accuracy']:.0%})")
    print(f"Classification accuracy:        {class_ok}/{n}")
    print(f"Planted errors caught:          {caught}/{detected}")
    print(f"Clean invoices wrongly flagged: {clean_flagged}/{clean_total}")
    print(f"Field accuracy (overall):       {total_hits}/{total_checks} ({summary['field_accuracy_overall']:.1%})")
    for f in FIELDS:
        print(f"   {f:<15} {field_hits[f]}/{field_total[f]}")
    print("\nMismatches:")
    for row in rows:
        if row["expected"] != row["got"] or row["wrong_fields"]:
            print(f"  {row['file']:<24} expected {row['expected']:<14} got {row['got']:<14} "
                  f"rules={row['rules']} wrong_fields={row['wrong_fields']}")

    plot_confusion(confusion)
    plot_fields(summary["field_accuracy"], field_total["vendor_name"])


def plot_confusion(conf):
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    grid = [[conf[a][p] for p in STATUSES] for a in STATUSES]
    ax.imshow(grid, cmap="Blues")
    labels = ["Auto-approved", "Needs review", "Rejected"]
    ax.set_xticks(range(3), labels)
    ax.set_yticks(range(3), labels)
    ax.set_xlabel("System status")
    ax.set_ylabel("Expected status")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, grid[i][j], ha="center", va="center",
                    color="white" if grid[i][j] > 6 else "black", fontsize=13)
    ax.set_title("Status decisions vs. ground truth")
    fig.tight_layout()
    fig.savefig(OUT / "status_confusion.png", dpi=150)
    plt.close(fig)


def plot_fields(acc, n_invoices):
    names = {"vendor_name": "Vendor name", "vendor_gstin": "Vendor GSTIN", "invoice_no": "Invoice no.",
             "invoice_date": "Invoice date", "due_date": "Due date", "po_number": "PO number",
             "subtotal": "Subtotal", "printed_total": "Total", "line_items": "Line items", "tax_rows": "Tax rows"}
    items = sorted(acc.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.barh([names[k] for k, _ in items], [v * 100 for _, v in items], color="#3b6ea8")
    for i, (_, v) in enumerate(items):
        ax.text(v * 100 + 1, i, f"{v:.0%}", va="center", fontsize=9)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Extraction accuracy (%)")
    ax.set_title(f"Field-level extraction accuracy ({n_invoices} invoices)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "field_accuracy.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
