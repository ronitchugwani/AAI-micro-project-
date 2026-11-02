"""Agent 5 - Validation Agent (cross-field checks + reference & business rules).

Implements rules R1-R11 from the design document. Each failed rule becomes a
flag with a rule id, severity and a plain-English message. Values are never
changed to make a check pass.
"""
import csv
import json
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path


def rs(x):
    return f"Rs. {x:,.2f}"


class ValidationAgent:
    name = "ValidationAgent"

    def __init__(self, reference_dir):
        ref = Path(reference_dir)
        with open(ref / "vendors.csv", newline="") as f:
            self.vendors = list(csv.DictReader(f))
        with open(ref / "purchase_orders.csv", newline="") as f:
            self.po_numbers = {row["po_number"] for row in csv.DictReader(f)}
        with open(ref / "business_rules.json") as f:
            self.rules = json.load(f)

    def run(self, record):
        f = record["fields"]
        tol = self.rules["total_tolerance"]
        flags = []

        def flag(rule, severity, message, field=None):
            flags.append({"rule": rule, "severity": severity, "message": message,
                          "field": field, "source": record["sources"].get(field) if field else None})

        # R1 required fields
        missing = [k for k in self.rules["required_fields"] if f.get(k) in (None, [], "")]
        for k in missing:
            flag("R1", "High", f"Required field '{k}' is missing or unreadable", k)

        items = f.get("line_items") or []
        subtotal = f.get("subtotal")
        taxes = f.get("tax_rows") or []
        total = f.get("printed_total")

        # R2 line-item arithmetic
        for it in items:
            if it["qty"] is not None and abs(it["qty"] * it["rate"] - it["amount"]) > tol:
                flag("R2", "Medium", f"Line '{it['description']}': {it['qty']} x {rs(it['rate'])} "
                                     f"= {rs(it['qty'] * it['rate'])}, but amount printed is {rs(it['amount'])}",
                     "line_items")

        # R3 subtotal
        if items and subtotal is not None:
            item_sum = sum(it["amount"] for it in items)
            if abs(item_sum - subtotal) > tol:
                flag("R3", "High", f"Line items add up to {rs(item_sum)} but subtotal is {rs(subtotal)} "
                                   f"(difference {rs(abs(item_sum - subtotal))})", "subtotal")

        # R4 tax rate and R5 duplicate tax rows
        rates = self.rules["gst_rates"]
        seen = {}
        for t in taxes:
            seen[t["type"]] = seen.get(t["type"], 0) + 1
            if subtotal is not None and t["type"] in rates:
                expected = round(subtotal * rates[t["type"]], 2)
                if abs(expected - t["amount"]) > tol:
                    flag("R4", "High", f"{t['label']} should be {rs(expected)} on a subtotal of "
                                       f"{rs(subtotal)}, but {rs(t['amount'])} is printed", "tax_rows")
        for typ, count in seen.items():
            if count > 1:
                flag("R5", "High", f"{typ} row appears {count} times; tax may be charged more than once",
                     "tax_rows")

        for label in record.get("unread_tax_rows", []):
            flag("R1", "High", f"Tax row '{label}' has no readable amount", "tax_rows")

        # R6 grand total = subtotal + correct tax (one row per tax type);
        # with no readable tax rows, check against every configured GST rate
        if subtotal is not None and total is not None:
            types = seen or rates
            expected_total = round(subtotal + sum(subtotal * rates.get(typ, 0) for typ in types), 2)
            if abs(expected_total - total) > tol:
                flag("R6", "High", f"Printed total {rs(total)} does not match subtotal + tax "
                                   f"{rs(expected_total)} (difference {rs(total - expected_total)})",
                     "printed_total")

        # R7 date logic
        if f.get("invoice_date") and f.get("due_date"):
            if date.fromisoformat(f["due_date"]) < date.fromisoformat(f["invoice_date"]):
                flag("R7", "Medium", f"Due date {f['due_date']} is before invoice date {f['invoice_date']}",
                     "due_date")

        # R8 vendor master (a GSTIN with an OCR ambiguity is resolved here)
        vendor = None
        for cand in record.get("gstin_candidates") or [f.get("vendor_gstin")]:
            vendor = self.match_vendor(f.get("vendor_name"), cand)
            if vendor and vendor["gstin"] == cand:
                if cand != f.get("vendor_gstin"):
                    flags.append({"rule": "NOTE", "severity": "Low", "field": "vendor_gstin",
                                  "source": record["sources"].get("vendor_gstin"),
                                  "message": f"GSTIN was ambiguous after OCR; read as {cand}, "
                                             f"which matches the vendor master"})
                    f["vendor_gstin"] = cand
                break
        record["vendor_match"] = vendor
        if f.get("vendor_name") and vendor is None:
            flag("R8", "High", f"Vendor '{f['vendor_name']}' (GSTIN {f.get('vendor_gstin')}) "
                               f"is not in the vendor master", "vendor_name")
        elif vendor and f.get("vendor_gstin") and f["vendor_gstin"] != vendor["gstin"]:
            flag("R8", "High", f"GSTIN {f['vendor_gstin']} does not match the registered GSTIN "
                               f"{vendor['gstin']} for {vendor['vendor_name']}", "vendor_gstin")

        # R9 purchase order
        if f.get("po_number") and f["po_number"] not in self.po_numbers:
            flag("R9", "Medium", f"PO number {f['po_number']} does not exist in the purchase-order list",
                 "po_number")

        # R10 approval threshold
        limit = self.rules["high_value_threshold"]
        if total is not None and total > limit:
            flag("R10", "Medium", f"Invoice total {rs(total)} exceeds the {rs(limit)} approval threshold",
                 "printed_total")

        # R11 OCR confidence (after retry)
        if record.get("ocr_low_confidence"):
            flag("R11", "Medium", f"Average OCR confidence is low ({record['ocr_confidence']}%) "
                                  f"even after a higher-resolution retry")

        # Low-confidence notes from extraction: shown to reviewers, do not block approval
        for note in record.get("extraction_notes", []):
            flags.append({"rule": "NOTE", "severity": "Low", "message": note, "field": None, "source": None})

        record["flags"] = flags
        return record

    def match_vendor(self, name, gstin):
        if gstin:
            for v in self.vendors:
                if v["gstin"] == gstin:
                    return v
        if name:
            best = max(self.vendors, key=lambda v: SequenceMatcher(
                None, v["vendor_name"].lower(), name.lower()).ratio())
            if SequenceMatcher(None, best["vendor_name"].lower(), name.lower()).ratio() >= 0.85:
                return best
        return None
