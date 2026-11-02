"""
Synthetic invoice dataset generator for the Document Intelligence micro project.

Creates GST-style Indian vendor invoices as PDFs, "scanned" PNGs and "phone photo"
JPGs, with deliberately planted errors and a ground-truth answer key.

Usage:  python generate_dataset.py                          (writes ./dataset, seed 42)
        python generate_dataset.py --seed 7 --out test_set   (a fresh, unseen test set)
Needs:  pip install reportlab pillow numpy   and poppler (pdftoppm) on PATH
"""
import argparse, csv, json, random, shutil, subprocess
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

OUT = Path(__file__).parent / "dataset"
BUYER = {"name": "Vertex Office Solutions Pvt Ltd", "gstin": "27AABCV1234F1Z5",
         "address": "Unit 12, Andheri East, Mumbai 400069"}

# ---------------------------------------------------------------- reference data
VENDORS = [
    ("V001", "Shree Ganesh Traders", "27AAEPG4521K1Z3", "Mumbai"),
    ("V002", "Patel Stationery Mart", "27ABHPP7781L1Z9", "Thane"),
    ("V003", "Om Sai Electricals", "27AACFO3390M1Z2", "Pune"),
    ("V004", "Kumar Packaging Industries", "27AADFK6612N1Z7", "Navi Mumbai"),
    ("V005", "Lakshmi Computer Services", "27AAGPL9045P1Z4", "Mumbai"),
    ("V006", "Sharma Furniture Works", "27ABCPS2287Q1Z1", "Nashik"),
    ("V007", "Deccan Cleaning Supplies", "27AAJFD5530R1Z6", "Pune"),
    ("V008", "Mehta Printing Press", "27AAKPM8814S1Z8", "Mumbai"),
]
UNREGISTERED = [
    ("Balaji Enterprises", "27AAZPB1199T1Z0", "Kalyan"),
    ("Royal Tech Distributors", "27AAYFR4470U1Z5", "Mumbai"),
]
CATALOG = {
    "V001": [("A4 Copier Paper (500 sheets)", 280), ("Ball Pens (box of 50)", 350), ("Stapler Pins (pack)", 40), ("File Folders", 25)],
    "V002": [("Whiteboard Markers (set)", 220), ("Spiral Notebooks", 60), ("Desk Organiser", 450), ("Sticky Notes Pad", 35)],
    "V003": [("LED Tube Light 20W", 320), ("Extension Board 6-way", 550), ("Ceiling Fan", 2100), ("MCB 32A", 380)],
    "V004": [("Corrugated Box Large", 45), ("Bubble Wrap Roll", 650), ("Packing Tape (6 rolls)", 300), ("Stretch Film Roll", 520)],
    "V005": [("Laptop Servicing", 1500), ("Wireless Mouse", 650), ("USB Keyboard", 800), ("Network Cable Cat6 (m)", 18)],
    "V006": [("Office Chair Ergonomic", 6500), ("Steel Filing Cabinet", 8200), ("Work Desk 4ft", 5400), ("Visitor Chair", 2300)],
    "V007": [("Floor Cleaner 5L", 480), ("Hand Wash 5L", 520), ("Garbage Bags (pack)", 120), ("Mop Set", 650)],
    "V008": [("Visiting Cards (100)", 250), ("Letterheads (500)", 1800), ("Flex Banner (sq ft)", 22), ("Brochures A4 (100)", 1400)],
    "UNREG": [("Printer Toner Cartridge", 2400), ("UPS 1kVA", 5200), ("Router Dual Band", 2900), ("HDMI Cable 2m", 350)],
}

# --------------------------------------------------------------- document specs
# (planted_error, output_format). Clean docs come first.
SPECS = (
    [("none", "pdf")] * 6 + [("none", "scan")] * 3 + [("none", "photo")] * 2 +
    [("total_mismatch", "pdf"), ("total_mismatch", "scan"),
     ("duplicate_tax_row", "pdf"), ("duplicate_tax_row", "scan"),
     ("unregistered_vendor", "pdf"), ("unregistered_vendor", "photo"),
     ("missing_invoice_number", "pdf"),
     ("due_before_invoice_date", "scan"),
     ("high_value_over_50000", "pdf"),
     ("invalid_po_number", "pdf"),
     ("tax_rate_error", "photo")]
)

EXPECTED_STATUS = {
    "none": "auto_approved",
    "total_mismatch": "needs_review",
    "duplicate_tax_row": "needs_review",
    "unregistered_vendor": "needs_review",
    "missing_invoice_number": "needs_review",
    "due_before_invoice_date": "needs_review",
    "high_value_over_50000": "needs_review",
    "invalid_po_number": "needs_review",
    "tax_rate_error": "needs_review",
}


def money(x):
    return f"{x:,.2f}"


def build_invoice(idx, error):
    if error == "unregistered_vendor":
        name, gstin, city = UNREGISTERED.pop(0)
        vendor = {"id": None, "name": name, "gstin": gstin, "city": city}
        items_pool = CATALOG["UNREG"]
    else:
        vid, name, gstin, city = random.choice(VENDORS)
        vendor = {"id": vid, "name": name, "gstin": gstin, "city": city}
        items_pool = CATALOG[vid]

    n_items = random.randint(2, 4)
    items = []
    for desc, rate in random.sample(items_pool, n_items):
        qty = random.choice([1, 2, 3, 4, 5, 10, 12, 20, 25])
        if rate > 2000:
            qty = random.choice([1, 2, 3])
        items.append({"description": desc, "qty": qty, "rate": float(rate), "amount": float(qty * rate)})

    if error == "high_value_over_50000":   # make sure it crosses the threshold
        items[0]["qty"] = max(items[0]["qty"], 1)
        while sum(i["amount"] for i in items) * 1.18 <= 50000:
            items[0]["qty"] += 5
            items[0]["amount"] = items[0]["qty"] * items[0]["rate"]
    else:                                   # keep others under the threshold
        while sum(i["amount"] for i in items) * 1.18 > 50000:
            big = max(items, key=lambda i: i["amount"])
            big["qty"] = max(1, big["qty"] // 2)
            big["amount"] = big["qty"] * big["rate"]

    subtotal = sum(i["amount"] for i in items)
    rate = 0.09
    cgst = round(subtotal * rate, 2)
    sgst = round(subtotal * rate, 2)
    correct_total = round(subtotal + cgst + sgst, 2)

    inv_date = date(2026, 9, 1) + timedelta(days=random.randint(0, 55))
    due_date = inv_date + timedelta(days=random.choice([15, 30, 45]))
    inv_no = f"INV-{random.randint(1000, 9999)}"
    po_no = f"PO-2026-{100 + idx:03d}"

    doc = {
        "vendor": vendor, "invoice_no": inv_no, "invoice_date": inv_date, "due_date": due_date,
        "po_number": po_no, "items": items, "subtotal": subtotal,
        "tax_rows": [("CGST @ 9%", cgst), ("SGST @ 9%", sgst)], "printed_total": correct_total,
    }
    note = ""

    if error == "total_mismatch":
        off = random.choice([1000, 1500, 2000, 2500])
        doc["printed_total"] = correct_total + off
        note = f"Printed total is Rs.{money(off)} higher than subtotal + tax."
    elif error == "duplicate_tax_row":
        doc["tax_rows"] = [("CGST @ 9%", cgst), ("SGST @ 9%", sgst), ("SGST @ 9%", sgst)]
        doc["printed_total"] = round(correct_total + sgst, 2)
        note = f"SGST row printed twice, inflating total by Rs.{money(sgst)}."
    elif error == "missing_invoice_number":
        doc["invoice_no"] = None
        note = "Invoice number field is blank."
    elif error == "due_before_invoice_date":
        doc["due_date"] = inv_date - timedelta(days=random.choice([5, 10, 20]))
        note = "Due date is earlier than invoice date."
    elif error == "invalid_po_number":
        doc["po_number"] = f"PO-2026-{random.randint(900, 999)}"
        note = "PO number does not exist in purchase_orders.csv."
    elif error == "tax_rate_error":
        wrong = round(subtotal * 0.12, 2)
        doc["tax_rows"] = [("CGST @ 9%", wrong), ("SGST @ 9%", sgst)]
        doc["printed_total"] = round(subtotal + wrong + sgst, 2)
        note = "CGST amount is computed at 12% although the label says 9%."
    elif error == "unregistered_vendor":
        note = "Vendor is not in vendors.csv."
    elif error == "high_value_over_50000":
        note = "Invoice total exceeds the Rs.50,000 approval threshold."

    doc["correct_total"] = correct_total
    doc["note"] = note
    return doc


def draw_invoice(doc, path):
    c = canvas.Canvas(str(path), pagesize=A4)
    W, H = A4
    v = doc["vendor"]
    y = H - 60
    c.setFont("Helvetica-Bold", 18); c.drawString(50, y, v["name"].upper())
    c.setFont("Helvetica", 10)
    c.drawString(50, y - 18, f"{v['city']}, Maharashtra")
    c.drawString(50, y - 32, f"GSTIN: {v['gstin']}")
    c.setFont("Helvetica-Bold", 22); c.drawRightString(W - 50, y, "TAX INVOICE")
    c.setFont("Helvetica", 10)
    c.drawRightString(W - 50, y - 22, f"Invoice No: {doc['invoice_no'] or ''}")
    c.drawRightString(W - 50, y - 36, f"Invoice Date: {doc['invoice_date'].strftime('%d/%m/%Y')}")
    c.drawRightString(W - 50, y - 50, f"Due Date: {doc['due_date'].strftime('%d/%m/%Y')}")
    c.drawRightString(W - 50, y - 64, f"PO Number: {doc['po_number']}")

    y -= 100
    c.line(50, y, W - 50, y)
    c.setFont("Helvetica-Bold", 10); c.drawString(50, y - 18, "Bill To:")
    c.setFont("Helvetica", 10)
    c.drawString(50, y - 32, BUYER["name"])
    c.drawString(50, y - 46, BUYER["address"])
    c.drawString(50, y - 60, f"GSTIN: {BUYER['gstin']}")

    # line-item table
    y -= 95
    cols = [50, 80, 330, 390, 470]
    c.setFillGray(0.88); c.rect(50, y - 6, W - 100, 20, fill=1, stroke=0); c.setFillGray(0)
    c.setFont("Helvetica-Bold", 10)
    for x, h in zip(cols, ["#", "Description", "Qty", "Rate", "Amount"]):
        c.drawString(x + 2, y, h)
    c.setFont("Helvetica", 10)
    for n, it in enumerate(doc["items"], 1):
        y -= 22
        c.drawString(cols[0] + 2, y, str(n))
        c.drawString(cols[1] + 2, y, it["description"])
        c.drawString(cols[2] + 2, y, str(it["qty"]))
        c.drawString(cols[3] + 2, y, money(it["rate"]))
        c.drawRightString(W - 52, y, money(it["amount"]))
    y -= 12; c.line(50, y, W - 50, y)

    # totals block
    def row(label, val, bold=False):
        nonlocal y
        y -= 20
        c.setFont("Helvetica-Bold" if bold else "Helvetica", 11 if bold else 10)
        c.drawString(360, y, label); c.drawRightString(W - 52, y, f"Rs. {money(val)}")
    row("Subtotal", doc["subtotal"])
    for label, val in doc["tax_rows"]:
        row(label, val)
    y -= 6; c.line(350, y, W - 50, y)
    row("TOTAL", doc["printed_total"], bold=True)

    y -= 60
    c.setFont("Helvetica", 9)
    c.drawString(50, y, "Payment to: HDFC Bank, A/c details on request. Subject to Mumbai jurisdiction.")
    c.drawRightString(W - 50, y - 40, f"For {v['name']}")
    c.drawRightString(W - 50, y - 70, "Authorised Signatory")
    c.save()


def draw_other(kind, path):
    """Non-invoice documents so the classifier has something to reject."""
    c = canvas.Canvas(str(path), pagesize=A4)
    W, H = A4
    vid, name, gstin, city = random.choice(VENDORS)
    if kind == "delivery_challan":
        c.setFont("Helvetica-Bold", 18); c.drawString(50, H - 60, name.upper())
        c.setFont("Helvetica-Bold", 20); c.drawRightString(W - 50, H - 60, "DELIVERY CHALLAN")
        c.setFont("Helvetica", 10)
        c.drawString(50, H - 90, f"Challan No: DC-{random.randint(100, 999)}   Date: 14/10/2026")
        c.drawString(50, H - 110, f"Deliver To: {BUYER['name']}, {BUYER['address']}")
        y = H - 150
        c.setFont("Helvetica-Bold", 10); c.drawString(50, y, "Description"); c.drawString(400, y, "Qty")
        c.setFont("Helvetica", 10)
        for desc, _ in CATALOG[vid][:3]:
            y -= 20; c.drawString(50, y, desc); c.drawString(400, y, str(random.randint(1, 10)))
        c.drawString(50, y - 50, "Goods sent for delivery only. Not a tax invoice. Receiver's signature: __________")
    else:  # quotation letter
        c.setFont("Helvetica-Bold", 16); c.drawString(50, H - 60, name)
        c.setFont("Helvetica", 11)
        lines = [f"Date: 05/10/2026", "", f"To, The Purchase Manager, {BUYER['name']}", "",
                 "Subject: Quotation for annual supply", "", "Dear Sir/Madam,",
                 "With reference to your enquiry, we are pleased to quote our best rates",
                 "for the items listed in your requirement. Prices are valid for 30 days",
                 "and are exclusive of GST. Delivery within 7 working days of order.", "",
                 "We look forward to your valued order.", "", "Regards,", f"Sales Team, {name}"]
        y = H - 100
        for ln in lines:
            c.drawString(50, y, ln); y -= 18
    c.save()
    return {"vendor": name}


def rasterize(pdf_path, dpi):
    stem = pdf_path.with_suffix("")
    subprocess.run(["pdftoppm", "-r", str(dpi), "-png", "-singlefile", str(pdf_path), str(stem)], check=True)
    return Image.open(str(stem) + ".png").convert("L")


def make_scan(img):
    """Grey, slightly skewed, noisy flatbed scan."""
    img = img.rotate(random.uniform(-2.5, 2.5), expand=True, fillcolor=255, resample=Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32)
    arr = arr * 0.92 + 12 + np.random.normal(0, 10, arr.shape)          # dull + noise
    arr[np.random.random(arr.shape) < 0.002] = 0                        # specks
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6))
    return img


def make_photo(img):
    """Phone photo: perspective tilt, uneven lighting, warm tint, JPEG."""
    w, h = img.size
    d = int(w * 0.04)
    coeffs = _perspective_coeffs(
        [(0, 0), (w, 0), (w, h), (0, h)],
        [(random.randint(0, d), random.randint(0, d)), (w - random.randint(0, d), random.randint(0, d)),
         (w - random.randint(0, d), h - random.randint(0, d)), (random.randint(0, d), h - random.randint(0, d))])
    img = img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC, fillcolor=90)
    img = img.rotate(random.uniform(-4, 4), expand=False, fillcolor=90, resample=Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    shade = 0.75 + 0.25 * (xx / w) * (1 - 0.4 * yy / h)                  # lighting gradient
    arr = arr * shade + np.random.normal(0, 6, arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    rgb = np.stack([arr, (arr * 0.97).astype(np.uint8), (arr * 0.88).astype(np.uint8)], -1)
    return Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(0.8))


def _perspective_coeffs(src, dst):
    A, B = [], []
    for (x, y), (u, v) in zip(dst, src):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); B.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y]); B.append(v)
    return np.linalg.solve(np.array(A), np.array(B)).tolist()


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    OUT = Path(args.out)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if OUT.exists():
        shutil.rmtree(OUT)
    docs_dir = OUT / "documents"
    tmp = OUT / "_tmp"
    docs_dir.mkdir(parents=True); tmp.mkdir()

    order = SPECS[:]
    random.shuffle(order)
    truth = []
    po_rows = []

    for i, (error, fmt) in enumerate(order, 1):
        doc = build_invoice(i, error)
        if error != "invalid_po_number":
            po_rows.append((doc["po_number"], doc["vendor"]["id"] or "", f"{doc['correct_total']:.2f}"))
        stem = f"invoice_{i:03d}"
        pdf = tmp / f"{stem}.pdf"
        draw_invoice(doc, pdf)
        if fmt == "pdf":
            fname = f"{stem}.pdf"; shutil.copy(pdf, docs_dir / fname)
        elif fmt == "scan":
            fname = f"{stem}_scan.png"; make_scan(rasterize(pdf, 200)).save(docs_dir / fname, optimize=True)
        else:
            fname = f"{stem}_photo.jpg"; make_photo(rasterize(pdf, 170)).save(docs_dir / fname, quality=80)

        truth.append({
            "file_name": fname, "format": fmt, "doc_type": "invoice",
            "vendor_name": doc["vendor"]["name"], "vendor_gstin": doc["vendor"]["gstin"],
            "vendor_registered": doc["vendor"]["id"] is not None,
            "invoice_no": doc["invoice_no"], "invoice_date": doc["invoice_date"].isoformat(),
            "due_date": doc["due_date"].isoformat(), "po_number": doc["po_number"],
            "line_items": doc["items"],
            "subtotal": doc["subtotal"],
            "tax_rows": [{"label": l, "amount": a} for l, a in doc["tax_rows"]],
            "printed_total": doc["printed_total"], "correct_total": doc["correct_total"],
            "planted_error": error, "expected_status": EXPECTED_STATUS[error], "error_note": doc["note"],
        })

    for j, kind in enumerate(["delivery_challan", "quotation_letter"], 1):
        pdf = tmp / f"other_{j:03d}.pdf"
        info = draw_other(kind, pdf)
        fname = f"other_{j:03d}_scan.png" if j == 1 else f"other_{j:03d}.pdf"
        if j == 1:
            make_scan(rasterize(pdf, 200)).save(docs_dir / fname, optimize=True)
        else:
            shutil.copy(pdf, docs_dir / fname)
        truth.append({"file_name": fname, "format": "scan" if j == 1 else "pdf", "doc_type": kind,
                      "vendor_name": info["vendor"], "planted_error": "not_an_invoice",
                      "expected_status": "rejected", "error_note": "Not an invoice; classifier should reject."})

    shutil.rmtree(tmp)

    # reference files
    with open(OUT / "vendors.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["vendor_id", "vendor_name", "gstin", "city"])
        w.writerows(VENDORS)
    with open(OUT / "purchase_orders.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["po_number", "vendor_id", "approved_amount"])
        w.writerows(sorted(po_rows))
    with open(OUT / "business_rules.json", "w") as f:
        json.dump({"high_value_threshold": 50000, "total_tolerance": 1.0,
                   "gst_rates": {"CGST": 0.09, "SGST": 0.09},
                   "required_fields": ["vendor_name", "vendor_gstin", "invoice_no", "invoice_date",
                                       "due_date", "po_number", "line_items", "printed_total"]}, f, indent=2)

    # ground truth
    with open(OUT / "ground_truth.json", "w") as f:
        json.dump(truth, f, indent=2)
    with open(OUT / "ground_truth.csv", "w", newline="") as f:
        fields = ["file_name", "format", "doc_type", "vendor_name", "invoice_no", "invoice_date",
                  "due_date", "po_number", "subtotal", "printed_total", "correct_total",
                  "planted_error", "expected_status", "error_note"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader()
        w.writerows(truth)

    print(f"Wrote {len(truth)} documents to {OUT}")


if __name__ == "__main__":
    main()
