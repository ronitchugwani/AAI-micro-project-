"""Agent 4 - Field Extraction Agent.

Pulls invoice fields out of the OCR lines with label-based regular expressions.
Every field records the line it came from (page + bounding box) so a reviewer
can find it on the original document. Fields that cannot be found are left
empty and listed in `missing_fields` - the agent never guesses.
"""
import re
from datetime import datetime

MONEY = r"\d{1,3}(?:[,.;]\d{3})*[.:]\d{2}|\d+[.:]\d{2}"
GSTIN_SHAPE = re.compile(r"^\d{2}[A-Z]{5}\d{4}[A-Z][0-9A-Z]Z[0-9A-Z]$")


def to_amount(text):
    """'Rs. 1,954.80' -> 1954.8 ; tolerates OCR reading '.' as ':'."""
    text = text.replace(" ", "")
    m = re.findall(MONEY, text)
    if not m:
        return None
    value = m[-1].replace(":", ".")
    whole, dec = value[:-3], value[-2:]
    return float(re.sub(r"[^\d]", "", whole) + "." + dec)


def to_iso_date(text):
    text = re.sub(r"\s", "", text)
    m = re.search(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})", text)
    if not m:
        return None
    try:
        return datetime(int(m.group(3)), int(m.group(2)), int(m.group(1))).date().isoformat()
    except ValueError:
        return None


# GSTIN layout: 2 digits (state) + 5 letters + 4 digits + 1 letter (PAN)
#               + 1 entity code + 'Z' + 1 check character
GSTIN_KINDS = "DDLLLLLDDDDLAZA"
TO_DIGIT = str.maketrans("OQDIlSBGZ", "001115862")
TO_LETTER = str.maketrans("01582", "OISBZ")


def repair_gstin(g):
    """Apply position-aware OCR fixes to a 15-character candidate."""
    out = []
    for ch, kind in zip(g, GSTIN_KINDS):
        if kind == "D":
            ch = ch.translate(TO_DIGIT)
        elif kind == "L":
            ch = ch.translate(TO_LETTER)
        elif kind == "Z" and ch in "27":
            ch = "Z"
        out.append(ch)
    return "".join(out)


def gstin_candidates(raw):
    """All readings of an OCR'd GSTIN that fit the GSTIN format. A 16-character
    reading usually has one inserted character, so each single deletion is
    tried. The Validation Agent picks between candidates using the vendor master."""
    tokens = []
    for tok in raw.split():
        if re.fullmatch(r"[A-Za-z][a-z]+\W*", tok):
            break                                  # reached the next label
        tokens.append(tok)
    g = re.sub(r"[^0-9A-Za-z]", "", "".join(tokens)).upper()
    pool = [g] if len(g) <= 15 else [g[:i] + g[i + 1:] for i in range(len(g))] if len(g) == 16 else [g[:15]]
    found = []
    for c in pool:
        fixed = repair_gstin(c)
        if len(fixed) == 15 and GSTIN_SHAPE.match(fixed) and fixed not in found:
            found.append(fixed)
    return found, repair_gstin(g[:15])


class ExtractionAgent:
    name = "ExtractionAgent"

    def run(self, record):
        lines = record["lines"]
        fields, sources, notes = {}, {}, []

        def find(pattern, flags=re.IGNORECASE, start=0, stop=None):
            """First line matching the pattern; falls back to a line's re-read text."""
            for i, line in enumerate(lines[start:stop], start=start):
                m = re.search(pattern, line["text"], flags)
                if m:
                    return m, i
            for i, line in enumerate(lines[start:stop], start=start):
                m = re.search(pattern, line.get("alt_text", ""), flags)
                if m:
                    line["used_reading"] = "re-read"
                    return m, i
            return None, None

        def source(i):
            l = lines[i]
            return {"page": l["page"], "bbox": l["bbox"], "line": l["text"], "ocr_conf": l["confidence"]}

        bill_to = find(r"bill\s*to")[1] or len(lines)

        # --- vendor name: prefer the "For <Vendor>" line above the signature
        m, i = find(r"^\s*For\s+(.{3,})$")
        if m:
            fields["vendor_name"] = m.group(1).strip(" .,")
        else:
            first = re.sub(r"tax\s+invoice", "", lines[0]["text"], flags=re.I).strip() if lines else ""
            fields["vendor_name"], i = (first.title() or None), 0
        if fields["vendor_name"]:
            sources["vendor_name"] = source(i)

        # --- vendor GSTIN: the first GSTIN above "Bill To"
        m, i = find(r"GSTIN\s*[:;]?\s*(.+)$", stop=bill_to)
        if m:
            candidates, fallback = gstin_candidates(m.group(1))
            fields["vendor_gstin"] = candidates[0] if candidates else fallback
            record["gstin_candidates"] = candidates
            sources["vendor_gstin"] = source(i)
            if not candidates:
                notes.append(f"vendor GSTIN '{fallback}' does not fit the GSTIN format; OCR may have misread it")

        # --- header fields
        for key, pattern, convert in [
            ("invoice_no", r"Invoice\s*[:;.]?\s*No\.?\s*[:;]?\s*([A-Z]{2,5}\s*-\s*\d+)", lambda s: re.sub(r"\s", "", s)),
            ("invoice_date", r"Invoice\s*Date\s*[:;]?\s*([\d/.\- ]{8,12})", to_iso_date),
            ("due_date", r"Due\s*Date\s*[:;]?\s*([\d/.\- ]{8,12})", to_iso_date),
            ("po_number", r"PO[.,]?\s*Number\s*[:;]?\s*(PO[\s\-\d]+)", lambda s: re.sub(r"\s", "", s)),
        ]:
            m, i = find(pattern)
            value = convert(m.group(1)) if m else None
            fields[key] = value
            if value:
                sources[key] = source(i)

        # --- line items: rows between the table header and "Subtotal"
        header = find(r"Description")[1]
        sub_m, sub_i = find(r"Sub\s*total")
        items = []
        if header is not None and sub_i is not None:
            for i in range(header + 1, sub_i):
                item = self.parse_item(lines[i]["text"])
                if "alt_text" in lines[i]:
                    # two readings of a weak line: keep the one whose qty x rate = amount
                    alt = self.parse_item(lines[i]["alt_text"])
                    if alt and (item is None or (item["qty_derived"] and not alt["qty_derived"])
                                or (self.consistent(alt) and not self.consistent(item))):
                        item = alt
                        notes.append(f"line '{alt['description']}' was re-read from an enlarged crop "
                                     f"(first reading: '{lines[i]['text']}')")
                if item:
                    items.append(item)
                    if item.pop("qty_derived"):
                        notes.append(f"quantity for '{item['description']}' was unreadable or "
                                     f"misread; derived from amount / rate = {item['qty']}")
                    sources.setdefault("line_items", []).append(source(i))
        fields["line_items"] = items

        # --- totals block
        fields["subtotal"] = to_amount(sub_m.group(0) + lines[sub_i]["text"][sub_m.end():]) if sub_m else None
        if sub_m:
            sources["subtotal"] = source(sub_i)

        taxes, unread_taxes = [], []
        tax_re = re.compile(r"\b([CS]GST|IGST)\W{0,3}@?\W{0,3}(\d{1,2})\s*%", re.I)
        for i, line in enumerate(lines):
            found = None
            for text in (line["text"], line.get("alt_text", "")):   # first reading, then re-read
                m = tax_re.search(text)
                amount = to_amount(text[m.end():]) if m else None
                if m and amount is not None:
                    found = (m, amount)
                    break
            if found:
                m, amount = found
                taxes.append({"label": f"{m.group(1).upper()} @ {m.group(2)}%",
                              "type": m.group(1).upper(), "rate": int(m.group(2)) / 100,
                              "amount": amount})
                sources.setdefault("tax_rows", []).append(source(i))
            elif tax_re.search(line["text"]):
                unread_taxes.append(line["text"].strip())
                notes.append(f"tax row '{line['text'].strip()}' has no readable amount")
        fields["tax_rows"] = taxes
        record["unread_tax_rows"] = unread_taxes

        m, i = find(r"^\s*TOTAL\b", flags=0)
        fields["printed_total"] = to_amount(lines[i]["text"]) if m else None
        if m:
            sources["printed_total"] = source(i)

        record["fields"] = fields
        record["sources"] = sources
        record["extraction_notes"] = notes
        record["missing_fields"] = [k for k, v in fields.items() if v in (None, [], "")]
        return record

    @staticmethod
    def consistent(item):
        return item["qty"] is not None and abs(item["qty"] * item["rate"] - item["amount"]) <= 0.5

    @staticmethod
    def parse_item(text):
        """'2 Bubble Wrap Roll 20 650.00 13,000.00' -> description/qty/rate/amount."""
        moneys = list(re.finditer(MONEY, text))
        if len(moneys) < 2:
            return None
        rate = to_amount(moneys[-2].group(0))
        amount = to_amount(moneys[-1].group(0))
        before = text[:moneys[-2].start()]
        before = re.sub(r"^[^A-Za-z]*?\d{1,2}\s+(?=[A-Za-z])", "", before)  # drop row number
        qty_m = re.search(r"(\d{1,4})\s*[^\w(]*\s*$", before)
        qty, derived = None, False
        if qty_m and not before.rstrip().endswith(")"):
            qty = int(qty_m.group(1))
            description = before[:qty_m.start()]
        else:
            description = before
        if (qty is None or abs(qty * rate - amount) > 0.5) and rate:
            guess = amount / rate
            if abs(guess - round(guess)) < 0.01:
                qty, derived = int(round(guess)), True
        description = re.sub(r"[^\w()\-/ ]", "", description).strip(" -")
        return {"description": description, "qty": qty, "rate": rate,
                "amount": amount, "qty_derived": derived}
