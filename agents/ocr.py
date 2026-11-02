"""Agent 2 - Layout & OCR Agent.

Runs Tesseract on every page and returns text lines with their bounding boxes
and confidence, so every extracted field can be traced back to where it was
found on the page.
"""
import os

import pytesseract
from PIL import Image, ImageFilter

# On Windows, point this at tesseract.exe if it is not on PATH, e.g.
#   set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
if os.environ.get("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]


class OCRAgent:
    name = "OCRAgent"

    def __init__(self, min_confidence=80.0, line_retry_below=60.0):
        self.min_confidence = min_confidence
        self.line_retry_below = line_retry_below

    @staticmethod
    def reread(page, bbox, scale=2):
        """OCR one line again from a padded, enlarged crop (single-line mode)."""
        x0, y0, x1, y1 = bbox
        if x1 - x0 < 20 or y1 - y0 < 8:                 # too small to be worth re-reading
            return None
        pad = max(8, (y1 - y0) // 2)
        crop = page.crop((max(0, x0 - pad), max(0, y0 - pad),
                          min(page.width, x1 + pad), min(page.height, y1 + pad)))
        crop = crop.resize((crop.width * scale, crop.height * scale), Image.LANCZOS)
        crop = crop.filter(ImageFilter.MedianFilter(3))  # removes scan speckle
        try:
            data = pytesseract.image_to_data(crop, config="--psm 7", output_type=pytesseract.Output.DICT)
        except pytesseract.TesseractError:
            return None
        words = [(w, float(c)) for w, c in zip(data["text"], data["conf"]) if w.strip() and float(c) >= 0]
        if not words:
            return None
        return {"text": " ".join(w for w, _ in words), "confidence": round(min(c for _, c in words), 1)}

    def run(self, record):
        lines, confs = [], []
        for page_no, page in enumerate(record["pages"], start=1):
            data = pytesseract.image_to_data(page, config="--psm 4",
                                             output_type=pytesseract.Output.DICT)
            grouped = {}
            for i, word in enumerate(data["text"]):
                if not word.strip():
                    continue
                key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
                grouped.setdefault(key, []).append(i)
                conf = float(data["conf"][i])
                if conf >= 0:
                    confs.append(conf)

            for key in sorted(grouped, key=lambda k: min(data["top"][i] for i in grouped[k])):
                idx = sorted(grouped[key], key=lambda i: data["left"][i])
                x0 = min(data["left"][i] for i in idx)
                y0 = min(data["top"][i] for i in idx)
                x1 = max(data["left"][i] + data["width"][i] for i in idx)
                y1 = max(data["top"][i] + data["height"][i] for i in idx)
                word_confs = [float(data["conf"][i]) for i in idx if float(data["conf"][i]) >= 0]
                lines.append({
                    "text": " ".join(data["text"][i] for i in idx),
                    "page": page_no,
                    "bbox": [x0, y0, x1, y1],
                    "confidence": round(min(word_confs), 1) if word_confs else 0.0,
                })

        # Targeted retry: re-read each weak line from an enlarged, denoised crop.
        # The second reading is kept beside the first; the Extraction Agent
        # decides which one is consistent with the rest of the document.
        rereads = 0
        for line in lines:
            if line["confidence"] < self.line_retry_below:
                alt = self.reread(record["pages"][line["page"] - 1], line["bbox"])
                if alt and alt["text"] != line["text"]:
                    line["alt_text"] = alt["text"]
                    line["alt_confidence"] = alt["confidence"]
                    rereads += 1

        record["lines"] = lines
        record["lines_reread"] = rereads
        record["raw_text"] = "\n".join(l["text"] for l in lines)
        record["ocr_confidence"] = round(sum(confs) / len(confs), 1) if confs else 0.0
        record["ocr_low_confidence"] = record["ocr_confidence"] < self.min_confidence
        return record
