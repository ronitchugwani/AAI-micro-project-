"""Agent 1 - Document Ingestion Agent.

Accepts a PDF, PNG or JPG file and turns it into a list of clean, straightened
greyscale page images that the OCR Agent can read.
"""
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from pdf2image import convert_from_path

SUPPORTED = {".pdf", ".png", ".jpg", ".jpeg"}


class IngestionAgent:
    name = "IngestionAgent"

    def __init__(self, dpi=300):
        self.dpi = dpi

    def run(self, record, dpi=None):
        path = Path(record["file_path"])
        ext = path.suffix.lower()
        if ext not in SUPPORTED:
            raise ValueError(f"Unsupported file type: {ext}")

        dpi = dpi or self.dpi
        if ext == ".pdf":
            # POPPLER_PATH: folder containing pdftoppm (needed on Windows only)
            raw_pages = convert_from_path(str(path), dpi=dpi,
                                          poppler_path=os.environ.get("POPPLER_PATH"))
            record["format"] = "pdf"
        else:
            img = Image.open(path)
            if dpi > self.dpi:                       # retry: upscale the image
                scale = dpi / self.dpi
                img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
            raw_pages = [img]
            record["format"] = "image"

        pages, angles = [], []
        for p in raw_pages:
            grey = ImageOps.autocontrast(ImageOps.grayscale(p), cutoff=1)
            angle = self.estimate_skew(grey)
            if abs(angle) >= 0.2:
                grey = grey.rotate(angle, expand=True, fillcolor=255, resample=Image.BICUBIC)
            pages.append(grey)
            angles.append(angle)

        record["pages"] = pages
        record["page_count"] = len(pages)
        record["dpi_used"] = dpi
        record["deskew_angles"] = angles
        record["deskew_applied"] = any(abs(a) >= 0.2 for a in angles)
        return record

    @staticmethod
    def estimate_skew(img, max_angle=6.0, step=0.25):
        """Projection-profile deskew: the rotation that makes text rows line up
        gives the sharpest (highest-variance) horizontal ink profile."""
        small = img.copy()
        small.thumbnail((900, 900))
        ink = (np.asarray(small) < 128).astype(np.uint8) * 255
        ink_img = Image.fromarray(ink)
        best_angle, best_score = 0.0, -1.0
        for angle in np.arange(-max_angle, max_angle + step, step):
            rotated = np.asarray(ink_img.rotate(angle, resample=Image.NEAREST))
            score = rotated.sum(axis=1).astype(np.float64).var()
            if score > best_score:
                best_angle, best_score = float(angle), score
        return round(best_angle, 2)
