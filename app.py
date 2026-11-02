"""Streamlit demo page for the Document Intelligence & Extraction Agent.

Run:  streamlit run app.py
"""
import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import ImageDraw

from orchestrator import BASE, Orchestrator

st.set_page_config(page_title="Document Intelligence Agent", layout="wide")

STATUS_STYLE = {
    "auto_approved": ("success", "Auto-approved"),
    "needs_review": ("warning", "Needs review"),
    "rejected": ("error", "Rejected"),
}
FIELD_LABELS = {
    "vendor_name": "Vendor name", "vendor_gstin": "Vendor GSTIN", "invoice_no": "Invoice no.",
    "invoice_date": "Invoice date", "due_date": "Due date", "po_number": "PO number",
    "subtotal": "Subtotal (Rs.)", "printed_total": "Printed total (Rs.)",
}


@st.cache_resource
def get_orchestrator():
    return Orchestrator(reference_dir=BASE / "dataset", out_dir=BASE / "output_app")


def highlight(record):
    """Draw every extracted field's source box: green = OK, red = flagged."""
    page = record["pages"][0].convert("RGB")
    draw = ImageDraw.Draw(page)
    flagged = {f["field"] for f in record.get("flags", []) if f["severity"] != "Low" and f["field"]}
    for field, src in record.get("sources", {}).items():
        for s in (src if isinstance(src, list) else [src]):
            colour = (220, 50, 47) if field in flagged else (40, 160, 70)
            draw.rectangle(s["bbox"], outline=colour, width=4)
    return page


def show_result(record):
    kind, label = STATUS_STYLE[record["status"]]
    getattr(st, kind)(f"**{label}**" + (f" (priority: {record['priority']})" if record.get("priority") else "")
                      + f" — {record['reason']}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Document type", record["doc_type"])
    c2.metric("OCR confidence", f"{record.get('ocr_confidence', 0)}%")
    c3.metric("Deskew applied", "yes" if record.get("deskew_applied") else "no")
    c4.metric("Processing time", f"{record['processing_seconds']} s")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Source document")
        st.caption("Green boxes: where each field was read. Red: fields with a flag.")
        if record.get("pages"):
            st.image(highlight(record) if record.get("sources") else record["pages"][0],
                     width="stretch")

    with right:
        if record["doc_type"] == "invoice":
            st.subheader("Flags")
            flags = [f for f in record.get("flags", [])]
            if flags:
                st.dataframe(pd.DataFrame([{"Rule": f["rule"], "Severity": f["severity"],
                                            "Message": f["message"]} for f in flags]),
                             hide_index=True, width="stretch")
            else:
                st.write("No flags raised — all checks passed.")

            st.subheader("Extracted fields")
            f = record["fields"]
            st.dataframe(pd.DataFrame([{"Field": FIELD_LABELS[k], "Value": str(f.get(k)),
                                        "Read from line": (record["sources"].get(k) or {}).get("line", "")}
                                       for k in FIELD_LABELS]),
                         hide_index=True, width="stretch")

            st.subheader("Line items")
            st.dataframe(pd.DataFrame(f.get("line_items", [])), hide_index=True, width="stretch")
            st.subheader("Tax rows")
            st.dataframe(pd.DataFrame(f.get("tax_rows", [])), hide_index=True, width="stretch")

        st.subheader("Structured output (JSON)")
        export = {k: v for k, v in record.items() if k not in ("pages", "lines", "raw_text", "file_path")}
        st.download_button("Download JSON record", json.dumps(export, indent=2, default=str),
                           file_name=f"{Path(record['file_name']).stem}.json", mime="application/json")
        with st.expander("Audit trail"):
            st.json(record.get("audit_trail", []))


st.title("Document Intelligence & Extraction Agent")
st.write("Upload a vendor invoice (PDF, scanned image or phone photo). Six agents read it, "
         "extract its fields, check them against business rules and decide whether it can be "
         "auto-approved or needs human review.")

tab_single, tab_batch = st.tabs(["Process one document", "Batch dashboard"])

with tab_single:
    upload = st.file_uploader("Choose a document", type=["pdf", "png", "jpg", "jpeg"])
    samples = sorted(p.name for p in (BASE / "dataset" / "documents").iterdir())
    sample = st.selectbox("...or pick a sample from the dataset", ["—"] + samples)

    path = None
    if upload:
        tmp = Path(tempfile.mkdtemp()) / upload.name
        tmp.write_bytes(upload.getvalue())
        path = tmp
    elif sample != "—":
        path = BASE / "dataset" / "documents" / sample

    if path:
        with st.spinner("Running the agents..."):
            record = get_orchestrator().process(path, keep_pages=True)
        show_result(record)

with tab_batch:
    st.write("Process every document in `dataset/documents` and summarise the results.")
    if st.button("Run batch"):
        orch = get_orchestrator()
        files = sorted(p for p in (BASE / "dataset" / "documents").iterdir()
                       if p.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg"})
        progress = st.progress(0.0)
        records = []
        for i, p in enumerate(files, 1):
            records.append(orch.process(p))
            progress.progress(i / len(files), text=f"{i}/{len(files)}: {p.name}")
        orch.output.write_batch_csv(records)
        df = pd.DataFrame([{"File": r["file_name"], "Status": r["status"], "Priority": r.get("priority"),
                            "Seconds": r["processing_seconds"], "Reason": r["reason"]} for r in records])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Auto-approved", int((df.Status == "auto_approved").sum()))
        c2.metric("Needs review", int((df.Status == "needs_review").sum()))
        c3.metric("Rejected", int((df.Status == "rejected").sum()))
        c4.metric("Avg time / document", f"{df.Seconds.mean():.1f} s")
        st.bar_chart(df.Status.value_counts())
        st.dataframe(df, hide_index=True, width="stretch")
