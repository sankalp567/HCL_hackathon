"""One-time Docling pass: each PDF -> lossless DoclingDocument JSON + Markdown.

Usage (GPU server): python extract/docling_run.py <artifacts_dir> <out_dir> <pdf> [<pdf> ...]
"""
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption


def main(artifacts: str, out_dir: str, pdfs: list[str]) -> None:
    opts = PdfPipelineOptions()
    opts.do_ocr = False  # all three reports have a text layer
    opts.artifacts_path = artifacts
    opts.do_table_structure = True
    opts.table_structure_options.mode = TableFormerMode.ACCURATE
    opts.do_picture_classification = True
    opts.accelerator_options = AcceleratorOptions(num_threads=8, device=AcceleratorDevice.CUDA)
    conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for pdf in pdfs:
        t0 = time.time()
        doc = conv.convert(pdf).document
        stem = Path(pdf).stem
        (out / f"{stem}.docling.json").write_text(json.dumps(doc.export_to_dict()))
        (out / f"{stem}.md").write_text(doc.export_to_markdown())
        print(f"{stem}: {len(doc.pages)} pages, {len(doc.tables)} tables, {len(doc.pictures)} pictures, "
              f"{len(doc.texts)} text items in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
