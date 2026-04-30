"""
Spatial / Patch-Vis PDF utilities — quick usage guide
----------------------------------------------------

This file contains two primary operations you will use in sequence:

1) generate_plot_per_run(...)
   - Run this **per run / per dataset folder** to create a PDF of images inside that
     dataset directory.
   - Example: for a Xenium run folder `XeniumPR1` run:
         generate_plot_per_run(base_root="/project/gutdecoder/kxu/hest/xenium_data",
                               datasets=["XeniumPR1"],
                               match="spatial_plots")         # exact filename spatial_plots.png
     or
         generate_plot_per_run(..., match="patch_vis")        # matches *patch_vis*.png
   - Common options:
       - match="spatial_plots"    → looks for exactly `spatial_plots.png` and saves `spatial_plots.pdf`
       - match="patch_vis"        → looks for `*patch_vis*.png` and saves `patch_vis.pdf`
       - match="*patch_vis*.png"  → arbitrary glob pattern supported
       - overwrite=True           → regenerate the PDF even if it already exists
       - metadata_path=...        → path to xenium_directory.xlsx to attach Patient/Slide/Type/Location

   - Output: a PDF saved inside the run folder (e.g., `.../XeniumPR1/spatial_plots.pdf`).

2) combine_pdfs(...)
   - After you have generated PDFs for multiple runs, use this to combine them into one
     consolidated PDF (with a title page per run).
   - Example:
         combine_pdfs(root_dir="/project/gutdecoder/kxu/hest/xenium_data",
                      folder_names=["XeniumPR1","XeniumPR2","XeniumR1"],
                      output_filename="/project/gutdecoder/kxu/hest/all_spatial_plots.pdf",
                      pdf_name="spatial_plots.pdf")
   - Notes:
       - `pdf_name` should match the filenames created by generate_plot_per_run (e.g. "patch_vis.pdf").
       - The combiner adds a simple title page for each run folder and rescales pages for consistent width.
       - If no run PDFs are found, no output will be written.

Recommended workflow
--------------------
1. For each run in your dataset list, call generate_plot_per_run(...) with the desired `match`
   (spatial_plots or patch_vis). This creates a per-run PDF inside each run folder.
2. Once all per-run PDFs exist, call combine_pdfs(...) with the list of run folder names you want
   combined into one master PDF.

Implementation details / caveats
-------------------------------
- Sample ID parsing: The script extracts canonical run prefixes like `XeniumPR1` or `XeniumR1`
  from dataset names such as `XeniumPR1_cell_l3` or `XeniumPR1_25_um_0.125_um_px`.
- File matching:
    - `match="spatial_plots"` — exact file name `spatial_plots.png` (recommended if you want exact match).
    - `match="patch_vis"` — a flexible pattern `*patch_vis*.png`.
    - You can also pass any glob string (e.g., "*patch_vis_v2*.png").
- The combiner writes output only if at least one run PDF was appended; missing runs will be printed.

"""

from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import math

import tempfile
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import portrait
import fnmatch
import os
import io
import pandas as pd
import textwrap
import re


def make_title_page_bytes(title: str, width_pt: float, height_pt: float, font_size: int = 36) -> bytes:
    """Create a one-page PDF (bytes) with `title` centered using ReportLab."""
    buf_path = Path(tempfile.gettempdir()) / f"title_{abs(hash(title)) & 0xffffffff}.pdf"
    c = canvas.Canvas(str(buf_path), pagesize=(width_pt, height_pt))
    x = width_pt / 2.0
    y = height_pt / 2.0
    c.setFont("Helvetica-Bold", font_size)
    # wrap/truncate long titles
    max_chars = max(10, int(width_pt // (font_size * 0.55)))
    if len(title) > max_chars:
        title = title[:max_chars-3] + "..."
    c.drawCentredString(x, y, title)
    c.showPage()
    c.save()
    return buf_path.read_bytes()


def resolve_folders(root: Path, names: list) -> list:
    root = Path(root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Root folder does not exist or is not a directory: {root}")
    elif names:
        folders = []
        for n in names:
            p = root / n
            if p.exists() and p.is_dir():
                folders.append(p)
            else:
                print(f"Warning: folder not found or not a directory, skipping: {p}")
    else:
        raise ValueError("You must specify folders, or use ALL or PATTERN.")
    return folders

TARGET_WIDTH_PT = 400.0  # adjust desired uniform width (pts)

def combine_folder_pdf(folder: Path, out_writer: PdfWriter, pdf_name = "spatial_plots.pdf",  title_font_size: int = 36) -> bool:
    pdf_path = folder / pdf_name
    if not pdf_path.exists():
        return False

    reader = PdfReader(str(pdf_path))
    if len(reader.pages) == 0:
        return False

    # title page (same width)
    title_h = 120.0
    title_bytes = make_title_page_bytes(folder.name, TARGET_WIDTH_PT, title_h, font_size=title_font_size)
    title_reader = PdfReader(io.BytesIO(title_bytes))
    out_writer.add_page(title_reader.pages[0])

    for page in reader.pages:
        try:
            ow = float(page.mediabox.width)
            oh = float(page.mediabox.height)
        except Exception:
            ow, oh = TARGET_WIDTH_PT, TARGET_WIDTH_PT * 0.75

        scale = TARGET_WIDTH_PT / ow

        # try to scale the page in-place
        try:
            page.scale_by(scale)
        except Exception:
            # fallback: leave unscaled (rare)
            pass

        new_w = float(page.mediabox.width)
        new_h = float(page.mediabox.height)

        # create a blank page of the scaled page size (width will equal TARGET_WIDTH_PT)
        blank = out_writer.add_blank_page(width=new_w, height=new_h)

        # merge the scaled page onto the blank at (0,0)
        try:
            blank.merge_translated_page(page, 0, 0)
        except Exception:
            try:
                blank.mergeScaledTranslatedPage(page, 1.0, 0, 0)
            except Exception:
                # last resort: append page directly
                out_writer.add_page(page)

    return True



def generate_plot_per_run(
    base_root: Path,
    datasets: list,
    cols: int = 4,
    max_show: int | None = None,
    thumb_size: tuple = (800, 800),
    metadata_path: Path | None = Path("/project/gutdecoder/kxu/hest/metadata/xenium_directory.xlsx"),
    match: str = "spatial_plots",    # "spatial_plots", "patch_vis", or any glob pattern
    out_filename: str | None = None, # if None, will be inferred from `match`
    overwrite: bool = False,
    verbose: bool = True,
):
    """
    Generate a grid PDF of images for each dataset, with metadata above each thumbnail.

    Args:
      base_root: Path root containing dataset folders.
      datasets: list of dataset folder names to search.
      cols: number of columns per page grid.
      max_show: limit number of images per dataset (None = all).
      thumb_size: thumbnail size for each image.
      metadata_path: path to xenium_directory.xlsx (or None to skip metadata).
      match: "spatial_plots" (matches EXACT 'spatial_plots.png'), "patch_vis" (matches '*patch_vis*.png'),
             or any glob pattern (e.g. '*patch_vis*.png' or 'spatial_plot*.png').
      out_filename: output pdf filename to save in each dataset folder. If None, inferred from `match`.
      overwrite: whether to overwrite existing output files.
      verbose: whether to print progress.
    """

    base_root = Path(base_root)

    # infer out_filename if not provided
    if out_filename is None:
        if match == "spatial_plots":
            out_filename = "spatial_plots.pdf"
        elif match == "patch_vis":
            out_filename = "patch_vis.pdf"
        else:
            # sanitize pattern to a filename
            out_filename = f"{re.sub(r'[^0-9A-Za-z]+', '_', match).strip('_')}.pdf"

    # load metadata if available
    if metadata_path is None or not Path(metadata_path).exists():
        if verbose:
            print("⚠️ xenium_directory.xlsx not found. Metadata columns will be empty.")
        meta_df = pd.DataFrame()
    else:
        meta_df = pd.read_excel(metadata_path)
        if "Sample_ID" not in meta_df.columns:
            raise ValueError(f"metadata sheet at {metadata_path} doesn't contain 'Sample_ID' column.")
        meta_df = meta_df.set_index("Sample_ID", drop=False)

    # regex helpers
    slide_re = re.compile(r"slide[_\- ]?0*(\d+)", re.IGNORECASE)
    roi_re = re.compile(r"roi[_\- ]?0*(\d+)", re.IGNORECASE)
    base_ds_re = re.compile(r"(XeniumPR\d+|XeniumR\d+)", re.IGNORECASE)

    def extract_sample_id_from_path(ds_name: str, full_path: Path, ds_dir: Path):
        # find canonical base anywhere in ds_name or ds_dir
        mbase = base_ds_re.search(ds_name) or base_ds_re.search(ds_dir.name)
        if mbase:
            base_name = mbase.group(1)
        else:
            base_name = ds_name.split("_")[0]

        try:
            parts = [p.lower() for p in full_path.relative_to(ds_dir).parts]
        except Exception:
            parts = [p.lower() for p in full_path.parts]

        slide_num = None
        roi_num = None
        for p in parts:
            m = slide_re.search(p)
            if m:
                slide_num = m.group(1)
            m2 = roi_re.search(p)
            if m2:
                roi_num = m2.group(1)

        # fallbacks
        if slide_num is None:
            for p in parts:
                m = re.search(r"^s?(\d{1,3})$", p)
                if m:
                    slide_num = m.group(1)
                    break
        if roi_num is None:
            for p in parts:
                m = re.search(r"^roi?0*(\d+)$", p)
                if m:
                    roi_num = m.group(1)
                    break

        slide_str = f"S{slide_num}" if slide_num else "S?"
        roi_str = f"ROI{roi_num}" if roi_num else "ROI?"
        return f"{base_name}{slide_str}{roi_str}"

    # choose rglob pattern based on match
    def _find_files(ds_dir: Path):
        if match == "spatial_plots":
            return sorted(p for p in ds_dir.rglob("spatial_plots.png") if p.is_file())
        elif match == "patch_vis":
            return sorted(p for p in ds_dir.rglob("*patch_vis*.png") if p.is_file())
        else:
            # treat match as a glob pattern
            return sorted(p for p in ds_dir.rglob(match) if p.is_file())

    # main loop
    for ds in datasets:
        ds_dir = base_root / ds
        if not ds_dir.exists():
            if verbose:
                print(f"⚠️ {ds_dir} not found, skipping.")
            continue

        out_pdf = ds_dir / out_filename
        if out_pdf.exists() and not overwrite:
            if verbose:
                print(f"⏭️  {out_pdf} already exists (overwrite=False), skipping {ds}.")
            continue
        if out_pdf.exists() and overwrite and verbose:
            print(f"♻️  Overwriting existing {out_pdf}")

        found = _find_files(ds_dir)
        if not found:
            if verbose:
                print(f"⚠️ No matching images found in {ds_dir} for match='{match}'")
            continue

        if max_show:
            found = found[:max_show]

        if verbose:
            print(f"🧩 {ds}: visualising {len(found)} plots")

        rows = math.ceil(len(found) / cols)
        if rows == 0:
            if verbose:
                print(f"⚠️ No images found for {ds}.")
            continue

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        # normalize axes list
        if isinstance(axes, plt.Axes):
            axes = [axes]
        else:
            axes = axes.flatten()

        for ax in axes:
            ax.axis("off")

        for i, (ax, path) in enumerate(zip(axes, found)):
            try:
                img = Image.open(path).convert("RGB")
                img.thumbnail(thumb_size)
                ax.imshow(img)
                ax.axis("off")

                sample_id = extract_sample_id_from_path(ds, path, ds_dir)

                # metadata lookup
                meta_row = None
                if not meta_df.empty:
                    if sample_id in meta_df.index:
                        meta_row = meta_df.loc[sample_id]
                    else:
                        short_id = re.sub(rf'^{re.escape(ds)}', "", sample_id, flags=re.IGNORECASE)
                        if short_id in meta_df.index:
                            meta_row = meta_df.loc[short_id]
                        else:
                            candidates = meta_df[meta_df.index.str.contains(re.escape(sample_id), case=False, na=False)]
                            if not candidates.empty:
                                meta_row = candidates.iloc[0]

                if meta_row is None:
                    rel = path.relative_to(ds_dir)
                    subtitle_parts = [f"Sample: {sample_id}", f"Path: {rel}"]
                else:
                    pid = meta_row.get("Patient ID", meta_row.get("PatientID", meta_row.get("Patient_Id", meta_row.get("Patient_id", ""))))
                    slide_col = meta_row.get("Slide_ID", meta_row.get("Slide ID", ""))
                    sample_type = meta_row.get("Sample_type", meta_row.get("Sample Type", meta_row.get("Sample_type", "")))
                    location = meta_row.get("Location", meta_row.get("LOCATION", ""))
                    pid = str(pid) if pd.notna(pid) else ""
                    # Slide ID: int if numeric, else keep string
                    if pd.notna(slide_col) and str(slide_col).strip() != "":
                        try:
                            slide_col = str(int(slide_col))
                        except (ValueError, TypeError):
                            slide_col = str(slide_col)
                    else:
                        slide_col = ""
                    #slide_col = str(int(slide_col)) if pd.notna(slide_col) and str(slide_col).strip() != "" else ""
                    sample_type = str(sample_type) if pd.notna(sample_type) else ""
                    location = str(location) if pd.notna(location) else ""
                    subtitle_parts = [
                        f"Patient: {pid or 'NA'}",
                        f"Slide: {slide_col or 'NA'}",
                        f"Sample_type: {sample_type or 'NA'}",
                        f"Location: {location or 'NA'}",
                    ]

                wrapped = textwrap.fill(", ".join([p for p in subtitle_parts if p]), width=45)
                # draw metadata above image, and push title higher
                ax.text(
                    0.5,
                    1.01,
                    wrapped,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    transform=ax.transAxes,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1)
                )
                ax.set_title(sample_id, fontsize=9, pad=30)
            except Exception as e:
                ax.axis("off")
                ax.text(0.5, 0.5, f"Error\n{e}", ha="center", va="center", fontsize=8)

        # hide unused axes
        for j in range(len(found), len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)
        if verbose:
            print(f"✔ Saved grid to {out_pdf}\n")

    if verbose:
        print("✅ Done generating all image grids.")



def combine_pdfs(
    root_dir,
    folder_names,
    output_filename,
    title_font_size=24,
    pdf_name="spatial_plots.pdf",
):
    """
    Combine specific PDF files (default 'spatial_plots.pdf') from multiple Xenium folders into one PDF.

    Args:
        root_dir (str or Path): Root directory containing the Xenium folders.
        folder_names (list[str]): List of folder names to include (relative to root_dir).
        output_filename (str or Path): Path for the final combined PDF.
        title_font_size (int): Font size used by combine_folder_pdf() for title pages.
        pdf_name (str): Filename to look for inside each folder (e.g. "spatial_plots.pdf" or "patch_vis.pdf").

    Requirements:
        - Assumes resolve_folders() and combine_folder_pdf() exist and that combine_folder_pdf accepts a pdf_name arg.
    """
    root = Path(root_dir)
    folders = resolve_folders(root, folder_names)
    print(f"Found {len(folders)} folders to check.")

    writer = PdfWriter()
    appended_any = False
    missing = []

    for f in folders:
        ok = combine_folder_pdf(f, writer, pdf_name=pdf_name, title_font_size=title_font_size)
        if ok:
            appended_any = True
            print(f"Appended: {f}/{pdf_name} (title page: '{f.name}')")
        else:
            missing.append(f)
            print(f"Missing or empty: {f}/{pdf_name} — skipped")

    if not appended_any:
        print("No PDFs were appended. Exiting without writing output.")
        return

    # Save output PDF
    out_path = Path(output_filename)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as of:
        writer.write(of)

    print(f"Combined PDF written to: {out_path.resolve()}")

    if missing:
        print(f"\nFolders skipped (no {pdf_name} found):")
        for m in missing:
            print("  -", m)
