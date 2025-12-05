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


def make_spatial_plot_per_run(
    base_root,
    datasets,
    cols=4,
    max_show=None,
    thumb_size=(800, 800)
):
    """
    Generate spatial_plots.pdf grids for each dataset folder.

    Args:
        base_root (str or Path): Top-level directory containing dataset folders.
        datasets (list[str]): List of dataset folder names under base_root.
        cols (int): Number of columns in each grid.
        max_show (int or None): Limit number of images per dataset. None = all.
        thumb_size (tuple): Thumbnail size for speeding up plotting.

    Produces:
        spatial_plots.pdf inside each dataset folder.
    """
    base_root = Path(base_root)

    for ds in datasets:
        ds_dir = base_root / ds
        if not ds_dir.exists():
            print(f"⚠️ {ds_dir} not found, skipping.")
            continue

        found = sorted(ds_dir.rglob("spatial_plots.png"))
        if not found:
            print(f"⚠️ No spatial_plots.png found in {ds_dir}")
            continue

        if max_show:
            found = found[:max_show]

        print(f"🧩 {ds}: visualising {len(found)} plots")

        # --- build grid ---
        rows = math.ceil(len(found) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        axes = axes.flatten()

        for i, (ax, path) in enumerate(zip(axes, found)):
            try:
                img = Image.open(path)
                img.thumbnail(thumb_size)
                ax.imshow(img)
                rel = path.relative_to(ds_dir)
                ax.set_title(str(rel.parent), fontsize=8)
                ax.axis("off")
            except Exception as e:
                ax.axis("off")
                ax.text(0.5, 0.5, f"Error\n{e}", ha="center", va="center", fontsize=8)

        # Hide unused axes
        for j in range(len(found), len(axes)):
            axes[j].axis("off")

        plt.tight_layout()

        # --- save PDF ---
        out_pdf = ds_dir / "spatial_plots.pdf"
        plt.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)

        print(f"✔ Saved grid to {out_pdf}\n")

    print("✅ Done generating all spatial_plots.pdf files.")


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



def make_patch_vis_per_run(
    base_root,
    datasets,
    cols=4,
    max_show=None,
    thumb_size=(800, 800),
):
    """
    Generate patch_vis.pdf grids for each dataset folder.

    Args:
        base_root (str or Path): Root directory containing dataset folders.
        datasets (list[str]): Folder names (relative to base_root).
        cols (int): Number of grid columns.
        max_show (int or None): Limit number of images per dataset (None = all).
        thumb_size (tuple): Image thumbnail resize to accelerate plotting.
    """

    base_root = Path(base_root)

    for ds in datasets:
        ds_dir = base_root / ds

        if not ds_dir.exists():
            print(f"⚠️ {ds_dir} not found, skipping.")
            continue

        # find any filename containing "patch_vis", including ROI9_patch_vis.png
        found = sorted(ds_dir.rglob("*patch_vis*.png"))
        if not found:
            print(f"⚠️ No *patch_vis*.png found in {ds_dir}")
            continue

        if max_show:
            found = found[:max_show]

        print(f"🧩 {ds}: visualising {len(found)} plots")

        # --- build grid ---
        rows = math.ceil(len(found) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        axes = axes.flatten()

        for ax, path in zip(axes, found):
            try:
                img = Image.open(path)
                img.thumbnail(thumb_size)
                ax.imshow(img)

                rel = path.relative_to(ds_dir)
                ax.set_title(str(rel.parent), fontsize=8)
                ax.axis("off")
            except Exception as e:
                ax.axis("off")
                ax.text(0.5, 0.5, f"Error\n{e}", ha="center", va="center", fontsize=8)

        # Hide unused axes
        for j in range(len(found), len(axes)):
            axes[j].axis("off")

        plt.tight_layout()

        # Save PDF
        out_pdf = ds_dir / "patch_vis.pdf"
        plt.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)
        print(f"✔ Saved grid to {out_pdf}\n")

    print("✅ Done generating all patch_vis.pdf files.")


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
