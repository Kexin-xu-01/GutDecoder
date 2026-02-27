import os
import glob
from typing import List, Optional
from PIL import Image, ImageDraw, ImageFont

def _img_path(run_dir: str, subdir: str, sample: str, suffix: str) -> str:
    return os.path.join(run_dir, subdir, f"{sample}{suffix}")

def _open_or_placeholder(path: str, size: tuple, missing_text: str):
    if path and os.path.exists(path):
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            pass
    # placeholder
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    try:
        f = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
    except Exception:
        f = ImageFont.load_default()
    w, h = draw.textsize(missing_text, font=f)
    draw.text(((size[0]-w)/2, (size[1]-h)/2), missing_text, fill="black", font=f)
    return img

def combine_run_plots_to_pdf(
    run_dir: str,
    out_pdf: Optional[str] = None,
    rows_per_page: int = 4,
    page_size: tuple = (2480, 3508),  # A4 ~300dpi
    spatial_subdir: str = "spatial_plots",
    conf_subdir: str = "confidence_plots",
    spatial_suffix: str = "_spatial_plot.png",
    conf_suffix: str = "_confidence_plot.png",
) -> str:
    if out_pdf is None:
        out_pdf = os.path.join(run_dir, "combined_plots.pdf")

    # collect sample names from spatial and confidence folders
    sp_dir = os.path.join(run_dir, spatial_subdir)
    cf_dir = os.path.join(run_dir, conf_subdir)
    samples = set()
    if os.path.isdir(sp_dir):
        for p in glob.glob(os.path.join(sp_dir, f"*{spatial_suffix}")):
            samples.add(os.path.splitext(os.path.basename(p))[0].replace(spatial_suffix[:-4], ""))
    if os.path.isdir(cf_dir):
        for p in glob.glob(os.path.join(cf_dir, f"*{conf_suffix}")):
            samples.add(os.path.splitext(os.path.basename(p))[0].replace(conf_suffix[:-4], ""))
    samples = sorted(samples)
    if not samples:
        raise ValueError("No samples found under spatial/confidence folders.")

    page_w, page_h = page_size
    row_h = page_h // rows_per_page
    half_w = page_w // 2
    label_h = max(36, row_h // 10)
    img_area = (half_w, row_h - label_h)

    pages = []
    page = Image.new("RGB", page_size, "white")
    row_idx = 0

    for sample in samples:
        sp_path = _img_path(run_dir, spatial_subdir, sample, spatial_suffix)
        cf_path = _img_path(run_dir, conf_subdir, sample, conf_suffix)

        sp = _open_or_placeholder(sp_path, img_area, "MISSING SPATIAL")
        cf = _open_or_placeholder(cf_path, img_area, "MISSING CONFIDENCE")

        # fit and center into area
        def fit(im, target):
            tw, th = target
            iw, ih = im.size
            scale = min(tw/iw, th/ih)
            nw, nh = int(iw*scale), int(ih*scale)
            imr = im.resize((nw, nh), Image.LANCZOS)
            canvas = Image.new("RGB", target, "white")
            canvas.paste(imr, ((tw-nw)//2, (th-nh)//2))
            return canvas

        spc = fit(sp, img_area)
        cfc = fit(cf, img_area)

        row = Image.new("RGB", (page_w, row_h), "white")
        row.paste(spc, (0, label_h))
        row.paste(cfc, (half_w, label_h))

        # draw label
        draw = ImageDraw.Draw(row)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", label_h//2)
        except Exception:
            font = ImageFont.load_default()
        draw.text((8, (label_h - font.getsize(sample)[1])//2), sample, fill="black", font=font)

        page.paste(row, (0, row_idx * row_h))
        row_idx += 1

        if row_idx >= rows_per_page:
            pages.append(page)
            page = Image.new("RGB", page_size, "white")
            row_idx = 0

    if row_idx > 0:
        pages.append(page)

    # save multipage PDF
    pages[0].save(out_pdf, "PDF", resolution=300.0, save_all=True, append_images=pages[1:])
    return out_pdf