import pytest
from PIL import Image
from pypdf import PdfWriter

from gutdecoder.vis.spatial_plot import TARGET_WIDTH_PT, combine_folder_pdf


def test_combine_folder_pdf_scales_pages_to_target_width(tmp_path):
    folder = tmp_path / "XeniumPR1"
    folder.mkdir()
    Image.new("RGB", (800, 600)).save(folder / "spatial_plots.pdf", "PDF", resolution=72.0)
    writer = PdfWriter()

    assert combine_folder_pdf(folder, writer) is True

    title, page = writer.pages
    assert float(page.mediabox.width) == pytest.approx(TARGET_WIDTH_PT)
    assert float(page.mediabox.height) == pytest.approx(TARGET_WIDTH_PT * 600 / 800)


def test_combine_folder_pdf_skips_folder_without_pdf(tmp_path):
    assert combine_folder_pdf(tmp_path, PdfWriter()) is False
