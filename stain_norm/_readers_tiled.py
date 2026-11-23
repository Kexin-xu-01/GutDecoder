"""Memory-aware, streamed Reinhard normalization for large WSIs.

Dependencies: numpy, tifffile, zarr, scikit-image, pillow, and either
openslide-python or tiffslide for WSI files. Flat TIFF uses tifffile memmap
when possible. CZI uses aicsimageio lazy dask slicing.
"""
from pathlib import Path
import math
import numpy as np

from ._readers import _get_format, _mpp_from_tifffile, _mpp_or_guess, to_rgb_uint8
from loguru import logger


class SlideSource:
    def __init__(self, path, target_mpp=0.5):
        self.path = Path(path)
        self.kind = None
        self.obj = None
        self.arr = None
        self._temp_dir = None

        if not self.path.is_file():
            raise FileNotFoundError(self.path)

        if target_mpp <= 0 or not math.isfinite(float(target_mpp)):
            raise ValueError("target_mpp must be a positive finite number")

        fmt = _get_format(self.path)
        if fmt == 'openslide':
            native = self._open_openslide()
        elif fmt == 'tiff':
            native = self._open_tiff()
        else:  # czi
            native = self._open_czi()

        if self.kind == 'flat':
            self._orient_flat_array()

        self._set_output_geometry(native, target_mpp)

    def _open_openslide(self):
        """Open an OpenSlide format; returns the MPP stored in its metadata."""
        import openslide

        self.obj = openslide.OpenSlide(str(self.path))
        self.kind = 'wsi'

        native = self.obj.properties.get(openslide.PROPERTY_NAME_MPP_X)

        return native

    def _open_tiff(self):
        """Open a TIFF with tiffslide, else memmap or decode it; returns the stored MPP."""
        import tempfile

        import tifffile

        # Read authoritative MPP from TIFF metadata first.
        # Prefer OME PhysicalSizeX over TIFF resolution tags.
        with tifffile.TiffFile(str(self.path)) as tf:
            import xml.etree.ElementTree as ET

            native = None

            if tf.ome_metadata:
                root = ET.fromstring(tf.ome_metadata)
                pixels = root.find(".//{*}Pixels")

                if pixels is not None:
                    physical_x = pixels.get("PhysicalSizeX")
                    unit_x = pixels.get(
                        "PhysicalSizeXUnit", "µm"
                    )

                    if physical_x is not None:
                        units_to_um = {
                            "µm": 1.0,
                            "um": 1.0,
                            "nm": 0.001,
                            "mm": 1000.0,
                            "cm": 10000.0,
                            "m": 1000000.0,
                        }

                        if unit_x in units_to_um:
                            native = float(physical_x) * units_to_um[unit_x]

            if native is None:
                native = _mpp_from_tifffile(tf.series[0].pages[0])

        # Try tiffslide for efficient regional reading.
        try:
            import tiffslide

            self.obj = tiffslide.TiffSlide(
                str(self.path)
            )

            self.kind = 'wsi'

            logger.info(
                f"TIFF opened using tiffslide: "
                f"{self.path.name}"
            )

        except Exception as e:
            logger.info(
                f"tiffslide unavailable for "
                f"{self.path.name}: {e}"
            )

            if self.obj is not None:
                self.obj.close()

            self.obj = tifffile.TiffFile(
                str(self.path)
            )

            self.kind = 'flat'

            # First try direct memory mapping.
            try:
                self.arr = tifffile.memmap(
                    str(self.path),
                    series=0,
                )

                logger.info(
                    "Direct TIFF memory mapping successful."
                )

            except (ValueError, TypeError, RuntimeError):
                logger.info(
                    "TIFF not directly memory-mappable."
                )
                logger.info(
                    "Decoding into temporary "
                    "disk-backed array..."
                )

                self._temp_dir = (
                    tempfile.TemporaryDirectory(
                        prefix="stain_norm_"
                    )
                )

                temp_path = (
                    Path(self._temp_dir.name)
                    / "decoded.dat"
                )

                try:
                    self.arr = tifffile.imread(
                        str(self.path),
                        series=0,
                        out=str(temp_path),
                    )

                except Exception:
                    self._temp_dir.cleanup()
                    self._temp_dir = None
                    self.obj.close()
                    self.obj = None
                    raise

        return native

    def _open_czi(self):
        """Open a CZI lazily; returns the MPP stored in its metadata."""
        from ._czi import open_czi

        self.obj = open_czi(self.path)
        self.kind = 'czi'

        native = self.obj.physical_pixel_sizes.X

        # RGB CZIs store colour in the Samples dim 'S' (C has size 1);
        # scenes are selected via set_scene(), not an 'S' index.
        if 'S' in self.obj.dims.order:
            self.arr = self.obj.get_image_dask_data(
                'YXS',
                T=0,
                C=0,
                Z=0,
            )
        else:
            self.arr = self.obj.get_image_dask_data(
                'YXC',
                T=0,
                Z=0,
            )

        return native

    def _orient_flat_array(self):
        """Put a flat TIFF's channel axis last."""
        if self.arr.ndim == 2:
            pass

        elif self.arr.ndim == 3:
            if (
                self.arr.shape[0] in (1, 3, 4)
                and self.arr.shape[-1] not in (1, 3, 4)
            ):
                self.arr = np.moveaxis(
                    self.arr, 0, -1
                )

        else:
            raise ValueError(
                f"Unsupported TIFF shape: "
                f"{self.arr.shape}"
            )

    def _set_output_geometry(self, native, target_mpp):
        """MPP, pyramid level, downsample and output size at target_mpp."""
        self.native_mpp = _mpp_or_guess(native, self.path)

        self.scale = max(
            1.0,
            float(target_mpp) / self.native_mpp
        )

        if self.kind == 'wsi':

            self.level = (
                self.obj.get_best_level_for_downsample(
                    self.scale
                )
            )

            self.downsample = float(
                self.obj.level_downsamples[self.level]
            )

            self.width, self.height = map(
                int,
                self.obj.level_dimensions[self.level]
            )

            self.mpp = (
                self.native_mpp * self.downsample
            )

        else:
            self.downsample = self.scale

            self.height = max(
                1,
                math.ceil(
                    self.arr.shape[0] / self.scale
                )
            )

            self.width = max(
                1,
                math.ceil(
                    self.arr.shape[1] / self.scale
                )
            )

            self.mpp = (
                self.native_mpp * self.scale
            )

        logger.info(
            f"Slide: {self.path.name}\n"
            f"       Reader: {self.kind}\n"
            f"       Native MPP: {self.native_mpp:.6f}\n"
            f"       Target MPP: {target_mpp:.6f}\n"
            f"       Actual MPP: {self.mpp:.6f}\n"
            f"       Dimensions: "
            f"{self.width} x {self.height}"
        )

    def read(self, x, y, w, h):
        if self.kind == 'wsi':
            return np.asarray(self.obj.read_region((round(x * self.downsample), round(y * self.downsample)), self.level, (w, h)).convert('RGB'))
        from PIL import Image
        x0, y0 = round(x * self.scale), round(y * self.scale)
        x1, y1 = min(self.arr.shape[1], round((x + w) * self.scale)), min(self.arr.shape[0], round((y + h) * self.scale))
        a = self.arr[y0:y1, x0:x1]
        if self.kind == 'czi':
            a = a.compute()
        a = to_rgb_uint8(a)
        if a.shape[:2] != (h, w):
            a = np.asarray(Image.fromarray(a).resize((w, h), Image.Resampling.LANCZOS))
        return a

    def close(self):
        import gc

        if self.arr is not None:
            del self.arr
            self.arr = None

        if self.obj is not None:
            if hasattr(self.obj, "close"):
                self.obj.close()
            self.obj = None

        gc.collect()

        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def sample_slide(slide, sample_size=1024, grid=5, thumb=256):
    """Pixels from a grid x grid set of windows spread over the slide, as a (1, N, 3) uint8 image.

    Normalisers fit their source parameters on this sample, so a slide gets the same
    parameters whether it is later normalised whole or in tiles.
    """
    from PIL import Image
    pixels = []
    for j in range(grid):
        for i in range(grid):
            w = min(slide.width, max(1, slide.width // grid))
            h = min(slide.height, max(1, slide.height // grid))
            x = min(slide.width - w, round((slide.width - w) * (i + .5) / grid))
            y = min(slide.height - h, round((slide.height - h) * (j + .5) / grid))
            # For very large slides, read small windows rather than huge grid cells.
            w, h = min(w, sample_size), min(h, sample_size)
            rgb = slide.read(x, y, w, h)
            rgb = np.asarray(Image.fromarray(rgb).resize((min(thumb, w), min(thumb, h))))
            pixels.append(rgb.reshape(-1, 3))
    return np.concatenate(pixels)[None]


def write_tiled_ome_tiff(slide, output_path, transform, memory_limit_gb=8,
                         overlap=128, max_patch_px=None, bytes_per_pixel=64):
    """Run a fitted normalizer over memory-sized regions and stream OME-TIFF.

    Crops overlapping read halos, storing each output pixel exactly once.
    Normalisers with fixed source parameters (fit_source) give seam-free output.
    """
    import math
    import tempfile
    import tifffile
    import zarr

    if memory_limit_gb is None or memory_limit_gb <= 0:
        raise ValueError('Tiled mode requires memory_limit_gb > 0')
    if overlap < 0 or bytes_per_pixel <= 0:
        raise ValueError('Invalid overlap or working-memory estimate')
    capacity = math.floor(memory_limit_gb * 1024**3 / bytes_per_pixel)
    side = math.isqrt(capacity)
    if max_patch_px is not None:
        if max_patch_px <= 0:
            raise ValueError('max_patch_px must be positive')
        side = min(side, max_patch_px)
    side = (side // 16) * 16
    core = side - 2 * overlap
    if core < 16:
        raise ValueError('Memory budget too small for the selected overlap')
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f'Largest patch {side}x{side}, inner core {core}x{core}; mpp={slide.mpp}')
    with tempfile.TemporaryDirectory(prefix='stain_canvas_', dir=output_path.parent) as tmp:
        canvas = zarr.open(str(Path(tmp) / 'canvas.zarr'), mode='w',
                           shape=(slide.height, slide.width, 3),
                           chunks=(min(512, slide.height), min(512, slide.width), 3),
                           dtype='u1')
        total = math.ceil(slide.height/core) * math.ceil(slide.width/core)
        done = 0
        for cy0 in range(0, slide.height, core):
            cy1 = min(cy0+core, slide.height)
            for cx0 in range(0, slide.width, core):
                cx1 = min(cx0+core, slide.width)
                rx0, ry0 = max(0, cx0-overlap), max(0, cy0-overlap)
                rx1, ry1 = min(slide.width, cx1+overlap), min(slide.height, cy1+overlap)
                rgb = slide.read(rx0, ry0, rx1-rx0, ry1-ry0)
                result = np.asarray(transform(rgb))
                if result.shape != rgb.shape:
                    raise ValueError(f'Normalizer returned {result.shape}, expected {rgb.shape}')
                if result.dtype != np.uint8:
                    raise TypeError('Normalizer must return uint8 RGB')
                canvas[cy0:cy1, cx0:cx1] = result[cy0-ry0:cy1-ry0, cx0-rx0:cx1-rx0]
                del result, rgb
                done += 1
                print(f'  processed {done}/{total} patches', end='\r', flush=True)
        print()
        tile = 256
        def tile_generator():
            for y in range(0, slide.height, tile):
                for x in range(0, slide.width, tile):
                    block = np.full((tile, tile, 3), 255, np.uint8)
                    h = min(tile, slide.height-y)
                    w = min(tile, slide.width-x)
                    block[:h, :w] = canvas[y:y+h, x:x+w]
                    yield block
        px_cm = 1e4 / slide.mpp
        # Temporary target avoids leaving an apparently valid partial file.
        partial = output_path.with_name(output_path.name + '.partial')
        try:
            tifffile.imwrite(str(partial), data=tile_generator(),
                shape=(slide.height, slide.width, 3), dtype=np.uint8,
                photometric='rgb', tile=(tile, tile), compression='lzw',
                bigtiff=True, ome=True,
                metadata={'axes': 'YXS', 'PhysicalSizeX': slide.mpp,
                          'PhysicalSizeXUnit': 'µm', 'PhysicalSizeY': slide.mpp,
                          'PhysicalSizeYUnit': 'µm'},
                resolution=(px_cm, px_cm), resolutionunit='CENTIMETER')
            partial.replace(output_path)
        finally:
            partial.unlink(missing_ok=True)
    return output_path
