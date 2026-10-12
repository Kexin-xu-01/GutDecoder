"""Memory-aware, streamed Reinhard normalization for large WSIs.

Dependencies: numpy, tifffile, zarr, scikit-image, pillow, and either
openslide-python or tiffslide for WSI files. Flat TIFF uses tifffile memmap
when possible. CZI uses aicsimageio lazy dask slicing.
"""
from pathlib import Path
import math
import numpy as np


def _rgb(a):
    a = np.asarray(a)
    if a.ndim == 2:
        a = a[..., None]
    if a.ndim == 3 and a.shape[0] in (1, 3, 4) and a.shape[-1] not in (1, 3, 4):
        a = np.moveaxis(a, 0, -1)
    if a.shape[-1] == 1:
        a = np.repeat(a, 3, axis=-1)
    if a.shape[-1] >= 4:
        a = a[..., :3]
    if a.dtype != np.uint8:
        if np.issubdtype(a.dtype, np.integer):
            a = (a.astype(np.float32) * (255.0 / np.iinfo(a.dtype).max)).clip(0, 255).astype(np.uint8)
        else:
            a = (np.asarray(a, dtype=np.float32) * 255).clip(0, 255).astype(np.uint8)
    return a


class SlideSource:
    def __init__(self, path, target_mpp=0.5):
        self.path = Path(path)
        self.kind = None
        self.obj = None
        ext = self.path.suffix.lower()
        if ext in ('.svs', '.ndpi', '.mrxs', '.scn', '.bif', '.vms', '.vmu'):
            import openslide
            self.obj = openslide.OpenSlide(str(path))
            self.kind = 'wsi'
            native = float(self.obj.properties.get(openslide.PROPERTY_NAME_MPP_X) or 0.5)
        elif ext in ('.tif', '.tiff'):
            try:
                import tiffslide
                self.obj = tiffslide.TiffSlide(str(path))
                self.kind = 'wsi'
                native = float(self.obj.properties.get('tiffslide.mpp-x') or self.obj.properties.get('openslide.mpp-x') or 0.5)
            except Exception:
                import tifffile
                self.obj = tifffile.TiffFile(str(path))
                self.kind = 'flat'
                page = self.obj.pages[0]
                unit = page.tags.get('ResolutionUnit')
                res = page.tags.get('XResolution')
                native = 0.5
                if unit and res:
                    x = res.value
                    x = x[0] / x[1] if isinstance(x, tuple) else float(x)
                    u = int(unit.value)
                    if x > 0 and u in (2, 3):
                        native = (25400 if u == 2 else 10000) / x
                try:
                    self.arr = tifffile.memmap(str(path), series=0)
                except Exception as e:
                    self.obj.close()
                    raise RuntimeError('Flat TIFF is not memory-mappable. Convert it to tiled TIFF/OME-TIFF first to enable bounded-memory reading.') from e
        elif ext == '.czi':
            from aicsimageio import AICSImage
            self.obj = AICSImage(str(path))
            self.kind = 'czi'
            native = float(self.obj.physical_pixel_sizes.X or 0.5)
            self.arr = self.obj.get_image_dask_data('YXC', S=0, T=0, Z=0)
        else:
            raise ValueError(f'Unsupported slide: {path}')
        self.native_mpp = native if native > 0 else 0.5
        self.scale = max(1.0, float(target_mpp) / self.native_mpp)
        if self.kind == 'wsi':
            self.level = self.obj.get_best_level_for_downsample(self.scale)
            self.downsample = float(self.obj.level_downsamples[self.level])
            self.width, self.height = map(int, self.obj.level_dimensions[self.level])
            self.mpp = self.native_mpp * self.downsample
        else:
            self.downsample = self.scale
            self.height = max(1, math.ceil(self.arr.shape[0] / self.scale))
            self.width = max(1, math.ceil(self.arr.shape[1] / self.scale))
            self.mpp = self.native_mpp * self.scale

    def read(self, x, y, w, h):
        if self.kind == 'wsi':
            return np.asarray(self.obj.read_region((round(x * self.downsample), round(y * self.downsample)), self.level, (w, h)).convert('RGB'))
        from PIL import Image
        x0, y0 = round(x * self.scale), round(y * self.scale)
        x1, y1 = min(self.arr.shape[1], round((x + w) * self.scale)), min(self.arr.shape[0], round((y + h) * self.scale))
        a = self.arr[y0:y1, x0:x1]
        if self.kind == 'czi':
            a = a.compute()
        a = _rgb(a)
        if a.shape[:2] != (h, w):
            a = np.asarray(Image.fromarray(a).resize((w, h), Image.Resampling.LANCZOS))
        return a

    def close(self):
        if hasattr(self.obj, 'close'):
            self.obj.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()




def _stats(rgb):
    from skimage.color import rgb2lab
    lab = rgb2lab(rgb)
    # Exclude white background; avoid all-background samples.
    mask = np.mean(rgb, axis=-1) < 235
    v = lab[mask] if mask.sum() >= 100 else lab.reshape(-1, 3)
    return v.mean(axis=0), v.std(axis=0).clip(1e-4), len(v)


def estimate_slide_stats(slide, sample_size=1024, grid=5):
    from PIL import Image
    means, stds, counts = [], [], []
    for j in range(grid):
        for i in range(grid):
            w = min(slide.width, max(1, slide.width // grid))
            h = min(slide.height, max(1, slide.height // grid))
            x = min(slide.width - w, round((slide.width - w) * (i + .5) / grid))
            y = min(slide.height - h, round((slide.height - h) * (j + .5) / grid))
            # For very large slides, read small windows rather than huge grid cells.
            w, h = min(w, sample_size), min(h, sample_size)
            rgb = slide.read(x, y, w, h)
            rgb = np.asarray(Image.fromarray(rgb).resize((min(256, w), min(256, h))))
            m, s, n = _stats(rgb)
            means.append(m); stds.append(s); counts.append(n)
    counts = np.asarray(counts, dtype=float)
    means = np.asarray(means); stds = np.asarray(stds)
    mu = np.average(means, axis=0, weights=counts)
    var = np.average(stds ** 2 + (means - mu) ** 2, axis=0, weights=counts)
    return mu, np.sqrt(var).clip(1e-4)


def _normalize(rgb, source_stats, target_stats):
    from skimage.color import rgb2lab, lab2rgb
    mu_s, sd_s = source_stats
    mu_t, sd_t = target_stats
    lab = rgb2lab(rgb)
    lab = (lab - mu_s) * (sd_t / sd_s) + mu_t
    lab[..., 0] = lab[..., 0].clip(0, 100)
    # Preserve near-white background exactly to avoid a tinted canvas.
    background = rgb.mean(axis=-1) >= 245
    out = (lab2rgb(lab).clip(0, 1) * 255).astype(np.uint8)
    out[background] = rgb[background]
    return out



def write_tiled_ome_tiff(slide, output_path, transform, memory_limit_gb=8,
                         overlap=128, max_patch_px=None, bytes_per_pixel=64):
    """Run a fitted normalizer over memory-sized regions and stream OME-TIFF.

    Crops overlapping read halos, storing each output pixel exactly once.
    This is NOT feathering: tile-local color-statistics can still cause seams.
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
    print(f'[INFO] Largest patch {side}x{side}, inner core {core}x{core}; mpp={slide.mpp}')
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
