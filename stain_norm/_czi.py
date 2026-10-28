"""Compatibility reader for RGB CZI mosaics."""

import numpy as np


def _stitch_tiles(data, data_dims, data_dims_shape, tile_bboxes, final_bbox):
    """Stitch by axis names, preserving the RGB samples axis.

    AICSImageIO's stitcher uses string identity for axis names and can drop
    the samples axis. Here S means samples; scene coordinates in tile
    metadata also use S and must not be used to index this axis.
    """
    dims = list(data_dims)
    output_dims = [dim for dim in dims if dim not in ("B", "M")]
    shape = [
        final_bbox.h if dim == "Y" else
        final_bbox.w if dim == "X" else data.shape[dims.index(dim)]
        for dim in output_dims
    ]
    if isinstance(data, np.ndarray):
        result = np.zeros(shape, dtype=data.dtype)
    else:
        import dask.array as da

        result = da.zeros(tuple(shape), dtype=data.dtype)

    for tile_info, box in tile_bboxes.items():
        coordinates = tile_info.dimension_coordinates
        source = [
            slice(None) if dim in ("Y", "X", "S") else
            coordinates.get("B", 0) if dim == "B" else coordinates[dim]
            for dim in dims
        ]
        destination = []
        for dim in output_dims:
            if dim == "Y":
                start = box.y - final_bbox.y
                destination.append(slice(start, start + box.h))
            elif dim == "X":
                start = box.x - final_bbox.x
                destination.append(slice(start, start + box.w))
            elif dim == "S":
                destination.append(slice(None))
            else:
                destination.append(coordinates[dim])
        result[tuple(destination)] = data[tuple(source)]
    return result


def open_czi(path):
    """Use the corrected stitcher without changing AICSImageIO globally."""
    from aicsimageio import AICSImage
    from aicsimageio.readers.czi_reader import CziReader

    class RGBMosaicCziReader(CziReader):
        _stitch_tiles = staticmethod(_stitch_tiles)

    return AICSImage(str(path), reader=RGBMosaicCziReader)
