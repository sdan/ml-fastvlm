#!/usr/bin/env python
"""
Quick sanity check for AnyRes tile-aware pointer mapping.

Scenario: 1920x1080 image, pinpoints [(1024,1024),(2048,1024)], tile_side=1024, base_per_side=16.
Expected: best resolution is (2048,1024) -> grid 2x1. A click at (0.5,0.5) is on the vertical
tile boundary, so both the right edge of the left tile and the left edge of the right tile
should be marked. With the 'spatial' flatten order used in llava_arch, indices should be {527, 528}
with total vlen=768 (base 256 + 2 tiles * 256).
"""

import importlib.util
import os
import sys


def _import_mm_utils():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, os.pardir))
    path = os.path.join(root, "llava", "mm_utils.py")
    # Stub minimal llava.constants to satisfy mm_utils import without heavy deps
    import types
    if 'llava' not in sys.modules:
        sys.modules['llava'] = types.ModuleType('llava')
    if 'llava.constants' not in sys.modules:
        const_mod = types.ModuleType('llava.constants')
        const_mod.IMAGE_TOKEN_INDEX = -1
        sys.modules['llava.constants'] = const_mod
    spec = importlib.util.spec_from_file_location("mm_utils", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def main():
    mm_utils = _import_mm_utils()
    image_size = (1920, 1080)
    pinpoints = [(1024, 1024), (2048, 1024)]
    tile_side = 1024
    base_per_side = 16
    coords = [(0.5, 0.5)]

    indices, vlen = mm_utils.compute_anyres_pointer_indices(image_size, pinpoints, tile_side, base_per_side, coords)
    print({
        "image_size": image_size,
        "grid_pinpoints": pinpoints,
        "tile_side": tile_side,
        "base_per_side": base_per_side,
        "coords": coords,
        "indices": indices,
        "vlen": vlen,
    })

    expected_vlen = 256 + 2 * 256
    expected = {527, 528}
    assert vlen == expected_vlen, f"Unexpected vlen: {vlen}, expected {expected_vlen}"
    assert set(indices) == expected, f"Unexpected indices: {indices}, expected {sorted(expected)}"
    print("PASS: indices and vlen match expected values.")


if __name__ == "__main__":
    main()
