"""Tests for the LiDAR-CAT3 preprocessing helpers (LAS writing, class mapping, NIR lookup, NDVI)."""
import importlib.util
import os
import pickle

# import torch before numpy: some conda MKL builds of numpy need the OpenMP runtime torch loads
import torch  # noqa: F401
import laspy
import numpy as np

PROC = os.path.join(os.path.dirname(__file__), '..', 'proc', 'cat3_data')


def load(name):
    """
    Import a proc/cat3_data script by file name (the names start with a digit, so `import` cannot load them).

    :param name: str, script file name
    :return: module
    """
    spec = importlib.util.spec_from_file_location(name.replace('.py', ''), os.path.join(PROC, name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


split = load('1_split_las.py')
prep = load('2_preprocessing.py')


def window(rng, n=500):
    """
    Synthetic CAT3 window with realistic UTM coordinates, [9, n] float64 (x, y, z, class, intensity, R, G, B, NIR).
    """
    x = 436000 + rng.uniform(0, 40, n)
    y = 4620000 + rng.uniform(0, 40, n)
    z = 250 + rng.uniform(0, 30, n)
    cls = rng.choice([2, 3, 5, 6, 14, 104, 106, 135], n)
    intensity = rng.integers(0, 65535, n)
    rgb = rng.integers(0, 65535, (3, n))
    nir = rng.integers(0, 65535, n)
    return np.vstack((x, y, z, cls, intensity, rgb, nir)).astype(np.float64)


def test_map_classes_fits_point_format_3():
    out = split.map_classes(np.array([2., 6., 31., 104., 106., 135., 200.]))
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, [2, 6, 31, 31, 31, 30, 31])


def test_las_keeps_centimetres_and_nir_lookup(tmp_path):
    rng = np.random.default_rng(0)
    pc = window(rng)
    split.store_las_file_from_pc(pc, 'w1', str(tmp_path))

    las = laspy.read(tmp_path / 'w1.las')
    # coordinates within half a centimetre of the input (1 cm LAS grid), not truncated to metres
    for i, c in enumerate((las.x, las.y, las.z)):
        assert np.max(np.abs(np.asarray(c) - pc[i])) <= 0.005 + 1e-9
    # 16-bit attributes are not wrapped
    np.testing.assert_array_equal(np.asarray(las.intensity), pc[4].astype(np.uint16))
    np.testing.assert_array_equal(np.asarray(las.red), pc[5].astype(np.uint16))
    # classes above 31 end up in the noise codes removed by 2_preprocessing.py
    np.testing.assert_array_equal(np.asarray(las.classification), split.map_classes(pc[3]))
    assert set(np.asarray(las.classification)[pc[3] > 31]) <= set(prep.NOISY_CLASSES)

    # 2_preprocessing.py finds the NIR of every point with its own key function
    with open(tmp_path / 'w1_NIR.pkl', 'rb') as f:
        nir_dict = pickle.load(f)
    nir = np.array([nir_dict[k] for k in prep.nir_key(las.x, las.y, las.z)])
    keys = split.nir_key(las.x, las.y, las.z)
    unique = np.array([keys.count(k) == 1 for k in keys])  # points sharing a centimetre share one NIR entry
    np.testing.assert_array_equal(nir[unique], pc[8].astype(np.int64)[unique])


def test_nir_key_ignores_offsets():
    # a rewrite with another offset on the same 1 cm grid (e.g. by PDAL) gives the same key
    x = np.array([436012.34, 436000.01])
    y = np.array([4620001.99, 4620039.50])
    z = np.array([251.07, 280.00])
    assert split.nir_key(x, y, z) == prep.nir_key(x + 1e-9, y - 1e-9, z)


def test_ndvi_minus_one_when_no_signal():
    # NIR + red = 0 -> -1, stored as 0 after the [0, 1] rescale
    ndvi = prep.compute_ndvi(np.array([0, 100, 0, 300]), np.array([0, 100, 50, 100]))
    np.testing.assert_allclose(ndvi, [-1.0, 0.0, -1.0, 0.5])
    assert np.isfinite(ndvi).all()
