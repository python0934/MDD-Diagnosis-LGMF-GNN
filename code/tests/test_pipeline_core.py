from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

import utils
from config import Config
from step3_denoise import run as denoise
from step2_preprocess import (
    _preprocess_signature,
    _read_signature,
    _write_signature,
    slice_timing_correction,
)
from step4_parcellate import run as parcellate


class SliceTimingTests(unittest.TestCase):
    def _assert_alignment(self, n_slices, acquisition_sequence):
        tr = 2.0
        n_time = 100
        frequency = 0.05
        sample_times = np.arange(n_time) * tr
        ranks = np.empty(n_slices, dtype=int)
        ranks[acquisition_sequence] = np.arange(n_slices)
        acquisition_times = ranks / n_slices * tr
        data = np.zeros((1, 1, n_slices, n_time), dtype=np.float32)
        for k in range(n_slices):
            data[0, 0, k] = np.sin(
                2 * np.pi * frequency * (sample_times + acquisition_times[k])
            )

        corrected = slice_timing_correction(
            data, tr, slice_order=acquisition_sequence.tolist(), ref="middle"
        )
        expected = np.sin(2 * np.pi * frequency * (sample_times + tr / 2))
        for k in range(n_slices):
            np.testing.assert_allclose(corrected[0, 0, k], expected, atol=2e-5)

    def test_ascending_alignment(self):
        self._assert_alignment(6, np.arange(6))

    def test_interleaved_alignment(self):
        seq = np.concatenate([np.arange(0, 6, 2), np.arange(1, 6, 2)])
        self._assert_alignment(6, seq)

    def test_invalid_order_rejected(self):
        data = np.zeros((1, 1, 4, 8), dtype=np.float32)
        with self.assertRaises(ValueError):
            slice_timing_correction(data, 2.0, [0, 1, 1, 3])


class PathAndNiftiTests(unittest.TestCase):
    def test_save_4d_preserves_tr(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bold.nii.gz"
            utils.save_nii(np.zeros((2, 3, 4, 5)), np.eye(4), path, tr=2.5)
            img = nib.load(str(path))
            self.assertAlmostEqual(img.header.get_zooms()[3], 2.5)
            self.assertEqual(img.header.get_xyzt_units()[1], "sec")

    def test_save_4d_requires_tr(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                utils.save_nii(np.zeros((2, 2, 2, 2)), np.eye(4), Path(tmp) / "x.nii.gz")

    def test_restore_existing_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sub-0001"
            target = out / "func" / "sub-0001_clean.nii.gz"
            target.parent.mkdir(parents=True)
            target.touch()
            paths = utils.SubjPaths("sub-0001", out).restore_existing()
            self.assertEqual(paths.bold_clean, target)

    def test_protocol_slice_order_parser(self):
        with tempfile.TemporaryDirectory() as tmp:
            protocol = Path(tmp) / "MRI_protocols_rsMRI.tsv"
            protocol.write_text(
                "Protocol #\t1\t2\n"
                "Slice acquisition order\tAscending\tAscending (Interleaved)\n",
                encoding="utf-8",
            )
            self.assertEqual(
                utils.get_slice_order_map(tmp), {1: "ascending", 2: "interleaved"}
            )

    def test_preprocess_signature_detects_parameter_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sub-0001"
            bold = out / "func" / "sub-0001_bold.nii.gz"
            t1 = out / "anat" / "sub-0001_T1w_brain.nii.gz"
            utils.save_nii(np.zeros((2, 2, 2, 4)), np.eye(4), bold, tr=2.0)
            utils.save_nii(np.zeros((2, 2, 2)), np.eye(4), t1)
            paths = utils.SubjPaths("sub-0001", out).restore_existing()
            mni = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), np.eye(4))
            cfg = Config()
            first = _preprocess_signature(paths, cfg, 2.0, "ascending", mni)
            signature_path = out / "preprocess_signature.json"
            _write_signature(signature_path, first)
            self.assertEqual(_read_signature(signature_path), first)
            cfg.fwhm_mm = 8.0
            second = _preprocess_signature(paths, cfg, 2.0, "ascending", mni)
            self.assertNotEqual(first, second)


class DenoiseTests(unittest.TestCase):
    def test_nuisance_is_filtered_and_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sub-0001"
            tr = 2.0
            n_time = 100
            times = np.arange(n_time) * tr
            nuisance = np.sin(2 * np.pi * 0.05 * times).astype(np.float32)
            wanted = np.sin(2 * np.pi * 0.02 * times).astype(np.float32)
            data = np.zeros((3, 1, 1, n_time), dtype=np.float32)
            data[0, 0, 0] = 100 + wanted + nuisance
            data[1, 0, 0] = 100 + nuisance
            data[2, 0, 0] = 100 + wanted
            for name in ("norm", "smooth"):
                utils.save_nii(data, np.eye(4), out / "func" / f"sub-0001_{name}.nii.gz", tr=tr)
            wm = np.zeros((3, 1, 1), dtype=np.float32)
            wm[1, 0, 0] = 1
            utils.save_nii(wm, np.eye(4), out / "masks" / "sub-0001_wm_mask.nii.gz")

            template_mask = nib.Nifti1Image(np.ones((3, 1, 1), dtype=np.float32), np.eye(4))
            cfg = Config()
            cfg.do_motion = False
            cfg.nuisance = ("wm",)
            cfg.standardize = False
            paths = utils.SubjPaths("sub-0001", out).restore_existing()
            result = denoise(
                paths, cfg, {"tr_map": {1: tr}, "mni": {"mask": template_mask}},
                {"protocol": "1"},
            )
            cleaned = np.asarray(nib.load(str(result.bold_clean)).dataobj)
            self.assertLess(abs(np.corrcoef(cleaned[0, 0, 0], nuisance)[0, 1]), 0.02)
            self.assertAlmostEqual(nib.load(str(result.bold_clean)).header.get_zooms()[3], tr)


class ParcellationTests(unittest.TestCase):
    def test_zero_masked_voxels_are_not_in_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sub-0001"
            clean_path = out / "func" / "sub-0001_clean.nii.gz"
            signal = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
            data = np.zeros((2, 1, 1, 4), dtype=np.float32)
            data[0, 0, 0] = signal
            utils.save_nii(data, np.eye(4), clean_path, tr=2.0)

            atlas = nib.Nifti1Image(np.ones((2, 1, 1), dtype=np.int16), np.eye(4))
            paths = utils.SubjPaths("sub-0001", out).restore_existing()
            utils.set_use_gpu(False)
            result = parcellate(paths, Config(), {"aal": (atlas, [(1, "region")])})
            ts = np.load(result.ts)
            np.testing.assert_allclose(ts[0], signal)


if __name__ == "__main__":
    unittest.main()
