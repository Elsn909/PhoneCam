"""Regression tests without camera access, model downloads or GPU inference."""
import ast
import sys
import threading
import time
import unittest
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from swap_plus import DEFAULTS, SwapPlus


class BeardCleanupTests(unittest.TestCase):
    def engine(self):
        engine = SwapPlus.__new__(SwapPlus)
        engine.p = dict(DEFAULTS)
        engine.p.update(beard=True, beard_reach=45, expression=0, occ_steady=0)
        engine._color = {}
        engine._occ = {}
        engine._grain = {}
        engine.ms = {}
        engine._turn_cv = threading.Condition()
        engine._turn = 0
        engine._match_color = lambda key, swapped, crop, mask: swapped
        engine._match_grain = lambda key, warped, roi, mask: warped
        return engine

    def scene(self, bottom=245, beard_top=251):
        frame = np.full((320, 256, 3), 180, np.uint8)
        frame[beard_top:beard_top + 16, 112:144] = 25
        face = np.zeros((320, 256), np.float32)
        face[70:bottom, 70:186] = 1
        hair = np.zeros_like(face)
        hair[beard_top:beard_top + 16, 112:144] = 1
        return frame, face, hair

    def test_custom_regions_still_supply_face_and_hair_masks(self):
        engine = self.engine()
        engine.p['regions'] = ['skin', 'nose']
        labels = np.ones((256, 256), np.float32)
        labels[220:240, 110:145] = 17
        session = SimpleNamespace(run=lambda *args: [labels[None, None]])
        engine.ready_session = lambda name: session
        with patch('swap_plus.video_turn'):
            region, raw = engine._region(np.full((256, 256, 3), 180, np.uint8))
        self.assertIsNotNone(raw)
        self.assertEqual(raw[1][225, 120], 1)
        self.assertEqual(raw[0][225, 120], 0)
        self.assertEqual(raw[0][100, 100], 1)

    def test_legacy_beard_option_never_runs_local_inpainting(self):
        engine = self.engine()
        engine.p.update(parser=True, occlusion=True, beard=True)
        face = np.ones((256, 256), np.float32)
        engine._region = lambda crop: (face, (face, face))
        engine._occlusion = lambda crop: face
        _, _, beard = engine._side(np.full((256, 256, 3), 180, np.uint8))
        self.assertIsNone(beard)
        self.assertFalse(hasattr(engine, '_repair_beard'))

    def render(self, engine, frame, region, occlusion, relax=1, beard=None):
        future = Future()
        future.set_result((region, occlusion, beard))
        matrix = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
        job = dict(tm={}, work=256, crop=frame[:256], swapped=np.full((256,256,3), 220, np.uint8),
                   expr=None, side=future, track_id=0, relax=relax, M=matrix,
                   beard_enabled=True, t0=time.perf_counter())
        return engine._stage_b(frame.copy(), dict(jobs=[job], seq=0, alive={0}))

    def test_fast_motion_does_not_paint_over_hand(self):
        engine = self.engine()
        frame = np.full((320,256,3), 50, np.uint8)
        region = np.ones((256,256), np.float32)
        keep = region.copy()
        keep[90:200,70:170] = 0
        out = self.render(engine, frame, region, keep)
        np.testing.assert_array_equal(out[90:200,70:170], frame[90:200,70:170])
        self.assertGreater(float(out[40:60,40:60].mean()), 200)

    def test_fast_motion_does_not_expand_to_neck_or_background(self):
        engine = self.engine()
        frame = np.full((320,256,3), 50, np.uint8)
        region = np.zeros((256,256), np.float32)
        region[70:200,70:180] = 1
        out = self.render(engine, frame, region, np.ones_like(region))
        np.testing.assert_array_equal(out[205:], frame[205:])
        np.testing.assert_array_equal(out[:65], frame[:65])
        np.testing.assert_array_equal(out[:,:60], frame[:,:60])

    def test_stale_beard_mask_cannot_overwrite_neck(self):
        engine = self.engine()
        frame = np.full((320,256,3), 50, np.uint8)
        region = np.zeros((256,256), np.float32)
        region[70:200,70:180] = 1
        out = self.render(engine, frame, region, np.ones_like(region),
                          beard=np.ones((301,256), np.float32))
        np.testing.assert_array_equal(out[205:], frame[205:])

    def test_occlusion_smoothing_cannot_reintroduce_a_hand(self):
        engine = self.engine()
        engine.p['occ_steady'] = 100
        engine._occ[0] = np.ones((256,256), np.float32)
        current = np.ones((256,256), np.float32)
        current[90:180,100:105] = 0
        out = engine._steady(0,current)
        np.testing.assert_array_equal(out[90:180,100:105], current[90:180,100:105])
        self.assertTrue(np.all(out <= current))

    def test_ui_setting_reaches_live_engine(self):
        # Execute the real UI/engine setters without loading their camera/GPU imports.
        def method(file, cls, name):
            tree = ast.parse((ROOT / file).read_text(encoding='utf-8'))
            klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
            fn = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == name)
            ns = {}
            exec(compile(ast.Module(body=[fn], type_ignores=[]), file, 'exec'), ns)
            return ns[name]
        setter = method('engine.py', 'Engine', '_set_plus')
        engine = SimpleNamespace(plus_p=dict(DEFAULTS), _plus=self.engine(), plus_enabled=True)
        engine._plus_load = Mock()
        engine.set_option = lambda key, value: setter(engine, key[5:], value)
        ui = SimpleNamespace(settings={}, pipeline=SimpleNamespace(engine=engine))
        ui_set = method('ui_live.py', 'LiveSpace', 'engine_set')
        for key, value in [('plus_beard', False), ('plus_beard_reach', 57), ('plus_beard', True)]:
            ui_set(ui, key, value)
            self.assertEqual(engine._plus.p[key[5:]], value)
            self.assertEqual(ui.settings['dlc'][key], value)
        engine._plus_load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
