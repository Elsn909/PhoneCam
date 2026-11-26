import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PySide6.QtWidgets import QApplication
from phonecam import Pipeline
from server import FrameSource
from ui_kit import Registry, ValueBar
from ui_live import LiveSpace


class SimpleModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.patches = [
            patch('webcam.cameras', return_value=['Test Camera']),
            patch('webcam.PcWebcam.start'),
            patch('voix.Voix.load'),
            patch('phonemic.devices', side_effect=lambda kind: ['Test Microphone'] if kind == 'input' else
                  ['Headphones', 'CABLE Input (VB-Audio Virtual Cable)']),
            patch('phonemic.default_device', return_value='Test Microphone'),
            patch('phonemic.rescan'),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        self.frames = FrameSource()
        self.pipe = Pipeline(self.frames)
        self.settings = {'mic_devices': {'output': 'Headphones'}}
        self.live = LiveSpace(self.pipe, self.frames, self.settings, Registry(self.settings), lambda text: None)
        self.live.timer.stop()
        self.addCleanup(self.live.close)

    def test_simple_is_default_and_advanced_navigation_still_works(self):
        self.assertTrue(self.live.simple_mode)
        self.assertEqual(self.live.rail.current(), self.live.home_index)
        self.live.rail.select(self.live.diag_index)
        self.assertFalse(self.live.simple_mode)
        self.live.set_simple_mode(True)
        self.assertEqual(self.live.rail.current(), self.live.home_index)

    def test_simple_toggles_and_pitch_reach_existing_controls(self):
        dashboard = self.live.dashboard
        self.assertEqual(len(dashboard.findChildren(ValueBar)), 1)
        self.live.voice_page.on.setChecked(False)
        dashboard.pitch.setValue(7)
        self.assertEqual(self.pipe.voice.p['pitch'], 7)
        self.live.swap_on.setChecked(False)
        self.assertFalse(self.live.swap_on.isChecked())

    def test_virtual_output_replaces_old_headphones_and_hint_matches(self):
        self.assertTrue(self.live.dev_out.currentText().startswith('CABLE Input'))
        self.assertIn('CABLE Output', self.live.dashboard.routing.text())


if __name__ == '__main__':
    unittest.main()
