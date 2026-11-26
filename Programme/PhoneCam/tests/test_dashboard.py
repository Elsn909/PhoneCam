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
from ui_kit import Registry
from ui_live import LiveSpace


class DashboardTests(unittest.TestCase):
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

    def test_original_navigation_is_restored(self):
        self.assertFalse(self.live.rail.rail.isHidden())
        self.assertEqual(self.live.rail.current(), 1)
        self.assertFalse(hasattr(self.live, 'dashboard'))
        self.live.rail.select(self.live.diag_index)
        self.assertEqual(self.live.rail.current(), self.live.diag_index)

    def test_virtual_output_still_replaces_old_headphones(self):
        self.assertTrue(self.live.dev_out.currentText().startswith('CABLE Input'))


if __name__ == '__main__':
    unittest.main()
