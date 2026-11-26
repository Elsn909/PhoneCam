import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audio_routing import preferred_output, virtual_microphone


class AudioRoutingTests(unittest.TestCase):
    def test_basic_cable_is_preferred_to_old_headphones(self):
        self.assertEqual(preferred_output(['Headphones', 'CABLE Input (VB-Audio Virtual Cable)'], 'Headphones'),
                         'CABLE Input (VB-Audio Virtual Cable)')

    def test_previously_selected_cable_is_kept(self):
        self.assertEqual(preferred_output(['CABLE Input', 'CABLE-B Input'], 'CABLE Input'), 'CABLE Input')

    def test_pair_name_is_correct_for_basic_and_extra_cables(self):
        self.assertEqual(virtual_microphone('CABLE Input (VB-Audio Virtual Cable)'), 'CABLE Output')
        self.assertEqual(virtual_microphone('CABLE-B Input (VB-Audio Cable B)'), 'CABLE-B Output')
        self.assertEqual(virtual_microphone('Headphones'), '')


if __name__ == '__main__':
    unittest.main()
