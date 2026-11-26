import io
import sys
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voice_import import import_voice_pack


class VoiceImportTests(unittest.TestCase):
    def test_imports_only_main_voice_and_index_and_is_repeatable(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            payload = io.BytesIO()
            torch.save(dict(weight={'sample': torch.ones(1)}, config=[48000], version='v2'), payload)
            archive = root / 'Example.zip'
            with ZipFile(archive, 'w') as pack:
                pack.writestr('Example/Example.pth', payload.getvalue())
                pack.writestr('Example/Example_e500.pth', payload.getvalue())
                pack.writestr('Example/added_Example.index', b'index fixture')
                pack.writestr('Example/install.bat', b'untrusted')
                pack.writestr('../outside.pth', payload.getvalue())
            library = root / 'Voix'
            path = import_voice_pack(archive, library)
            self.assertEqual(Path(path).name, 'Example.pth')
            self.assertEqual(sorted(p.name for p in Path(path).parent.iterdir()), ['Example.index', 'Example.pth'])
            self.assertEqual(import_voice_pack(archive, library), path)
            self.assertEqual(len(list(library.iterdir())), 1)

    def test_photos_are_not_a_voice_pack(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / 'photos.zip'
            with ZipFile(archive, 'w') as pack:
                pack.writestr('photos/selfie.tiff', b'photo fixture')
            with self.assertRaisesRegex(ValueError, 'RVC'):
                import_voice_pack(archive, root / 'Voix')


if __name__ == '__main__':
    unittest.main()
