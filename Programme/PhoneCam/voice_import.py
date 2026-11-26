"""Import local d'un pack RVC : uniquement les poids et leur index, jamais de script."""
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from zipfile import ZipFile


def import_voice_pack(archive, library):
    import torch

    library = Path(library).resolve()
    library.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as pack:
        entries = []
        for info in pack.infolist():
            part = PurePosixPath(info.filename.replace('\\', '/'))
            if (info.is_dir() or part.is_absolute() or '..' in part.parts
                    or '__MACOSX' in part.parts or any(p.startswith('.') for p in part.parts)):
                continue
            if part.suffix.lower() in ('.pth', '.index'):
                if info.file_size > 1024 ** 3:
                    raise ValueError('Un fichier du pack depasse 1 Go.')
                entries.append((info, part))
        weights = [(info, part) for info, part in entries
                   if part.suffix.lower() == '.pth' and not part.name.startswith(('G_', 'D_', 'f0G', 'f0D'))]
        if not weights:
            raise ValueError('Ce ZIP ne contient pas de modele RVC .pth.')
        # Un pack peut contenir plusieurs checkpoints de la meme voix : proposer le modele principal.
        stem = Path(archive).stem
        weights.sort(key=lambda item: (item[1].stem.lower() != stem.lower(),
                                      '_e' in item[1].stem.lower(), len(item[1].name)))
        info, part = weights[0]
        safe_name = ''.join(c for c in part.stem if c.isalnum() or c in '-_ ')[:80].strip() or 'Voix importee'
        with tempfile.TemporaryDirectory(prefix='.import-', dir=library) as scratch:
            temporary = Path(scratch).resolve()
            if not temporary.is_relative_to(library):
                raise ValueError('Dossier temporaire hors de la bibliotheque.')
            pth = temporary / (safe_name + '.pth')
            with pack.open(info) as src, pth.open('wb') as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
            checkpoint = torch.load(pth, map_location='cpu', weights_only=True)
            if (not isinstance(checkpoint, dict) or 'weight' not in checkpoint or 'config' not in checkpoint
                    or checkpoint.get('version', 'v1') not in ('v1', 'v2')):
                raise ValueError('Le fichier ne correspond pas a une voix RVC compatible.')
            if checkpoint.get('vocoder', 'HiFi-GAN') != 'HiFi-GAN':
                raise ValueError('Ce vocodeur nest pas encore pris en charge (HiFi-GAN requis).')
            if checkpoint.get('embedder_model', 'contentvec') not in ('contentvec', 'hubert'):
                raise ValueError('Ce modele utilise un encodeur vocal non pris en charge.')
            indices = [(i, p) for i, p in entries if p.suffix.lower() == '.index']
            indices.sort(key=lambda item: (item[1].parent != part.parent,
                                          part.stem.lower() not in item[1].stem.lower(),
                                          not item[1].name.lower().startswith('added_')))
            if indices:
                index, _ = indices[0]
                with pack.open(index) as src, (temporary / (safe_name + '.index')).open('wb') as dst:
                    shutil.copyfileobj(src, dst, 1024 * 1024)
            target = library / safe_name
            if target.exists():
                existing = target / pth.name
                if existing.exists() and _hash(existing) == _hash(pth):
                    return str(existing)
                suffix = 2
                while (library / f'{safe_name} {suffix}').exists():
                    suffix += 1
                target = library / f'{safe_name} {suffix}'
            # Move only the reviewed data files; TemporaryDirectory cleans its own validated directory.
            target.mkdir()
            for item in temporary.iterdir():
                os.replace(item, target / item.name)
            return str(target / pth.name)


def _hash(path):
    with open(path, 'rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()
