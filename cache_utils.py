from pathlib import Path

import mne
from mne.preprocessing import read_ica


def ensure_parent_dir(path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_epochs(epochs, path, overwrite=True) -> Path:
    path = ensure_parent_dir(path)
    epochs.save(path, overwrite=overwrite)
    return path


def load_epochs(path):
    return mne.read_epochs(path, preload=True, verbose=False)


def save_ica(ica, path) -> Path:
    path = ensure_parent_dir(path)
    ica.save(path, overwrite=True)
    return path


def load_ica(path):
    return read_ica(path, verbose=False)


def cache_exists(path) -> bool:
    return Path(path).exists()
