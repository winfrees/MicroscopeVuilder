import os

# Qt must be told to run headless before QApplication is constructed.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest


@pytest.fixture(autouse=True)
def isolated_progress(tmp_path, monkeypatch):
    """Keep the test run out of the player's real save file.

    Progress lives in the per-user config directory, so a test that solves a round
    was writing into whoever ran the suite -- and reading their history back in,
    which made carry-forward tests depend on what a previous run had left behind.
    """
    from microscopevuilder.game import progress as progress_module

    save = tmp_path / "progress.json"
    monkeypatch.setattr(progress_module, "default_save_path", lambda: save)
    return save
