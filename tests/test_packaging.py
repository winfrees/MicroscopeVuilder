"""M9: the things that break only once the app is frozen."""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def test_catalog_resolves_from_a_frozen_bundle(monkeypatch, tmp_path):
    # PyInstaller unpacks data to a temp directory and points sys._MEIPASS at it,
    # so a path built from __file__ alone finds nothing in the packaged build.
    from microscopevuilder.bench import catalog

    bundled = tmp_path / "microscopevuilder" / "assets"
    bundled.mkdir(parents=True)
    (bundled / "components.toml").write_text(
        catalog.CATALOG_PATH.read_text(encoding="utf-8"), encoding="utf-8"
    )

    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert catalog._catalog_path() == bundled / "components.toml"


def test_catalog_resolves_from_source_when_not_frozen(monkeypatch):
    from microscopevuilder.bench import catalog

    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    path = catalog._catalog_path()
    assert path.exists()
    assert path.name == "components.toml"


def test_the_entry_point_defaults_to_opening_the_workspace():
    # A double-clicked executable has no arguments, so it must land somewhere
    # useful rather than printing usage to a console nobody can see.
    source = (REPO / "packaging" / "entry.py").read_text(encoding="utf-8")
    assert "--ui" in source
    assert "freeze_support" in source  # or one-file builds re-fork themselves


def test_the_spec_bundles_the_component_catalog():
    spec = (REPO / "packaging" / "microscopevuilder.spec").read_text(encoding="utf-8")
    assert "assets/*.toml" in spec, "the catalog is data and must be collected explicitly"
    assert 'name="MicroscopeVuilder"' in spec


def test_the_spec_builds_one_file_not_a_directory():
    # The deliverable is a single executable, not an installer or a folder.
    spec = (REPO / "packaging" / "microscopevuilder.spec").read_text(encoding="utf-8")
    assert "COLLECT(" not in spec, "COLLECT produces a one-folder build"
    assert "a.binaries" in spec and "a.datas" in spec  # folded into the EXE


def test_release_workflow_publishes_executables_for_every_platform():
    workflow = (REPO / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    for asset in (
        "MicroscopeVuilder-linux-x86_64",
        "MicroscopeVuilder-windows-x86_64.exe",
        "MicroscopeVuilder-macos-x86_64",
        "MicroscopeVuilder-macos-arm64",
    ):
        assert asset in workflow, asset
    assert "softprops/action-gh-release" in workflow
    assert 'tags: ["v*"]' in workflow
    assert "contents: write" in workflow


def test_release_workflow_tests_and_smoke_tests_before_publishing():
    workflow = (REPO / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "pytest -q" in workflow, "never ship an untested binary"
    # The smoke test must run a real round, not just check the file exists: that is
    # what catches a bundled catalog or a frozen import going missing.
    assert "--round 12" in workflow


def test_linux_builds_install_the_xcb_libraries_pyinstaller_needs_at_build_time():
    # Found by building locally: without these, PyInstaller cannot resolve the Qt
    # xcb platform plugin's dependencies and silently ships a broken binary.
    workflow = (REPO / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    for library in ("libxcb-cursor0", "libxkbcommon-x11-0", "libxcb-icccm4"):
        assert library in workflow, library


def test_linux_release_is_built_on_the_oldest_supported_runner():
    # glibc is forward compatible but not backward: a binary built on 24.04 will
    # not start on an older lab machine.
    workflow = (REPO / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "ubuntu-22.04" in workflow
