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


def test_the_intel_macos_job_uses_a_label_that_still_has_runners():
    # macos-13 was used here first and never received a runner: the image is
    # retired, so the job sat queued indefinitely and, because publish depended on
    # the whole matrix, no release could go out at all. timeout-minutes does not
    # help, since it only counts once a job is running. macos-15-intel is the
    # current Intel image.
    import yaml

    workflow = (REPO / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    spec = yaml.safe_load(workflow)
    runners = {m["os"] for m in spec["jobs"]["build"]["strategy"]["matrix"]["include"]}
    assert "macos-13" not in runners
    assert "macos-15-intel" in runners


def test_all_four_platforms_publish_from_one_job():
    # An earlier version built Intel in a separate job with its own publish step,
    # to keep a slow runner off the critical path. That turned out to be solving
    # the wrong problem -- the label was retired, not scarce -- and it left a
    # window where the release existed with only one asset on it.
    import yaml

    spec = yaml.safe_load((REPO / ".github" / "workflows" / "release.yml").read_text())
    assert set(spec["jobs"]) == {"build", "publish"}
    labels = {m["label"] for m in spec["jobs"]["build"]["strategy"]["matrix"]["include"]}
    assert labels == {
        "linux-x86_64", "windows-x86_64", "macos-x86_64", "macos-arm64",
    }


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


def test_every_commit_to_main_publishes_a_build():
    workflow = (REPO / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "branches: [main]" in workflow
    assert "build-${GITHUB_RUN_NUMBER}" in workflow


def test_main_builds_are_prereleases_so_they_do_not_displace_a_tagged_version():
    # Otherwise every commit would steal the "Latest release" badge from the
    # version people are meant to download.
    workflow = (REPO / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "prerelease=true" in workflow
    assert "prerelease=false" in workflow
    assert "prerelease: ${{ steps.kind.outputs.prerelease }}" in workflow


def test_a_hand_triggered_run_builds_without_publishing():
    workflow = (REPO / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "if: github.event_name == 'push'" in workflow


def test_superseded_main_builds_are_cancelled_but_tagged_releases_are_not():
    workflow = (REPO / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "concurrency:" in workflow
    assert "cancel-in-progress: ${{ !startsWith(github.ref, 'refs/tags/') }}" in workflow


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


# --- the CLI's own surface ---------------------------------------------------


def test_help_does_not_crash():
    """`--help` must work. It is the first thing anyone types.

    argparse runs help strings through %-formatting, so a literal percent sign in
    one raises ValueError at the moment help is rendered -- and nowhere else, so
    nothing else in the suite would catch it. A "5% practice tolerance" in the
    --strict help did exactly that.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "microscopevuilder", "--help"],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert result.returncode == 0, result.stderr
    assert "--strict" in result.stdout


def test_every_flag_the_readme_documents_exists():
    """The README names specific flags; a renamed flag should fail here, not in
    front of someone following the instructions."""
    import argparse
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "microscopevuilder", "--help"],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    readme = (REPO / "README.md").read_text(encoding="utf-8")

    for flag in ("--list", "--round", "--ui", "--verify-catalog", "--bench", "--diff",
                 "--strict", "--no-measure"):
        assert flag in readme, f"{flag} is no longer documented"
        assert flag in result.stdout, f"{flag} is documented but does not exist"
