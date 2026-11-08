"""A training set's folder on disk (bundle.py)."""

from pathlib import Path

import pytest

from omero_annotate_ai.bundle import (
    HOME_ENV,
    atomic_write,
    collect_masks,
    fresh_scratch,
    home_dir,
    local_bundle,
    mask_path,
    new_local_id,
    omero_bundle,
    server_slug,
)


@pytest.mark.unit
class TestHomeDir:
    def test_argument_wins_over_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv(HOME_ENV, str(tmp_path / "env"))
        assert home_dir(tmp_path / "arg") == tmp_path / "arg"

    def test_environment_wins_over_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv(HOME_ENV, str(tmp_path / "env"))
        assert home_dir() == tmp_path / "env"

    def test_default_is_in_the_user_home(self, monkeypatch):
        monkeypatch.delenv(HOME_ENV, raising=False)
        assert home_dir() == Path.home() / "omero_annotate_ai"


@pytest.mark.unit
class TestFolderNames:
    @pytest.mark.parametrize(
        ("host", "port", "slug"),
        [
            ("omero.example.org", 4064, "omero.example.org-4064"),
            ("wss://omero.example.org/omero-ws", None, "omero.example.org-omero-ws"),
            ("LOCALHOST", 6064, "localhost-6064"),
        ],
    )
    def test_server_slug(self, host, port, slug):
        assert server_slug(host, port) == slug

    def test_omero_bundle_is_named_after_server_and_table(self, tmp_path):
        folder = omero_bundle(tmp_path, "localhost-6064", 42, "Nuclei set #1")
        assert folder == tmp_path / "localhost-6064" / "42_nuclei-set-1"

    def test_local_bundles_get_distinct_ids(self, tmp_path):
        first = local_bundle(tmp_path, new_local_id(), "run")
        second = local_bundle(tmp_path, new_local_id(), "run")
        assert first.parent == tmp_path / "local"
        assert first.name.endswith("_run") and first != second

    def test_mask_path(self, tmp_path):
        assert (
            mask_path(tmp_path, "a1") == tmp_path / "annotation_output" / "a1_mask.tif"
        )


@pytest.mark.unit
class TestAtomicWrite:
    def test_writes_and_leaves_no_temp_file(self, tmp_path):
        target = tmp_path / "sub" / "table.csv"
        atomic_write(target, lambda tmp: tmp.write_text("x,y\n"))
        assert target.read_text() == "x,y\n"
        assert [p.name for p in target.parent.iterdir()] == ["table.csv"]

    def test_failed_write_keeps_the_old_file(self, tmp_path):
        target = tmp_path / "table.csv"
        target.write_text("old")

        def fail(tmp: Path) -> None:
            tmp.write_text("half")
            raise OSError("disk full")

        with pytest.raises(OSError, match="disk full"):
            atomic_write(target, fail)
        assert target.read_text() == "old"
        assert [p.name for p in tmp_path.iterdir()] == ["table.csv"]


@pytest.mark.unit
class TestScratchAndMasks:
    def test_fresh_scratch_removes_left_overs(self, tmp_path):
        stale = fresh_scratch(tmp_path, 1, 1) / "seg_00000.tif"
        stale.write_bytes(b"old")
        scratch = fresh_scratch(tmp_path, 1, 1)
        assert scratch == tmp_path / ".scratch" / "s1-b1"
        assert list(scratch.iterdir()) == []

    def test_masks_are_matched_by_exact_name(self, tmp_path):
        (tmp_path / "seg_00002.tif").write_bytes(b"")
        (tmp_path / "seg_00000.tiff").write_bytes(b"")
        (tmp_path / "other.tif").write_bytes(b"")
        masks = collect_masks(tmp_path, ["a", "b", "c"])
        assert {k: p.name for k, p in masks.items()} == {
            "a": "seg_00000.tiff",
            "c": "seg_00002.tif",
        }
