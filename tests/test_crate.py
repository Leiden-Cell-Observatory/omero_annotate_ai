"""RO-Crate description of a training set's folder."""

import json

import pytest

from omero_annotate_ai import provenance
from omero_annotate_ai.bundle import mask_path
from omero_annotate_ai.core.annotation_config import (
    AuthorInfo,
    SessionRecord,
    create_default_config,
)
from omero_annotate_ai.crate import METADATA_FILE, SOFTWARE_ID, SPEC, build_ro_crate
from omero_annotate_ai.rows import Row
from tests.fakes import annotate, create_set, omero_set


def _crate(tmp_path, *, n_images=3, limit=2, note="nuclei only", storage="omero"):
    config, source, omero = omero_set(tmp_path, n_images=n_images)
    config.storage = storage
    config.authors = [AuthorInfo(name="A. Researcher", affiliation="Lab")]
    ts = create_set(config, source, omero, tmp_path / "home")
    annotate(ts, limit=limit, note=note)
    path = ts.write_ro_crate()
    crate = json.loads(path.read_text(encoding="utf-8"))
    return ts, path, crate, {e["@id"]: e for e in crate["@graph"]}


def _identifiers(by_id, entity):
    """``{propertyID: value}`` of a file's identifier PropertyValues."""
    return {
        by_id[i["@id"]]["propertyID"]: by_id[i["@id"]]["value"]
        for i in _many(entity["identifier"])
    }


def _many(value):
    """A property value as a list (RO-Crate writes one value without a list)."""
    return value if isinstance(value, list) else [value]


def _refs(value):
    """Every ``{"@id": ...}`` reference inside a property value."""
    if isinstance(value, dict):
        return [value["@id"]] if set(value) == {"@id"} else []
    if isinstance(value, list):
        return [ref for item in value for ref in _refs(item)]
    return []


@pytest.mark.unit
class TestRoCrate:
    def test_descriptor_and_root_follow_the_spec(self, tmp_path):
        ts, path, crate, by_id = _crate(tmp_path)

        assert path == ts.bundle_dir / METADATA_FILE
        assert crate["@context"] == f"{SPEC}/context"
        descriptor = by_id[METADATA_FILE]
        assert descriptor["conformsTo"] == {"@id": SPEC}
        assert descriptor["about"] == {"@id": "./"}
        root = by_id["./"]
        assert root["@type"] == "Dataset"
        for key in ("name", "description", "datePublished", "license"):
            assert root[key], key
        assert by_id[root["license"]["@id"]]["name"] == "CC-BY-4.0"
        assert by_id[_many(root["author"])[0]["@id"]]["name"] == "A. Researcher"

    def test_flat_graph_with_every_reference_defined(self, tmp_path):
        _, _, crate, by_id = _crate(tmp_path)

        assert len(by_id) == len(crate["@graph"])  # ids are unique
        for entity in crate["@graph"]:
            for key, value in entity.items():
                if key.startswith("@") or key == "conformsTo":
                    continue
                assert not (isinstance(value, dict) and set(value) != {"@id"}), (
                    entity["@id"],
                    key,
                )
                for ref in _refs(value):
                    assert ref in by_id, (entity["@id"], key, ref)

    def test_files_exist_and_masks_point_to_their_images(self, tmp_path):
        ts, _, _, by_id = _crate(tmp_path)

        parts = [p["@id"] for p in by_id["./"]["hasPart"]]
        assert {"config.yaml", "tracking_table.csv"} <= set(parts)
        assert all((ts.bundle_dir / p).is_file() for p in parts)
        done = [r for r in ts.rows if r.processed]
        assert len(done) == 2
        for row in done:
            mask = by_id[f"annotation_output/{row.annotation_id}_mask.tif"]
            based_on = [ref["@id"] for ref in _many(mask["isBasedOn"])]
            assert based_on == [
                f"annotation_input/{row.annotation_id}.tif",
                row.image_uri,
            ]
            values = _identifiers(by_id, mask)
            assert values["pixel-sha256"] == row.mask_digest
            if row.mask_iscc:
                assert values["iscc"] == row.mask_iscc
        pending = next(r for r in ts.rows if not r.processed)
        assert f"annotation_output/{pending.annotation_id}_mask.tif" not in by_id

    def test_excluded_unit_lists_no_mask(self, tmp_path):
        ts, _, _, _ = _crate(tmp_path)
        unit = next(r for r in ts.rows if not r.processed)
        unit.mark_excluded(session=1)
        mask = mask_path(ts.bundle_dir, unit.annotation_id)
        mask.write_bytes(b"left over")  # e.g. kept after a conflicting upload

        crate = json.loads(ts.write_ro_crate().read_text(encoding="utf-8"))

        ids = {e["@id"] for e in crate["@graph"]}
        assert f"annotation_output/{unit.annotation_id}_mask.tif" not in ids

    def test_sessions_are_actions_with_agent_note_and_results(self, tmp_path):
        ts, _, _, by_id = _crate(tmp_path)

        (record,) = ts.config.sessions
        action = by_id[f"#session-{record.id}"]
        assert action["@type"] == "CreateAction"
        assert action["description"] == "nuclei only"
        assert action["actionStatus"] == "http://schema.org/CompletedActionStatus"
        assert {"@id": SOFTWARE_ID} in _many(action["instrument"])
        assert by_id[action["agent"]["@id"]]["@type"] == "Person"
        assert len(_many(action["result"])) == 2
        assert {"@id": action["@id"]} in _many(by_id["./"]["mentions"])

    def test_local_set_and_rewrite(self, tmp_path):
        ts, path, _, by_id = _crate(tmp_path, storage="local", limit=None)

        assert ts.is_local and len(by_id["./"]["hasPart"]) == 2 + 2 * 3
        ts.write_ro_crate()  # writing again replaces the file
        assert json.loads(path.read_text(encoding="utf-8"))["@graph"][1]["@id"] == "./"

    def test_without_the_provenance_extra_files_carry_only_the_digest(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(provenance, "_iscc_lib", lambda: None)
        ts, _, _, by_id = _crate(tmp_path)

        row = next(r for r in ts.rows if r.processed)
        assert row.mask_iscc is None
        mask = by_id[f"annotation_output/{row.annotation_id}_mask.tif"]
        assert _identifiers(by_id, mask) == {"pixel-sha256": row.mask_digest}

    def test_non_ascii_note_is_written_as_utf8(self, tmp_path):
        note = "Zellkerne – 5 µm, rand ✓"
        _, path, _, by_id = _crate(tmp_path, note=note)

        assert note in path.read_bytes().decode("utf-8")
        assert any(e.get("description") == note for e in by_id.values())

    def test_local_paths_never_enter_the_crate(self, tmp_path):
        local = "/home/someone/data/plate.zarr/A/1/0"
        url = "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/6001240.zarr"
        rows = [
            Row(image_uri=local, image_name="0", annotation_id="a_0_0"),
            Row(image_uri=url, image_name="6001240.zarr", annotation_id="b_0_0"),
        ]
        crate = build_ro_crate(create_default_config(), rows, tmp_path)

        text = json.dumps(crate)
        assert "/home/someone" not in text
        ids = {e["@id"] for e in crate["@graph"]}
        assert url in ids
        assert any(i.startswith("#source-") for i in ids)

    def test_similar_names_get_separate_entities(self, tmp_path):
        config = create_default_config()
        started = config.created
        config.sessions = [
            SessionRecord(id=1, started=started, user="a b"),
            SessionRecord(id=2, started=started, user="a-b"),
        ]
        crate = build_ro_crate(config, [], tmp_path)

        people = [e for e in crate["@graph"] if e.get("@type") == "Person"]
        assert sorted(p["name"] for p in people) == ["a b", "a-b"]
