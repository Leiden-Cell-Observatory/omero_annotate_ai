"""A training set as Bioimage Annotation Schema documents, and its RO-Crate."""

import json

import pytest
from bioimage_annotation_schema import datamodel as bas
from bioimage_annotation_schema import read_records, read_set
from bioimage_annotation_schema.crate import METADATA_FILE, SPEC, build_ro_crate

from omero_annotate_ai import provenance
from omero_annotate_ai.bundle import mask_path
from omero_annotate_ai.core.annotation_config import (
    SessionRecord,
    create_default_config,
)
from omero_annotate_ai.rows import Row
from omero_annotate_ai.schema_export import (
    annotation_records,
    annotation_set,
    write_schema_documents,
)
from tests.fakes import annotate, create_set, omero_set


def _crate(tmp_path, *, n_images=3, limit=2, note="nuclei only", storage="omero"):
    config, source, omero = omero_set(tmp_path, n_images=n_images)
    config.storage = storage
    config.authors = [
        bas.Person(
            first_name="A.",
            last_name="Researcher",
            affiliations=[bas.Organisation(name="Lab")],
        )
    ]
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
        assert {
            "annotation-set.yaml",
            "annotation-records.csv",
            "config.yaml",
            "tracking_table.csv",
        } <= set(parts)
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
        software = [by_id[i["@id"]]["name"] for i in _many(action["instrument"])]
        assert any(name.startswith("omero-annotate-ai") for name in software)
        assert by_id[action["agent"]["@id"]]["@type"] == "Person"
        assert len(_many(action["result"])) == 2
        assert {"@id": action["@id"]} in _many(by_id["./"]["mentions"])

    def test_local_set_and_rewrite(self, tmp_path):
        ts, path, _, by_id = _crate(tmp_path, storage="local", limit=None)

        assert ts.is_local and len(by_id["./"]["hasPart"]) == 4 + 2 * 3
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
        crate = build_ro_crate(
            annotation_set(create_default_config()), annotation_records(rows), tmp_path
        )

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
        crate = build_ro_crate(annotation_set(config), [], tmp_path)

        people = [e for e in crate["@graph"] if e.get("@type") == "Person"]
        assert sorted(p["name"] for p in people) == ["a b", "a-b"]


@pytest.mark.unit
class TestSchemaDocuments:
    """The set's folder holds the two schema documents, valid and complete."""

    def test_round_trip_through_the_schema_library(self, tmp_path):
        ts, _, _, _ = _crate(tmp_path, limit=2)

        annotation_set_ = read_set(ts.bundle_dir)  # validated on read
        records = read_records(ts.bundle_dir)
        assert annotation_set_.name == ts.config.name
        assert annotation_set_.sessions[0].note == "nuclei only"
        assert len(records) == len(ts.rows)
        by_id = {r.annotation_id: r for r in records}
        for row in ts.rows:
            record = by_id[row.annotation_id]
            assert record.image_uri == row.image_uri
            assert record.split == row.category
            assert record.status == ("annotated" if row.processed else "pending")
            if row.processed:
                assert record.mask_digest == row.mask_digest
                assert (ts.bundle_dir / record.mask_file).is_file()
                assert (ts.bundle_dir / record.input_file).is_file()

    def test_omero_ids_become_external_ids(self, tmp_path):
        ts, _, _, _ = _crate(tmp_path, limit=1)
        record = next(r for r in read_records(ts.bundle_dir) if r.status == "annotated")
        row = next(r for r in ts.rows if r.annotation_id == record.annotation_id)
        assert json.loads(record.external_ids)["omero_label_image"] == str(row.label_id)

    def test_excluded_unit_has_status_and_no_mask(self, tmp_path):
        rows = [Row(image_uri="omero:image:1", image_name="a", annotation_id="a_0")]
        rows[0].mark_excluded(session=1)
        (record,) = annotation_records(rows)
        assert (record.status, record.mask_file) == ("excluded", None)

    def test_tools_come_from_the_ai_model_when_not_described(self):
        config = create_default_config()
        (tool,) = annotation_set(config).methodology.tools
        assert (tool.name, tool.model.name) == ("micro-sam", "vit_b_lm")

    def test_written_documents_pass_the_schema(self, tmp_path):
        config = create_default_config()
        rows = [Row(image_uri="omero:image:1", image_name="a", annotation_id="a_0")]
        set_path, records_path = write_schema_documents(config, rows, tmp_path)
        assert read_set(set_path).schema_version == "0.1.0"
        assert read_records(records_path)[0].status == "pending"
