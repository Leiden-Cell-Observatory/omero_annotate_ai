"""Test data for the Docker OMERO integration tests.

Creates a uniquely named project/dataset with small synthetic images (through the
API) and a small plate (through ``omero import`` of a Bio-Formats ``.fake`` file
inside the server container), and removes them again afterwards.
"""

from __future__ import annotations

import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

COMPOSE_FILE = Path(__file__).parent / "docker-compose.yml"
COMPOSE_PROJECT = "oai-test"
OMERO_CLI = "/opt/omero/server/OMERO.server/bin/omero"

IMAGE_SHAPE = {"size_t": 2, "size_c": 2, "size_z": 3, "size_y": 64, "size_x": 80}
CHANNEL_NAMES = ["DAPI", "GFP"]


def compose_cmd(*args: str) -> list[str]:
    """``docker compose`` command for the test stack."""
    return ["docker", "compose", "-p", COMPOSE_PROJECT, "-f", str(COMPOSE_FILE), *args]


def synthetic_plane(z: int, c: int, t: int, index: int) -> np.ndarray:
    """A deterministic plane, so tests can check pixel values they read back."""
    y, x = IMAGE_SHAPE["size_y"], IMAGE_SHAPE["size_x"]
    base = 1000 * index + 100 * z + 10 * c + t
    return (np.arange(y * x, dtype=np.uint16).reshape(y, x) % 50) + np.uint16(base)


@dataclass
class SeedData:
    """IDs of the objects created for one test session."""

    tag: str
    project_id: int
    dataset_id: int
    image_ids: list[int]
    plate_id: int | None = None
    plate_image_ids: list[int] = field(default_factory=list)


def _create_images(conn, dataset, tag: str, n_images: int) -> list[int]:
    sz, sc, st = IMAGE_SHAPE["size_z"], IMAGE_SHAPE["size_c"], IMAGE_SHAPE["size_t"]
    ids = []
    for index in range(n_images):
        # createImageFromNumpySeq iterates T fastest, then C, then Z.
        planes = (
            synthetic_plane(z, c, t, index)
            for z in range(sz)
            for c in range(sc)
            for t in range(st)
        )
        image = conn.createImageFromNumpySeq(
            planes,
            f"{tag}_img{index}",
            sizeZ=sz,
            sizeC=sc,
            sizeT=st,
            dataset=dataset,
            channelList=list(range(sc)),
        )
        for channel, name in zip(image.getChannels(), CHANNEL_NAMES, strict=True):
            logical = channel.getLogicalChannel()
            logical.setName(name)
            logical.save()
        ids.append(image.getId())
    return ids


def _import_plate(conn, tag: str, user: str, password: str) -> int | None:
    """Import a 2x2 fake plate inside the server container; None if that is impossible."""
    from omero.rtypes import rstring

    fake = f"{tag}&plates=1&plateRows=2&plateCols=2&fields=1&sizeC=2&sizeX=64&sizeY=64.fake"
    script = (
        f'cd /tmp && touch "{fake}" && '
        f'{OMERO_CLI} -s localhost -u {user} -w {password} import --output=ids "{fake}"; '
        f'rc=$?; rm -f "{fake}"; exit $rc'
    )
    try:
        result = subprocess.run(
            compose_cmd("exec", "-T", "omero", "sh", "-c", script),
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in result.stdout.splitlines():
        if line.startswith("Plate:"):
            plate_id = int(line.split(":", 1)[1].split(",")[0])
            plate = conn.getObject("Plate", plate_id)
            plate.setName(rstring(f"{tag}_plate"))
            plate.save()
            return plate_id
    return None


def seed(conn, user: str, password: str, n_images: int = 6) -> SeedData:
    """Create the session's project, dataset, images and plate."""
    import omero.model
    from omero.gateway import DatasetWrapper, ProjectWrapper
    from omero.rtypes import rstring

    tag = f"oai-test-{uuid.uuid4().hex[:8]}"
    project = ProjectWrapper(conn, omero.model.ProjectI())
    project.setName(rstring(f"{tag}_project"))
    project.save()
    dataset = DatasetWrapper(conn, omero.model.DatasetI())
    dataset.setName(rstring(f"{tag}_dataset"))
    dataset.save()
    link = omero.model.ProjectDatasetLinkI()
    link.setParent(omero.model.ProjectI(project.getId(), False))
    link.setChild(omero.model.DatasetI(dataset.getId(), False))
    conn.getUpdateService().saveObject(link)

    image_ids = _create_images(conn, dataset, tag, n_images)
    data = SeedData(tag, project.getId(), dataset.getId(), image_ids)

    data.plate_id = _import_plate(conn, tag, user, password)
    if data.plate_id is not None:
        plate = conn.getObject("Plate", data.plate_id)
        data.plate_image_ids = [
            well.getImage(0).getId() for well in plate.listChildren()
        ]
    return data


def cleanup(conn, data: SeedData) -> None:
    """Delete everything :func:`seed` created (and anything attached to it)."""
    deletions = [("Project", [data.project_id])]
    if data.plate_id is not None:
        deletions.append(("Plate", [data.plate_id]))
    for object_type, ids in deletions:
        try:
            conn.deleteObjects(
                object_type, ids, deleteAnns=True, deleteChildren=True, wait=True
            )
        except Exception as exc:  # cleanup must not hide test results
            print(f"Cleanup of {object_type} {ids} failed: {exc}")
