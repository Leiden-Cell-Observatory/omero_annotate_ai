"""The tracking table on OMERO, written once and then updated in place.

A training set is identified by its tracking table's ID, so the table must keep
that ID for its whole life. This module never deletes and re-creates a table:
changed rows are written with ``Table.update`` and new rows with
``Table.addData``. Both keep the same OriginalFile and FileAnnotation.

OMERO.tables fixes the width of a string column when the table is created and
rejects longer values later (``ValidationException``). Tables are therefore
created with generous widths per column, and every write checks lengths first
so the user gets a clear error instead of a server exception.

Tables carry the marker :data:`TABLE_MARKER` in their FileAnnotation
description. Tables without it (written by versions before 0.5.0) are refused:
old tables are not supported.
"""

from collections.abc import Iterable

import pandas as pd

from ..errors import TableValueTooLongError, UnsupportedTableError
from ._untyped import NSBULKANNOTATIONS, omero_grid, omero_model, omero_rtypes

rstring = omero_rtypes.rstring


TABLE_MARKER = "omero-annotate-ai table v2"

# Width of each string column. Anything not listed gets DEFAULT_STRING_WIDTH.
STRING_WIDTHS: dict[str, int] = {
    "image_uri": 1024,
    "image_name": 512,
    "annotation_id": 256,
    "category": 16,
    "annotation_type": 64,
    "annotation_created_at": 64,
    "annotation_updated_at": 64,
    "schema_attachment_id": 32,
    "well": 32,
    "field": 64,
    "well_metadata": 4096,
}
DEFAULT_STRING_WIDTH = 256

# DataFrame column -> (OMERO column name, column class, description).
#
# The names are not cosmetic. OMERO.web resolves a row to an object by name
# (a column literally called Image or Roi), and the column type makes the cell
# a link to the image or ROI.
LINK_COLUMNS = {
    "image_id": ("Image", "ImageColumn", "Annotated OMERO image"),
    "roi_id": ("Roi", "RoiColumn", "ROI holding the annotation"),
}
_LINK_COLUMNS_REVERSED = {
    omero_name: df_name for df_name, (omero_name, _, _) in LINK_COLUMNS.items()
}

KEY_COLUMN = "annotation_id"


def _require_omero():
    if omero_grid is None:
        raise ImportError("omero-py and ezomero are required for OMERO tracking tables")


def tag_for(table_id: int, annotation_id: str) -> str:
    """Tag written on every ROI and label file a training set uploads."""
    return f"omero-annotate-ai:{int(table_id)}:{annotation_id}"


def tag_prefix(table_id: int) -> str:
    """Prefix shared by every tag of one training set."""
    return f"omero-annotate-ai:{int(table_id)}:"


def config_file_name(table_id: int) -> str:
    """File name of the one config YAML a training set keeps on OMERO."""
    return f"annotation_config_table_{int(table_id)}.yaml"


# -----------------------------------------------------------------------------
# Columns
# -----------------------------------------------------------------------------


def _df_name(omero_name: str) -> str:
    return _LINK_COLUMNS_REVERSED.get(omero_name, omero_name)


def _string_width(name: str) -> int:
    return STRING_WIDTHS.get(name, DEFAULT_STRING_WIDTH)


def check_string_lengths(
    df: pd.DataFrame, widths: dict[str, int] | None = None
) -> None:
    """Raise TableValueTooLongError if any string is wider than its column."""
    for name in df.columns:
        series = df[name]
        if name in LINK_COLUMNS or not (
            pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)
        ):
            continue
        width = (widths or {}).get(name, _string_width(name))
        for position, value in enumerate(series):
            text = "" if value is None else str(value)
            if len(text) > width:
                key = (
                    df[KEY_COLUMN].iloc[position]
                    if KEY_COLUMN in df.columns
                    else position
                )
                raise TableValueTooLongError(
                    f"Column '{name}' allows {width} characters, but row {key!r} has "
                    f"{len(text)}: {text[:60]}..."
                )


def build_columns(df: pd.DataFrame) -> list:
    """OMERO columns for a prepared DataFrame, with fixed string widths."""
    _require_omero()
    columns = []
    for name in df.columns:
        series = df[name]
        link = LINK_COLUMNS.get(name)
        if link is not None:
            omero_name, class_name, description = link
            column_class = getattr(omero_grid, class_name)
            columns.append(
                column_class(omero_name, description, [int(v) for v in series])
            )
        elif pd.api.types.is_bool_dtype(series):
            columns.append(omero_grid.BoolColumn(name, "", [bool(v) for v in series]))
        elif pd.api.types.is_integer_dtype(series):
            columns.append(omero_grid.LongColumn(name, "", [int(v) for v in series]))
        elif pd.api.types.is_float_dtype(series):
            columns.append(
                omero_grid.DoubleColumn(name, "", [float(v) for v in series])
            )
        else:
            values = ["" if v is None else str(v) for v in series]
            columns.append(
                omero_grid.StringColumn(name, "", _string_width(name), values)
            )
    return columns


def _values_for(header, series: pd.Series) -> list:
    """Values of a DataFrame column converted to the type of a table column."""
    if isinstance(header, (omero_grid.BoolColumn,)):
        return [bool(v) for v in series]
    if isinstance(
        header, (omero_grid.LongColumn, omero_grid.ImageColumn, omero_grid.RoiColumn)
    ):
        return [int(v) for v in series]
    if isinstance(header, omero_grid.DoubleColumn):
        return [float(v) for v in series]
    return ["" if v is None else str(v) for v in series]


def _header_widths(headers) -> dict[str, int]:
    return {
        h.name: h.size
        for h in headers
        if isinstance(h, omero_grid.StringColumn) and getattr(h, "size", None)
    }


def _fill_columns(headers, df: pd.DataFrame) -> list:
    """Table headers holding the DataFrame's values, in the table's column order."""
    missing = [h.name for h in headers if _df_name(h.name) not in df.columns]
    if missing:
        raise ValueError(f"Rows lack table columns: {', '.join(missing)}")
    for header in headers:
        header.values = _values_for(header, df.loc[:, _df_name(header.name)])
    return headers


# -----------------------------------------------------------------------------
# Opening a table
# -----------------------------------------------------------------------------


def _file_annotation(conn, table_id: int):
    file_ann = conn.getObject("FileAnnotation", int(table_id))
    if file_ann is None:
        raise ValueError(f"Tracking table {table_id} not found (or not visible to you)")
    return file_ann


def is_supported(file_ann) -> bool:
    """True if a table FileAnnotation carries the current marker."""
    description = file_ann.getDescription() or ""
    return TABLE_MARKER in description


def check_supported(conn, table_id: int):
    """Return the table's FileAnnotation, or raise UnsupportedTableError."""
    file_ann = _file_annotation(conn, table_id)
    if not is_supported(file_ann):
        raise UnsupportedTableError(
            f"Table {table_id} was not written by omero-annotate-ai 0.5 or later. "
            "Older tracking tables are not supported; start a new training set."
        )
    return file_ann


def _open(conn, table_id: int):
    file_ann = check_supported(conn, table_id)
    resources = conn.c.sf.sharedResources()
    file_id = file_ann.getFile().getId()
    table = resources.openTable(
        omero_model.OriginalFileI(int(file_id), False), conn.SERVICE_OPTS
    )
    if table is None:
        raise ValueError(f"Could not open tracking table {table_id}")
    return table


# -----------------------------------------------------------------------------
# Create, read, write
# -----------------------------------------------------------------------------


def create_table(
    conn,
    df: pd.DataFrame,
    title: str,
    container_type: str,
    container_ids: Iterable[int],
) -> int:
    """Create a tracking table and link it to its containers.

    Args:
        conn: OMERO connection
        df: Rows, already prepared for OMERO (see ``prepare_rows``)
        title: Table name shown in OMERO
        container_type: 'dataset', 'plate', 'project' or 'screen'
        container_ids: Containers to link the table to (at least one)

    Returns:
        The table's FileAnnotation ID, which identifies the training set.
    """
    _require_omero()
    from .omero_functions import link_table_to_containers

    container_ids = [int(c) for c in container_ids if c]
    if not container_ids:
        raise ValueError("A tracking table needs at least one container")
    if KEY_COLUMN not in df.columns:
        raise ValueError(f"Tracking table rows need an '{KEY_COLUMN}' column")
    check_string_lengths(df)

    columns = build_columns(df)
    resources = conn.c.sf.sharedResources()
    repository_id = resources.repositories().descriptions[0].getId().getValue()
    table = resources.newTable(repository_id, title, conn.SERVICE_OPTS)
    try:
        table.initialize(columns)
        if len(df):
            table.addData(columns)
        original_file = table.getOriginalFile()
    finally:
        table.close()

    file_ann = omero_model.FileAnnotationI()
    file_ann.setNs(rstring(NSBULKANNOTATIONS))
    file_ann.setDescription(rstring(TABLE_MARKER))
    # Reference the file unloaded: the table service already wrote it, and
    # saving the loaded copy again raises an OptimisticLockException.
    file_ann.setFile(omero_model.OriginalFileI(original_file.getId().getValue(), False))
    file_ann = conn.getUpdateService().saveAndReturnObject(file_ann, conn.SERVICE_OPTS)
    table_id = file_ann.getId().getValue()

    link_table_to_containers(conn, table_id, container_type, container_ids)
    return table_id


def read_table(conn, table_id: int) -> pd.DataFrame:
    """All rows of a tracking table, with link columns named image_id / roi_id."""
    table = _open(conn, table_id)
    try:
        headers = table.getHeaders()
        n_rows = table.getNumberOfRows()
        if n_rows == 0:
            return pd.DataFrame(columns=[_df_name(h.name) for h in headers])
        data = table.read(list(range(len(headers))), 0, n_rows)
        return pd.DataFrame({_df_name(c.name): list(c.values) for c in data.columns})
    finally:
        table.close()


def write_rows(conn, table_id: int, df: pd.DataFrame) -> dict[str, int]:
    """Update rows that exist (by annotation_id) and append the others.

    The table keeps its ID. Rows in the table that are not in ``df`` are left
    as they are.

    Args:
        conn: OMERO connection
        table_id: Tracking table ID
        df: Rows, already prepared for OMERO (see ``prepare_rows``)

    Returns:
        ``{"updated": n, "added": m}``
    """
    if df is None or len(df) == 0:
        return {"updated": 0, "added": 0}
    if df[KEY_COLUMN].duplicated().any():
        raise ValueError("Duplicate annotation_id values in the rows to write")

    table = _open(conn, table_id)
    try:
        headers = table.getHeaders()
        check_string_lengths(df, _header_widths(headers))
        key_index = [h.name for h in headers].index(KEY_COLUMN)
        n_rows = table.getNumberOfRows()
        existing: dict[str, int] = {}
        if n_rows:
            keys = table.read([key_index], 0, n_rows).columns[0].values
            existing = {str(k): i for i, k in enumerate(keys)}

        keys = df[KEY_COLUMN].astype(str)
        to_update = df.loc[keys.isin(existing)]
        to_add = df.loc[~keys.isin(existing)]

        if len(to_update):
            row_numbers = [existing[str(k)] for k in to_update[KEY_COLUMN]]
            data = table.readCoordinates(row_numbers)
            # readCoordinates returns the rows in the order asked for.
            for column in data.columns:
                column.values = _values_for(
                    column, to_update.loc[:, _df_name(column.name)]
                )
            table.update(data)
        if len(to_add):
            table.addData(_fill_columns(table.getHeaders(), to_add))
        return {"updated": len(to_update), "added": len(to_add)}
    finally:
        table.close()
