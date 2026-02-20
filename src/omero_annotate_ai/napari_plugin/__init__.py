"""Napari plugin entry point for OMERO Annotate AI."""

import napari


def get_widget(napari_viewer: napari.Viewer):  # noqa: ANN201
    """Return the OMERO Annotate AI dock widget (npe2 entry point)."""
    from ._widget import OMEROAnnotateWidget

    return OMEROAnnotateWidget(napari_viewer)
