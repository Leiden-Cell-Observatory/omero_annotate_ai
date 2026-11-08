# Widgets

Jupyter (ipywidgets) interfaces. They do not work in marimo; the marimo notebooks build
the same choices from `mo.ui` elements on the same core functions.

::: omero_annotate_ai.widgets.omero_connection_widget.create_omero_connection_widget

::: omero_annotate_ai.widgets.omero_connection_widget.OMEROConnectionWidget
    options:
      members: [display, get_connection]

::: omero_annotate_ai.widgets.workflow_widget.create_workflow_widget

::: omero_annotate_ai.widgets.workflow_widget.WorkflowWidget
    options:
      members: [display, get_config, request, training_set, annotate, events, set_connection]

::: omero_annotate_ai.widgets.workflow_widget.open_or_create

::: omero_annotate_ai.widgets.workflow_widget.NewTrainingSet

::: omero_annotate_ai.widgets.workflow_widget.OpenTrainingSet

::: omero_annotate_ai.widgets.training_data_widget.create_training_data_widget

::: omero_annotate_ai.widgets.training_data_widget.TrainingDataWidget
    options:
      members: [display, selected, get_selected_table_id, selected_config, export, set_connection]
