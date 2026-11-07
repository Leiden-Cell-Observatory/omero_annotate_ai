# API reference

Most people only need the functions below, all importable from the top-level package:

```python
from omero_annotate_ai import (
    load_config_from_yaml,
    create_default_config,  # configuration
    create_pipeline,  # run annotation
    create_omero_connection_widget,  # Jupyter widgets
    create_workflow_widget,
    create_training_data_widget,
    prepare_training_data_from_table,  # training data
    reorganize_local_data_for_training,
    setup_training,
    run_training,  # micro-SAM training
    prepare_bia_data_from_table,  # BioImage Archive export
)
```

| Page | Contents |
|---|---|
| [Configuration](config.md) | `AnnotationConfig` and its sections, loading and saving YAML |
| [Pipeline](pipeline.md) | `create_pipeline`, `AnnotationPipeline` and its workflow methods |
| [Widgets](widgets.md) | Jupyter widgets for connecting, configuring and selecting tables |
| [Training data](training.md) | Preparing training folders, micro-SAM training, BioImage Archive export |
| [OMERO helpers](omero.md) | Connections, tracking tables, downloading configs |

This reference is generated from the docstrings in the source code.
