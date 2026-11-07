# Pipeline

::: omero_annotate_ai.core.annotation_pipeline.create_pipeline

::: omero_annotate_ai.core.annotation_pipeline.AnnotationPipeline
    options:
      members:
        - run_full_micro_sam_workflow
        - run_cp_workflow
        - collect_annotations_from_disk
        - get_annotation_status_from_disk
        - prepare_training_data_from_local
        - run_custom_annotation
        - initialize_workflow
        - define_annotation_schema
        - create_tracking_table
        - run_micro_sam_annotation
        - run_cellpose_preparation
        - set_table_id
