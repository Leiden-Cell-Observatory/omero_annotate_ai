# Training sets

A training set is created or opened as a `TrainingSet`; everything else (annotating,
adding images, notes) is a method on it.

::: omero_annotate_ai.trainingset.TrainingSet
    options:
      members:
        - create
        - open
        - config
        - table_id
        - bundle_dir
        - units
        - extend
        - annotate
        - start_session
        - set_session_note
        - reconcile
        - prepare_external
        - collect_masks

::: omero_annotate_ai.trainingset.list_training_sets

::: omero_annotate_ai.trainingset.summarize

## Annotation sessions

::: omero_annotate_ai.session.AnnotationSession

::: omero_annotate_ai.session.BatchReport

## Several computers (claims)

::: omero_annotate_ai.claims
    options:
      members: false

## Rows

::: omero_annotate_ai.rows.Row

## Deprecated one-call pipeline

::: omero_annotate_ai.core.annotation_pipeline.create_pipeline

::: omero_annotate_ai.core.annotation_pipeline.AnnotationPipeline
    options:
      members:
        - run_full_micro_sam_workflow
        - run_cp_workflow
        - collect_annotations_from_disk
