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
        - review
        - start_session
        - set_session_note
        - reconcile
        - prepare_external
        - collect_masks
        - write_ro_crate

::: omero_annotate_ai.trainingset.list_training_sets

::: omero_annotate_ai.trainingset.summarize

::: omero_annotate_ai.trainingset.training_set_choices

::: omero_annotate_ai.trainingset.read_set_config

## Annotation sessions

::: omero_annotate_ai.session.AnnotationSession

::: omero_annotate_ai.session.BatchReport

## Content codes (provenance extra)

::: omero_annotate_ai.provenance
    options:
      members:
        - available
        - image_code
        - file_code
        - distance
        - describe_change

## RO-Crate

::: omero_annotate_ai.crate
    options:
      members:
        - build_ro_crate
        - write_ro_crate

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
