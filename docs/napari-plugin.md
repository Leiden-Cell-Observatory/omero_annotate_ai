# napari plugin (experimental)

!!! warning "Experimental"
    The plugin opens and annotates existing training sets (on OMERO or in a local
    folder). Create a new training set with the [notebooks](notebooks.md) or
    `TrainingSet.create` first; the plugin cannot create sets yet.

The plugin lets you annotate a training set with [micro-SAM](https://github.com/computational-cell-analytics/micro-sam)
inside your own napari window, one image at a time, without leaving napari.

## Install

```bash
pip install "omero-annotate-ai[napari]" "napari[all]"
```

With pixi, the default environment of this repository already contains napari and
micro-SAM. Open napari and choose **Plugins → OMERO Annotate AI → Annotate training set**.

The `napari` extra pins micro-SAM to `>=1.8,<1.9`, because the plugin drives
micro-SAM's annotator directly.

## Use

1. **Pick a training set.** Sets that have a folder in your home folder
   (`~/omero_annotate_ai`, or `$OMERO_ANNOTATE_AI_HOME`) are listed straight away;
   *Open folder…* opens any other set folder. For sets on OMERO, **connect** first
   (host, user, password; tick *Remember password in keychain* to store it), choose
   the container type and ID, and click *List sets*. Connecting runs in the
   background, so napari stays responsive. The summary shows progress per split and
   the last sessions with their notes; listing never changes anything.
2. **Annotate.** Click *Annotate*. The set is opened, and repaired first if an earlier
   run was interrupted (*Cancel* stops that). The image appears with micro-SAM's annotator docked
   next to it; the image embedding is computed in the background first (a progress
   bar shows while it runs). Segment with micro-SAM's usual keys, commit the objects,
   then:
    - **Accept** (`N`) saves the committed objects for this image and moves on. An
      image without objects asks for a second press, so empty images are saved on
      purpose only.
    - **Skip** (`Shift-N`) leaves the image for a later session.
    - **Prev** goes back to an earlier image of the current batch that is not saved yet.
3. **Finish.** Add a short note about the session (for example *"nuclei only, skipped
   small debris"*) and click *Finish session*. The note is stored with the session.

Each accepted image is saved straight away (and, for OMERO sets, uploaded with its
ROIs, tagged with the set); the OMERO config is updated once, when you finish.
Quitting napari mid-session records the session as *interrupted*; skipped and
unsaved images stay pending for the next session.

## Known limits

- 2D only: the plugin refuses 3D sets. Creating a set and adding images to a set come
  later (use `TrainingSet.create` / `extend`).
- Images that need review (their local and uploaded masks differ) are skipped; resolve
  them as described in [Resume and repeat](tutorials/resume-and-repeat.md).
- micro-SAM's own annotator widgets share one state per napari window. Do not open
  micro-SAM's annotators from the Plugins menu while this plugin is annotating.
- The plugin uses the layer names `image` and `committed_objects`; it refuses to start
  if your viewer already has layers with those names.
