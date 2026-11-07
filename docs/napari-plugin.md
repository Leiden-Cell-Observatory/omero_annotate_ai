# napari plugin (experimental)

!!! warning "Experimental"
    The plugin works today only with a built-in **demo training set** of synthetic
    images. Opening training sets on OMERO or from a local folder arrives with the
    0.5.0 core (the step-by-step training-set API). Until then, create and annotate
    real training sets with the [notebooks](notebooks.md).

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

1. **Connect** (optional for the demo): host, user and password. Tick *Remember
   password in keychain* to store it in your system keychain. The connection runs in
   the background, so napari stays responsive.
2. **Pick a training set.** *Open demo set* adds a small set of synthetic images. The
   summary shows progress per split and the last sessions with their notes.
3. **Annotate.** Click *Annotate*. The image appears with micro-SAM's annotator docked
   next to it; the image embedding is computed in the background first (a progress
   bar shows while it runs). Segment with micro-SAM's usual keys, commit the objects,
   then:
    - **Accept** (`N`) saves the committed objects for this image and moves on. An
      image without objects asks for a second press, so empty images are saved on
      purpose only.
    - **Skip** (`Shift-N`) leaves the image for a later session.
    - **Prev** goes back to an earlier image of the current batch that is not saved yet.
4. **Finish.** Add a short note about the session (for example *"nuclei only, skipped
   small debris"*) and click *Finish session*. The note is stored with the session.

Each accepted image is saved straight away; closing napari loses at most the image you
are working on.

## Known limits

- 2D only. 3D, creating a new training set and adding images to a set come later.
- micro-SAM's own annotator widgets share one state per napari window. Do not open
  micro-SAM's annotators from the Plugins menu while this plugin is annotating.
- The plugin uses the layer names `image` and `committed_objects`; it refuses to start
  if your viewer already has layers with those names.
