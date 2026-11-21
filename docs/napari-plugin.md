# napari plugin (experimental)

!!! warning "Experimental"
    The plugin works on OMERO images (and on sets in a local folder); OME-Zarr sets
    are made with the [notebooks](notebooks.md).

The plugin takes you from your OMERO data to annotated training images inside your own
napari window, with [micro-SAM](https://github.com/computational-cell-analytics/micro-sam):
find the data, make a training set (or open one), and annotate it one image at a time.

## Install

```bash
pip install "omero-annotate-ai[napari]" "napari[all]"
```

With pixi, the default environment of this repository already contains napari and
micro-SAM. Open napari and choose **Plugins → OMERO Annotate AI → Annotate training set**.

The `napari` extra pins micro-SAM to `>=1.8,<1.9`, because the plugin drives
micro-SAM's annotator directly; with another micro-SAM version the plugin stops with a
message naming the one it needs.

## Use

The dock is a small wizard in three steps; every call to OMERO runs in the background,
so napari stays responsive.

1. **Server.** Pick a *Saved login* and click *Log in*, or log in with host, user and
   password (or a session key). The password is never stored; with *Stay logged in*
   ticked a token is saved instead, so next time no password is needed. Sets that have
   a folder on this computer need no server: go to step 3.
2. **Data.** Find the images: *Group* switches to another of your groups, *Owner* shows
   your data (*Me*), everyone's (*All members*) or one colleague's, and *Type* is
   project, dataset, plate or screen. *Search* filters the list by name or ID as you
   type; *Go to ID* (or Enter) finds a container by its ID in any of your groups and
   switches to that group. Ctrl/Shift-click picks several containers of one type for one
   training set. Below the list: the number of images, a sample image's size and its
   channel names. *Next* goes to step 3.
3. **Training set.** The sets on the picked container and the sets with a folder on
   this computer are listed (*Open folder…* opens any other set folder); the summary
   shows progress per split and the last sessions with their notes. *Add images* adds
   images per split to the picked set before annotating. Tick *Review annotated
   images* to correct images annotated earlier instead (step 4b). *Model* and *Batch
   size* apply to the session; click *Annotate*.

    **New training set…** opens the form in its own, resizable window (the viewer
    stays usable). Left: *Name*; **Channels**: the channel you draw on (by name), the
    channels the model trains on (*Train on*), extra channels napari shows next to it
    while you annotate (*Show*; never used for training), the colour of each, the
    annotated channel included (gray by default), and the intensity normalisation.
    *Show sample image in viewer* adds each channel of a sample image as a layer, to
    see which channel is which. **Images**: timepoints and z-slices (specific, all or
    random), 3D and patches. Right: the number of images for training / validation /
    test, the seed (empty: random, stored), where the set is kept (OMERO, or this
    computer only), and for plates and screens **Plate wells**, which keeps or leaves
    out wells by their metadata: keys added with
    [MIHCSME](https://github.com/Leiden-Cell-Observatory/MIHCSME_py) (cell line,
    treatment, ...) come first, and the number of wells kept updates as you pick.
    *Create and annotate* makes the set and starts; a field that is not right is named
    in the window and nothing is made. After the session the new set is listed and
    selected.
4. **Annotate.** The set is opened (or created), and repaired first if an earlier
   run was interrupted (*Cancel* stops that). The image appears with micro-SAM's annotator docked
   next to it; the image embedding is computed in the background first (a progress
   bar shows while it runs); the next image's embedding is computed while you
   annotate, so moving on is quick. Meanwhile the next image is shown dimmed, without the
   previous image's objects, and micro-SAM's dock is disabled. Segment with micro-SAM's usual keys, commit the objects,
   then:
    - **Accept** (`N`) saves the committed objects for this image and moves on. An
      image without objects asks for a second press, so empty images are saved on
      purpose only.
    - **Skip** (`Shift-N`) leaves the image for a later session.
    - **Exclude** leaves the image out of the training set for good (out of focus,
      artefact): it counts as done, is never shown again and is not exported. Prev
      back to it and Accept to save it after all.
    - **Prev** goes back one image, through every image of this session: a saved
      image opens with its objects, and Accept saves your corrections (replacing the
      upload; an unchanged mask keeps it). Skip leaves it as it is.

    If an image cannot be shown (for example its embedding fails), the error is shown
    and **Skip** / **Prev** still work.

    **4b. Review.** With *Review annotated images* ticked, the annotated images come
    back one by one with their saved objects already committed (going back with
    **Prev** shows the saved mask again, not unaccepted edits). Correct them and
    press **Accept**: a changed mask replaces the label file and ROI on OMERO (the new
    ones are uploaded before the old ones are deleted); an image accepted unchanged
    keeps its upload. The progress line counts the revised images.
5. **Finish.** Add a short note about the session (for example *"nuclei only, skipped
   small debris"*) and click *Finish session*. The note is stored with the session.

Each accepted image is saved straight away (and, for OMERO sets, uploaded with its
ROIs, tagged with the set); the OMERO config is updated once, when you finish.
Closing the napari window (once you confirm), quitting napari or closing the plugin's
dock mid-session records the session as *interrupted* and releases its images; skipped
and unsaved images stay pending for the next session. Once you click *Finish session*,
no further image is shown or saved.

**Several computers.** While a batch is shown its images are claimed on the OMERO
table, so another computer annotating the same set skips them; the plugin says how
many images are being annotated elsewhere. Your own claims never block you, for
example after a dropped connection. An image another computer saved first is not
uploaded again: your mask is kept in the set folder and the plugin names the file.

## Known limits

- Annotating is 2D only: a 3D set can be made in the dock but is annotated with the
  notebooks (`ts.annotate()`).
- Images that need review (their local and uploaded masks differ) are skipped, and the
  plugin says how many; resolve them as described in
  [Resume and repeat](tutorials/resume-and-repeat.md).
- micro-SAM's own annotator widgets share one state per napari window. Do not open
  micro-SAM's annotators from the Plugins menu while this plugin is annotating.
- The plugin uses the layer names `image` and `committed_objects`; it refuses to start
  if your viewer already has layers with those names, before anything is claimed.
  Finishing a session removes the layers the plugin and micro-SAM added (your masks
  are saved by then); layers you added, also during the session, stay.
