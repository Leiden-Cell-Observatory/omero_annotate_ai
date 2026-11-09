package qupath.ext.omeroannotate.core;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * One sitting at the microscope images: hands out units one by one and saves what comes
 * back.
 * <p>
 * The write order of a unit is fixed so that a crash never loses or duplicates work:
 * the mask goes into the set's folder, the local table is updated, then (OMERO sets)
 * the label file and the ROI are uploaded, each tagged with the set and unit, and last
 * the unit's row in the OMERO table is written. In a review session the annotated units
 * come back; a changed mask replaces the earlier upload (new one first, old one deleted
 * after), an unchanged one is left alone.
 */
public final class AnnotationSession {

    public enum Outcome {
        /** The unit is saved (and uploaded, for OMERO sets). */
        SAVED,
        /** A saved unit was accepted again with the same mask: nothing was changed. */
        UNCHANGED,
        /** A saved unit got a new mask, which replaced the earlier one. */
        REVISED,
        /** The unit is excluded from the training set. */
        EXCLUDED,
        /** Another computer saved this unit first; this mask was kept aside as a file. */
        SAVED_ELSEWHERE,
        /** The mask is safe in the set's folder, but OMERO could not be updated. */
        UPLOAD_FAILED
    }

    public record Result(Outcome outcome, String message) {
        public boolean ok() {
            return outcome != Outcome.UPLOAD_FAILED && outcome != Outcome.SAVED_ELSEWHERE;
        }
    }

    private final TrainingSet set;
    private final boolean review;
    private final TrainingSet.MaskTracer tracer;
    private final int id;
    private final String token;
    private final Set<String> handedOut = new LinkedHashSet<>();
    private final Set<String> saved = new LinkedHashSet<>();
    private boolean finished = false;

    /**
     * Start a session.
     *
     * @param review hand out annotated units (to correct them) instead of open ones
     * @param model  what the annotator used, recorded with the session (e.g. "QuPath + Cellpose")
     */
    public AnnotationSession(TrainingSet set, boolean review, String model, TrainingSet.MaskTracer tracer)
            throws Exception {
        this.set = set;
        this.review = review;
        this.tracer = tracer;
        this.id = set.beginSession(review ? "review" : "annotate", model);
        this.token = set.token(id);
    }

    public int id() {
        return id;
    }

    public boolean isReview() {
        return review;
    }

    public TrainingSet set() {
        return set;
    }

    /** Units saved in this session (annotated, or changed in a review). */
    public int savedCount() {
        return saved.size();
    }

    public boolean isFinished() {
        return finished;
    }

    private boolean eligible(Row row) {
        if (row.isVolumetric || row.level != 0)
            return false;
        if (row.error != null && row.error.startsWith(TrainingSet.NEEDS_REVIEW))
            return false;
        if (review ? !row.isAnnotated() : row.processed)
            return false;
        if (!review && row.error != null && Files.exists(Bundle.maskPath(set.bundle(), row.annotationId)))
            return false; // waiting for its upload to be retried, not for a new annotation
        return !TrainingSet.heldByOther(row.claimedBy, row.claimedAt, token, set.config().claimHours());
    }

    /** How many units are left for this session. */
    public int remaining() {
        int n = 0;
        for (Row row : set.rows())
            if (eligible(row) && !handedOut.contains(row.annotationId))
                n++;
        return n;
    }

    /** How many open units other computers are annotating right now (they are skipped). */
    public int heldElsewhere() {
        int n = 0;
        for (Row row : set.rows())
            if (!row.processed && TrainingSet.heldByOther(row.claimedBy, row.claimedAt, token,
                    set.config().claimHours()))
                n++;
        return n;
    }

    /**
     * The next unit to annotate, claimed for this session, or null when none is left.
     * Each unit is handed out once per session.
     */
    public Row next() throws Exception {
        checkOpen();
        set.refresh();
        for (Row row : set.rows()) {
            if (!eligible(row) || handedOut.contains(row.annotationId))
                continue;
            if (set.claim(row, token, review)) {
                handedOut.add(row.annotationId);
                set.saveLocalTable();
                return row;
            }
        }
        return null;
    }

    /**
     * Save a unit's mask.
     *
     * @param mask  the label image of the unit: 0 is background, each object one number
     * @param input the pixels the annotator saw (the annotated channel), or null to keep
     *              an input image saved earlier
     */
    public Result commit(Row row, PixelArray mask, PixelArray input) throws Exception {
        checkOpen();
        Path bundle = set.bundle();
        if (input != null) {
            TiffIO.write(Bundle.inputPath(bundle, row.annotationId), input, false);
            row.inputDigest = input.digest();
            row.inputIscc = null; // belonged to the earlier pixels, if any
        }
        if (row.isAnnotated())
            return revise(row, mask);

        RemoteStore remote = set.remote();
        String unreachable = null;
        if (remote != null) {
            List<Map<String, Object>> now = List.of();
            try {
                now = remote.readRows(set.tableId(), List.of(row.annotationId));
            } catch (Exception e) {
                // OMERO cannot be asked: the work is saved here first and uploaded later.
                unreachable = e.getClass().getSimpleName() + ": " + e.getMessage();
            }
            if (!now.isEmpty()) {
                Row theirs = RowCodec.fromCells(now.get(0));
                if (theirs.processed && !(theirs.isExcluded() && row.isExcluded())) {
                    Path kept = Bundle.conflictPath(bundle, row.annotationId, id);
                    TiffIO.write(kept, mask, false);
                    if (theirs.inputDigest == null)
                        theirs.inputDigest = row.inputDigest;
                    row.copyFrom(theirs);
                    set.saveLocalTable();
                    return new Result(Outcome.SAVED_ELSEWHERE, "Another computer saved this image first. "
                            + "Your mask was kept as " + kept);
                }
            }
        }

        String digest = mask.digest();
        if (remote != null && unreachable == null && (row.labelId != null || row.roiId != null)
                && !digest.equals(row.maskDigest)) {
            // An earlier attempt got part of its upload through, and the mask has changed
            // since: those uploads belong to the old mask and must not be kept for this one.
            for (Object[] part : new Object[][]{{"FileAnnotation", row.labelId}, {"Roi", row.roiId}}) {
                if (part[1] == null)
                    continue;
                try {
                    remote.delete((String) part[0], List.of((Long) part[1]));
                } catch (Exception e) {
                    unreachable = e.getClass().getSimpleName() + ": " + e.getMessage();
                }
            }
            if (unreachable == null) {
                row.labelId = null;
                row.roiId = null;
            }
        }
        TiffIO.write(Bundle.maskPath(bundle, row.annotationId), mask, false);
        row.maskDigest = digest;
        row.maskIscc = null;
        row.session = id;
        row.error = null;
        row.claimedBy = "";
        row.claimedAt = "";
        row.markProcessed(set.config().annotationType());
        set.saveLocalTable();
        if (remote == null) {
            saved.add(row.annotationId);
            return new Result(Outcome.SAVED, null);
        }
        if (unreachable != null) {
            row.processed = false;
            row.error = unreachable;
            set.saveLocalTable();
            return new Result(Outcome.UPLOAD_FAILED, unreachable);
        }
        if (!set.uploadUnit(row, mask, tracer)) {
            releaseQuietly(row);
            return new Result(Outcome.UPLOAD_FAILED, row.error);
        }
        saved.add(row.annotationId);
        if (!set.writeTable(List.of(row)))
            return new Result(Outcome.UPLOAD_FAILED, "The mask and ROIs are on OMERO, but the tracking table "
                    + "could not be updated. Use \"Upload now\" on the set's overview to complete it.");
        return new Result(Outcome.SAVED, null);
    }

    /** Replace the mask of a saved unit, if it changed. */
    private Result revise(Row row, PixelArray mask) throws Exception {
        Path target = Bundle.maskPath(set.bundle(), row.annotationId);
        String digest = mask.digest();
        if (digest.equals(row.maskDigest) || sameObjects(mask, target))
            return new Result(Outcome.UNCHANGED, null);
        RemoteStore remote = set.remote();
        if (remote != null) {
            // Somebody else may have corrected this unit since it was shown here (a claim
            // lasts hours, not for ever): their newer upload must not be replaced blindly.
            Row theirs = null;
            String unreachable = null;
            try {
                List<Map<String, Object>> now = remote.readRows(set.tableId(), List.of(row.annotationId));
                if (!now.isEmpty())
                    theirs = RowCodec.fromCells(now.get(0));
            } catch (Exception e) {
                unreachable = e.getClass().getSimpleName() + ": " + e.getMessage();
            }
            boolean changedElsewhere = theirs != null && theirs.processed && theirs.labelId != null
                    && row.labelId != null && theirs.labelId > row.labelId;
            if (unreachable != null || changedElsewhere) {
                Path kept = Bundle.conflictPath(set.bundle(), row.annotationId, id);
                TiffIO.write(kept, mask, false);
                if (changedElsewhere) {
                    if (theirs.inputDigest == null)
                        theirs.inputDigest = row.inputDigest;
                    row.copyFrom(theirs);
                    set.saveLocalTable();
                    return new Result(Outcome.SAVED_ELSEWHERE, "Somebody else corrected this image in the "
                            + "meantime. Their version is kept; your correction was saved as " + kept);
                }
                return new Result(Outcome.UPLOAD_FAILED, "OMERO could not be asked (" + unreachable
                        + "). The earlier mask is kept; your correction was saved as " + kept);
            }
        }
        Path previous = target.resolveSibling(row.annotationId + "_mask.previous.tif");
        if (Files.exists(target))
            Files.move(target, previous, StandardCopyOption.REPLACE_EXISTING);
        TiffIO.write(target, mask, false);
        Row before = row.copy();
        row.maskDigest = digest;
        row.maskIscc = null;
        row.session = id;
        row.error = null;
        row.claimedBy = "";
        row.claimedAt = "";
        row.markProcessed(set.config().annotationType());
        set.saveLocalTable();
        if (remote == null) {
            Files.deleteIfExists(previous);
            saved.add(row.annotationId);
            return new Result(Outcome.REVISED, null);
        }
        try { // new uploads first: a crash leaves the old ones, never none
            String description = set.uploadDescription(row);
            row.labelId = remote.uploadLabel(row.imageId, TiffIO.encode(mask, true), set.labelFileName(row), description);
            set.saveLocalTable();
            row.roiId = remote.uploadRois(row.imageId,
                    tracer.trace(mask, row.isPatch ? row.patchX : 0, row.isPatch ? row.patchY : 0),
                    row.zSlice, row.timepoint, row.channel, set.name() + "_ROIs", description);
            set.saveLocalTable();
        } catch (Exception e) {
            // The unit stays as it was: old mask, old uploads. The correction is kept aside.
            if (row.labelId != null && !row.labelId.equals(before.labelId)) {
                try {
                    remote.delete("FileAnnotation", List.of(row.labelId));
                } catch (Exception ignored) {
                    // a leftover label carries the unit's tag and is harmless
                }
            }
            Path kept = Bundle.conflictPath(set.bundle(), row.annotationId, id);
            Files.createDirectories(kept.getParent());
            Files.move(target, kept, StandardCopyOption.REPLACE_EXISTING);
            if (Files.exists(previous))
                Files.move(previous, target, StandardCopyOption.REPLACE_EXISTING);
            String error = e.getClass().getSimpleName() + ": " + e.getMessage();
            row.copyFrom(before);
            row.error = error;
            set.saveLocalTable();
            return new Result(Outcome.UPLOAD_FAILED, "OMERO could not be updated (" + error
                    + "). The earlier mask is kept; your correction was saved as " + kept);
        }
        Files.deleteIfExists(previous);
        saved.add(row.annotationId);
        // The table first, the old uploads after: while the table still names the old
        // label it must exist. A failed table write is finished by repair(), which then
        // removes the superseded uploads.
        if (!set.writeTable(List.of(row)))
            return new Result(Outcome.UPLOAD_FAILED, "The new mask is on OMERO, but the tracking table could "
                    + "not be updated. Use \"Upload now\" on the set's overview to complete it.");
        set.deleteSuperseded(before.labelId, before.roiId, row);
        return new Result(Outcome.REVISED, null);
    }

    /**
     * Whether a mask holds the same objects as the saved one. The numbers of the objects
     * may differ: a mask made by Cellpose or micro-SAM numbers its objects in its own
     * order, QuPath numbers them from the top left.
     */
    static boolean sameObjects(PixelArray mask, Path savedFile) {
        if (!Files.exists(savedFile))
            return false;
        try {
            PixelArray saved = TiffIO.read(savedFile);
            if (saved.width() != mask.width() || saved.height() != mask.height() || saved.samples() != 1)
                return false;
            int[] a = saved.toInts(), b = mask.toInts();
            Map<Integer, Integer> forward = new java.util.HashMap<>(), backward = new java.util.HashMap<>();
            for (int i = 0; i < a.length; i++) {
                if ((a[i] == 0) != (b[i] == 0))
                    return false;
                if (a[i] == 0)
                    continue;
                Integer to = forward.putIfAbsent(a[i], b[i]);
                Integer from = backward.putIfAbsent(b[i], a[i]);
                if ((to != null && to != b[i]) || (from != null && from != a[i]))
                    return false;
            }
            return true;
        } catch (IOException | RuntimeException e) {
            return false;
        }
    }

    /**
     * Leave a unit out of the training set for good (out of focus, artefact): it counts
     * as done, is never shown again and is not exported.
     */
    public Result exclude(Row row) throws Exception {
        checkOpen();
        if (review || row.isAnnotated())
            throw new IllegalStateException("An image that already has a saved mask cannot be excluded");
        RemoteStore remote = set.remote();
        boolean reachable = true;
        if (remote != null) {
            List<Map<String, Object>> now = List.of();
            try {
                now = remote.readRows(set.tableId(), List.of(row.annotationId));
            } catch (Exception e) {
                reachable = false; // excluded here; the table follows when OMERO is back
            }
            if (!now.isEmpty()) {
                Row theirs = RowCodec.fromCells(now.get(0));
                if (theirs.processed) {
                    row.copyFrom(theirs);
                    set.saveLocalTable();
                    return new Result(Outcome.SAVED_ELSEWHERE, "Another computer saved this image first.");
                }
            }
            // Uploads of an earlier, failed attempt would otherwise stay behind.
            for (Object[] part : new Object[][]{{"FileAnnotation", row.labelId}, {"Roi", row.roiId}}) {
                if (part[1] == null || !reachable)
                    continue;
                try {
                    remote.delete((String) part[0], List.of((Long) part[1]));
                } catch (Exception ignored) {
                    // tagged leftovers are harmless
                }
            }
        }
        row.markExcluded(id);
        Files.deleteIfExists(Bundle.maskPath(set.bundle(), row.annotationId));
        set.saveLocalTable();
        if (remote != null && !(reachable ? set.writeTable(List.of(row)) : set.markTablePending(row)))
            return new Result(Outcome.UPLOAD_FAILED, "The image is excluded in this folder, but the tracking "
                    + "table could not be updated. Use \"Upload now\" on the set's overview to complete it.");
        return new Result(Outcome.EXCLUDED, null);
    }

    /** Leave a unit for a later session. */
    public void skip(Row row) {
        releaseQuietly(row);
    }

    private void releaseQuietly(Row row) {
        try {
            set.release(row, token);
            set.saveLocalTable();
        } catch (Exception e) {
            // the claim expires by itself
        }
    }

    /**
     * End the session: hand back the units it still holds and record how it went.
     *
     * @param note        a short note about the session, or null
     * @param interrupted true when the session did not end normally (window closed)
     */
    public void finish(String note, boolean interrupted) throws Exception {
        if (finished)
            return;
        finished = true;
        RemoteStore remote = set.remote();
        if (remote != null) {
            List<String> held = new ArrayList<>();
            for (Row row : set.rows())
                if (token.equals(row.claimedBy))
                    held.add(row.annotationId);
            if (!held.isEmpty()) {
                try {
                    remote.updateCells(set.tableId(), held, cells -> token.equals(String.valueOf(cells.get("claimed_by")))
                            ? Map.of("claimed_by", "", "claimed_at", "") : null);
                    for (Row row : set.rows())
                        if (token.equals(row.claimedBy)) {
                            row.claimedBy = "";
                            row.claimedAt = "";
                        }
                } catch (Exception e) {
                    // claims expire by themselves
                }
            }
        }
        set.saveLocalTable();
        set.endSession(id, saved.size(), note, interrupted ? "interrupted" : "finished");
    }

    private void checkOpen() {
        if (finished)
            throw new IllegalStateException("This session is finished");
    }
}
