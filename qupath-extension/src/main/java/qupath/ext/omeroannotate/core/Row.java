package qupath.ext.omeroannotate.core;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * One annotation unit of a training set and its state: what to annotate, and what was saved.
 * <p>
 * This mirrors {@code omero_annotate_ai.rows.Row} of the Python package field by field, so a
 * set annotated in QuPath can be continued, reviewed or exported there (and the other way
 * round). A unit is one 2D plane of an image, or a rectangular patch of it.
 */
public final class Row {

    /** {@code annotation_type} of a unit the annotator excluded (done, without a mask). */
    public static final String EXCLUDED = "excluded";
    public static final String TRAINING = "training";
    public static final String VALIDATION = "validation";
    public static final String TEST = "test";

    // The image
    public String imageUri = "";
    /** OMERO image ID, or null for images that are not on OMERO. */
    public Long imageId;
    public String imageName = "";
    public String annotationId = "";
    public int level = 0;
    public String well = "";
    public String field = "";
    public Map<String, String> wellMetadata = new LinkedHashMap<>();

    // The unit
    public String category = TRAINING;
    public int timepoint = -1;
    public int zSlice = -1;
    public int channel = -1;
    public boolean isVolumetric = false;
    public int zStart = -1;
    public int zEnd = -1;
    public int zLength = 1;
    public boolean isPatch = false;
    public int patchX = 0;
    public int patchY = 0;
    public int patchWidth = 0;
    public int patchHeight = 0;

    // State
    public boolean processed = false;
    public String annotationCreatedAt;
    public String annotationUpdatedAt;
    public String annotationType = "segmentation_mask";
    /** Session that annotated the unit (-1: none). */
    public int session = -1;
    public Long roiId;
    public Long labelId;
    public Long annotationInputId;
    public String inputDigest;
    public String maskDigest;
    public String inputIscc;
    public String maskIscc;
    public String claimedBy = "";
    public String claimedAt = "";
    /** Last upload error; kept in the local CSV only. */
    public String error;

    /** Whether the annotator excluded the unit: done, with no mask and no uploads. */
    public boolean isExcluded() {
        return processed && EXCLUDED.equals(annotationType);
    }

    /** Whether the unit has a saved mask. */
    public boolean isAnnotated() {
        return processed && !isExcluded();
    }

    /** Mark the unit annotated, keeping the time of its first annotation. */
    public void markProcessed(String type) {
        String now = pythonNow();
        processed = true;
        annotationUpdatedAt = now;
        if (annotationCreatedAt == null)
            annotationCreatedAt = now;
        annotationType = type;
    }

    /** Mark the unit excluded by {@code session}: done, without mask, uploads or claim. */
    public void markExcluded(int session) {
        markProcessed(EXCLUDED);
        this.session = session;
        labelId = null;
        roiId = null;
        maskDigest = null;
        maskIscc = null;
        error = null;
        claimedBy = "";
        claimedAt = "";
    }

    /** Copy every field of {@code other} into this row. */
    public void copyFrom(Row other) {
        imageUri = other.imageUri;
        imageId = other.imageId;
        imageName = other.imageName;
        annotationId = other.annotationId;
        level = other.level;
        well = other.well;
        field = other.field;
        wellMetadata = new LinkedHashMap<>(other.wellMetadata);
        category = other.category;
        timepoint = other.timepoint;
        zSlice = other.zSlice;
        channel = other.channel;
        isVolumetric = other.isVolumetric;
        zStart = other.zStart;
        zEnd = other.zEnd;
        zLength = other.zLength;
        isPatch = other.isPatch;
        patchX = other.patchX;
        patchY = other.patchY;
        patchWidth = other.patchWidth;
        patchHeight = other.patchHeight;
        processed = other.processed;
        annotationCreatedAt = other.annotationCreatedAt;
        annotationUpdatedAt = other.annotationUpdatedAt;
        annotationType = other.annotationType;
        session = other.session;
        roiId = other.roiId;
        labelId = other.labelId;
        annotationInputId = other.annotationInputId;
        inputDigest = other.inputDigest;
        maskDigest = other.maskDigest;
        inputIscc = other.inputIscc;
        maskIscc = other.maskIscc;
        claimedBy = other.claimedBy;
        claimedAt = other.claimedAt;
        error = other.error;
    }

    public Row copy() {
        Row row = new Row();
        row.copyFrom(this);
        return row;
    }

    /** The current local time as Python's {@code datetime.now().isoformat()} writes it. */
    public static String pythonNow() {
        return LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss.SSSSSS"));
    }

    @Override
    public String toString() {
        return "Row[" + annotationId + ", " + category + (processed ? ", done" : "") + "]";
    }
}
