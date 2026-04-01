package qupath.ext.omero.annotate.core;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.time.Instant;

/**
 * Tracks the state of a single annotation unit (image or patch).
 * <p>
 * Mirrors the Python {@code ImageAnnotation} class from
 * {@code omero_annotate_ai.core.annotation_config} to ensure
 * full round-trip compatibility between QuPath and Python workflows.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class ImageAnnotationRecord {

    @JsonProperty("image_id")
    private long imageId;

    @JsonProperty("image_name")
    private String imageName;

    @JsonProperty("category")
    private String category = "training"; // training, validation, test

    @JsonProperty("channel")
    private int channel = 0;

    @JsonProperty("z_slice")
    private int zSlice = 0;

    @JsonProperty("timepoint")
    private int timepoint = 0;

    @JsonProperty("processed")
    private boolean processed = false;

    @JsonProperty("roi_id")
    private Long roiId;

    @JsonProperty("label_id")
    private Long labelId;

    @JsonProperty("is_patch")
    private boolean isPatch = false;

    @JsonProperty("is_volumetric")
    private boolean isVolumetric = false;

    @JsonProperty("patch_x")
    private int patchX = 0;

    @JsonProperty("patch_y")
    private int patchY = 0;

    @JsonProperty("patch_width")
    private int patchWidth = 0;

    @JsonProperty("patch_height")
    private int patchHeight = 0;

    @JsonProperty("annotation_type")
    private String annotationType = "qupath_manual";

    @JsonProperty("annotation_created_at")
    private String annotationCreatedAt;

    @JsonProperty("annotation_updated_at")
    private String annotationUpdatedAt;

    @JsonProperty("schema_attachment_id")
    private Long schemaAttachmentId;

    @JsonProperty("label_input_id")
    private Long labelInputId;

    @JsonProperty("z_start")
    private int zStart = 0;

    @JsonProperty("z_end")
    private int zEnd = 0;

    @JsonProperty("z_length")
    private int zLength = 0;

    public ImageAnnotationRecord() {
    }

    public ImageAnnotationRecord(long imageId, String imageName) {
        this.imageId = imageId;
        this.imageName = imageName;
    }

    /**
     * Mark this annotation as processed with the current timestamp.
     */
    public void markProcessed() {
        this.processed = true;
        this.annotationUpdatedAt = Instant.now().toString();
        if (this.annotationCreatedAt == null) {
            this.annotationCreatedAt = this.annotationUpdatedAt;
        }
    }

    /**
     * Mark as processed with OMERO IDs for the uploaded ROI and label.
     */
    public void markProcessed(Long roiId, Long labelId) {
        this.roiId = roiId;
        this.labelId = labelId;
        markProcessed();
    }

    // --- Conversion helpers for OMERO table compatibility ---

    /**
     * Whether this record belongs to the training set.
     */
    public boolean isTrain() {
        return "training".equals(category);
    }

    /**
     * Whether this record belongs to the validation set.
     */
    public boolean isValidate() {
        return "validation".equals(category);
    }

    /**
     * Convert an optional Long to String, using "None" for null.
     * Matches Python's _optional_int_to_str() behavior.
     */
    public static String optionalLongToStr(Long value) {
        return value == null ? "None" : String.valueOf(value);
    }

    // --- Getters and setters ---

    public long getImageId() { return imageId; }
    public void setImageId(long imageId) { this.imageId = imageId; }

    public String getImageName() { return imageName; }
    public void setImageName(String imageName) { this.imageName = imageName; }

    public String getCategory() { return category; }
    public void setCategory(String category) { this.category = category; }

    public int getChannel() { return channel; }
    public void setChannel(int channel) { this.channel = channel; }

    public int getZSlice() { return zSlice; }
    public void setZSlice(int zSlice) { this.zSlice = zSlice; }

    public int getTimepoint() { return timepoint; }
    public void setTimepoint(int timepoint) { this.timepoint = timepoint; }

    public boolean isProcessed() { return processed; }
    public void setProcessed(boolean processed) { this.processed = processed; }

    public Long getRoiId() { return roiId; }
    public void setRoiId(Long roiId) { this.roiId = roiId; }

    public Long getLabelId() { return labelId; }
    public void setLabelId(Long labelId) { this.labelId = labelId; }

    public boolean isPatch() { return isPatch; }
    public void setIsPatch(boolean isPatch) { this.isPatch = isPatch; }

    public boolean isVolumetric() { return isVolumetric; }
    public void setIsVolumetric(boolean isVolumetric) { this.isVolumetric = isVolumetric; }

    public int getPatchX() { return patchX; }
    public void setPatchX(int patchX) { this.patchX = patchX; }

    public int getPatchY() { return patchY; }
    public void setPatchY(int patchY) { this.patchY = patchY; }

    public int getPatchWidth() { return patchWidth; }
    public void setPatchWidth(int patchWidth) { this.patchWidth = patchWidth; }

    public int getPatchHeight() { return patchHeight; }
    public void setPatchHeight(int patchHeight) { this.patchHeight = patchHeight; }

    public String getAnnotationType() { return annotationType; }
    public void setAnnotationType(String annotationType) { this.annotationType = annotationType; }

    public String getAnnotationCreatedAt() { return annotationCreatedAt; }
    public void setAnnotationCreatedAt(String annotationCreatedAt) { this.annotationCreatedAt = annotationCreatedAt; }

    public String getAnnotationUpdatedAt() { return annotationUpdatedAt; }
    public void setAnnotationUpdatedAt(String annotationUpdatedAt) { this.annotationUpdatedAt = annotationUpdatedAt; }

    public Long getSchemaAttachmentId() { return schemaAttachmentId; }
    public void setSchemaAttachmentId(Long schemaAttachmentId) { this.schemaAttachmentId = schemaAttachmentId; }

    public Long getLabelInputId() { return labelInputId; }
    public void setLabelInputId(Long labelInputId) { this.labelInputId = labelInputId; }

    public int getZStart() { return zStart; }
    public void setZStart(int zStart) { this.zStart = zStart; }

    public int getZEnd() { return zEnd; }
    public void setZEnd(int zEnd) { this.zEnd = zEnd; }

    public int getZLength() { return zLength; }
    public void setZLength(int zLength) { this.zLength = zLength; }
}
