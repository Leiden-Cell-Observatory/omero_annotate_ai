package qupath.ext.omero.annotate.core;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * CSV-based tracking table for local-only mode (no OMERO).
 * <p>
 * Uses the same column schema as the OMERO tracking table so that
 * data can be migrated to OMERO later if needed. Mirrors the napari
 * plugin's local {@code tracking_table.csv} approach.
 */
public class LocalTrackingTable {

    private static final Logger logger = LoggerFactory.getLogger(LocalTrackingTable.class);

    private static final String[] COLUMNS = {
            "image_id", "image_name", "train", "validate",
            "channel", "z_slice", "timepoint",
            "label_id", "roi_id",
            "is_patch", "is_volumetric",
            "patch_x", "patch_y", "patch_width", "patch_height",
            "processed", "annotation_type",
            "annotation_created_at", "annotation_updated_at",
            "schema_attachment_id", "label_input_id",
            "z_start", "z_end", "z_length"
    };

    private final List<ImageAnnotationRecord> records = new ArrayList<>();
    private Path csvPath;

    public LocalTrackingTable(Path outputDirectory) {
        this.csvPath = outputDirectory.resolve("tracking_table.csv");
    }

    /**
     * Add a record to the tracking table.
     */
    public void addRecord(ImageAnnotationRecord record) {
        records.add(record);
    }

    /**
     * Add multiple records to the tracking table.
     */
    public void addRecords(List<ImageAnnotationRecord> newRecords) {
        records.addAll(newRecords);
    }

    /**
     * Update a record by image ID - marks it as processed.
     */
    public void updateRecord(long imageId, Long roiId, Long labelId) {
        for (var record : records) {
            if (record.getImageId() == imageId) {
                record.markProcessed(roiId, labelId);
            }
        }
    }

    /**
     * Save the tracking table to CSV.
     */
    public void saveToCsv() throws IOException {
        saveToCsv(csvPath);
    }

    /**
     * Save the tracking table to a specific CSV path.
     */
    public void saveToCsv(Path path) throws IOException {
        logger.info("Saving tracking table to {}", path);
        Files.createDirectories(path.getParent());

        try (BufferedWriter writer = Files.newBufferedWriter(path)) {
            // Header
            writer.write(String.join(",", COLUMNS));
            writer.newLine();

            // Data rows
            for (var record : records) {
                writer.write(formatRow(record));
                writer.newLine();
            }
        }
    }

    /**
     * Load tracking table from an existing CSV file.
     */
    public static LocalTrackingTable loadFromCsv(Path path) throws IOException {
        logger.info("Loading tracking table from {}", path);
        var lines = Files.readAllLines(path);
        if (lines.isEmpty()) {
            throw new IOException("Empty tracking table file: " + path);
        }

        // Parse header to get column indices
        var header = lines.get(0).split(",");
        var table = new LocalTrackingTable(path.getParent());

        for (int i = 1; i < lines.size(); i++) {
            var values = lines.get(i).split(",", -1);
            var record = parseRow(header, values);
            table.addRecord(record);
        }

        logger.info("Loaded {} records from tracking table", table.records.size());
        return table;
    }

    public List<ImageAnnotationRecord> getRecords() {
        return records;
    }

    public Path getCsvPath() {
        return csvPath;
    }

    // --- Private helpers ---

    private String formatRow(ImageAnnotationRecord r) {
        return String.join(",",
                String.valueOf(r.getImageId()),
                escapeCsv(r.getImageName()),
                String.valueOf(r.isTrain()),
                String.valueOf(r.isValidate()),
                String.valueOf(r.getChannel()),
                String.valueOf(r.getZSlice()),
                String.valueOf(r.getTimepoint()),
                ImageAnnotationRecord.optionalLongToStr(r.getLabelId()),
                ImageAnnotationRecord.optionalLongToStr(r.getRoiId()),
                String.valueOf(r.isPatch()),
                String.valueOf(r.isVolumetric()),
                String.valueOf(r.getPatchX()),
                String.valueOf(r.getPatchY()),
                String.valueOf(r.getPatchWidth()),
                String.valueOf(r.getPatchHeight()),
                String.valueOf(r.isProcessed()),
                escapeCsv(r.getAnnotationType()),
                escapeCsv(r.getAnnotationCreatedAt()),
                escapeCsv(r.getAnnotationUpdatedAt()),
                ImageAnnotationRecord.optionalLongToStr(r.getSchemaAttachmentId()),
                ImageAnnotationRecord.optionalLongToStr(r.getLabelInputId()),
                String.valueOf(r.getZStart()),
                String.valueOf(r.getZEnd()),
                String.valueOf(r.getZLength())
        );
    }

    private static ImageAnnotationRecord parseRow(String[] header, String[] values) {
        var record = new ImageAnnotationRecord();
        for (int i = 0; i < header.length && i < values.length; i++) {
            String col = header[i].trim();
            String val = values[i].trim();
            switch (col) {
                case "image_id" -> record.setImageId(Long.parseLong(val));
                case "image_name" -> record.setImageName(val);
                case "train" -> { if ("true".equalsIgnoreCase(val)) record.setCategory("training"); }
                case "validate" -> { if ("true".equalsIgnoreCase(val)) record.setCategory("validation"); }
                case "channel" -> record.setChannel(Integer.parseInt(val));
                case "z_slice" -> record.setZSlice(Integer.parseInt(val));
                case "timepoint" -> record.setTimepoint(Integer.parseInt(val));
                case "label_id" -> record.setLabelId(parseOptionalLong(val));
                case "roi_id" -> record.setRoiId(parseOptionalLong(val));
                case "is_patch" -> record.setIsPatch(Boolean.parseBoolean(val));
                case "is_volumetric" -> record.setIsVolumetric(Boolean.parseBoolean(val));
                case "patch_x" -> record.setPatchX(Integer.parseInt(val));
                case "patch_y" -> record.setPatchY(Integer.parseInt(val));
                case "patch_width" -> record.setPatchWidth(Integer.parseInt(val));
                case "patch_height" -> record.setPatchHeight(Integer.parseInt(val));
                case "processed" -> record.setProcessed(Boolean.parseBoolean(val));
                case "annotation_type" -> record.setAnnotationType(val);
                case "annotation_created_at" -> record.setAnnotationCreatedAt("None".equals(val) ? null : val);
                case "annotation_updated_at" -> record.setAnnotationUpdatedAt("None".equals(val) ? null : val);
                case "schema_attachment_id" -> record.setSchemaAttachmentId(parseOptionalLong(val));
                case "label_input_id" -> record.setLabelInputId(parseOptionalLong(val));
                case "z_start" -> record.setZStart(Integer.parseInt(val));
                case "z_end" -> record.setZEnd(Integer.parseInt(val));
                case "z_length" -> record.setZLength(Integer.parseInt(val));
            }
        }
        return record;
    }

    private static Long parseOptionalLong(String val) {
        if (val == null || val.isEmpty() || "None".equals(val)) return null;
        try {
            return Long.parseLong(val);
        } catch (NumberFormatException e) {
            return null;
        }
    }

    private static String escapeCsv(String value) {
        if (value == null) return "None";
        if (value.contains(",") || value.contains("\"") || value.contains("\n")) {
            return "\"" + value.replace("\"", "\"\"") + "\"";
        }
        return value;
    }
}
