package qupath.ext.omero.annotate.omero;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.omero.annotate.core.ImageAnnotationRecord;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;

/**
 * Manages OMERO tracking tables with the same schema as the Python
 * {@code omero_annotate_ai} package.
 * <p>
 * All OMERO API calls use reflection to avoid compile-time dependencies,
 * enabling the extension to work in local-only mode without OMERO classes.
 * <p>
 * Table schema (23 columns):
 * image_id, image_name, train, validate, channel, z_slice, timepoint,
 * label_id, roi_id, is_patch, is_volumetric, patch_x, patch_y, patch_width,
 * patch_height, processed, annotation_type, annotation_created_at,
 * annotation_updated_at, schema_attachment_id, label_input_id,
 * z_start, z_end, z_length
 */
public class OmeroTableManager {

    private static final Logger logger = LoggerFactory.getLogger(OmeroTableManager.class);

    private static final String TABLE_NAME = "omero_annotate_ai_tracking";

    private final OmeroConnectionManager connectionManager;

    public OmeroTableManager(OmeroConnectionManager connectionManager) {
        this.connectionManager = connectionManager;
    }

    /**
     * Create a new tracking table on OMERO attached to the given dataset.
     *
     * @param datasetId the OMERO dataset ID to attach the table to
     * @param records   initial records to populate the table with
     * @return the OMERO table annotation ID, or -1 if creation failed
     */
    public long createTable(long datasetId, List<ImageAnnotationRecord> records) {
        if (!connectionManager.isConnected()) {
            logger.warn("Cannot create table: not connected to OMERO");
            return -1;
        }

        try {
            var gateway = connectionManager.getGateway();
            var ctx = connectionManager.getSecurityContext();

            // Get TablesFacility via reflection
            Class<?> tablesFacilityClass = Class.forName("omero.gateway.facility.TablesFacility");
            Method getFacility = gateway.getClass().getMethod("getFacility", Class.class);
            Object tablesFacility = getFacility.invoke(gateway, tablesFacilityClass);

            // Build TableData
            Object tableData = buildTableData(records);

            // Add table to dataset
            Class<?> dataObjectClass = Class.forName("omero.gateway.model.DatasetData");
            Object datasetData = dataObjectClass.getDeclaredConstructor().newInstance();
            Method setId = datasetData.getClass().getMethod("setId", long.class);
            setId.invoke(datasetData, datasetId);

            Method addTable = tablesFacility.getClass().getMethod(
                    "addTable", ctx.getClass(), dataObjectClass, String.class,
                    Class.forName("omero.gateway.model.TableData")
            );
            Object result = addTable.invoke(tablesFacility, ctx, datasetData, TABLE_NAME, tableData);

            // Get the table ID from the result
            Method getOriginalFileId = result.getClass().getMethod("getOriginalFileId");
            long tableId = (long) getOriginalFileId.invoke(result);

            logger.info("Created OMERO tracking table with ID {} on dataset {}", tableId, datasetId);
            return tableId;

        } catch (Exception e) {
            logger.error("Failed to create OMERO tracking table: {}", e.getMessage(), e);
            return -1;
        }
    }

    /**
     * Read records from an existing OMERO tracking table.
     *
     * @param tableId the OMERO table annotation ID
     * @return list of records, or empty list if reading failed
     */
    public List<ImageAnnotationRecord> readTable(long tableId) {
        if (!connectionManager.isConnected()) {
            logger.warn("Cannot read table: not connected to OMERO");
            return List.of();
        }

        try {
            var gateway = connectionManager.getGateway();
            var ctx = connectionManager.getSecurityContext();

            Class<?> tablesFacilityClass = Class.forName("omero.gateway.facility.TablesFacility");
            Method getFacility = gateway.getClass().getMethod("getFacility", Class.class);
            Object tablesFacility = getFacility.invoke(gateway, tablesFacilityClass);

            // Get table data
            Method getTable = tablesFacility.getClass().getMethod(
                    "getTable", ctx.getClass(), long.class
            );
            Object tableData = getTable.invoke(tablesFacility, ctx, tableId);

            return parseTableData(tableData);

        } catch (Exception e) {
            logger.error("Failed to read OMERO tracking table {}: {}", tableId, e.getMessage(), e);
            return List.of();
        }
    }

    /**
     * Add rows to an existing OMERO tracking table.
     *
     * @param tableId the OMERO table annotation ID
     * @param records records to add
     * @return true if rows were added successfully
     */
    public boolean addRows(long tableId, List<ImageAnnotationRecord> records) {
        if (!connectionManager.isConnected() || records.isEmpty()) {
            return false;
        }

        try {
            var gateway = connectionManager.getGateway();
            var ctx = connectionManager.getSecurityContext();

            Class<?> tablesFacilityClass = Class.forName("omero.gateway.facility.TablesFacility");
            Method getFacility = gateway.getClass().getMethod("getFacility", Class.class);
            Object tablesFacility = getFacility.invoke(gateway, tablesFacilityClass);

            Object tableData = buildTableData(records);

            Method addRows = tablesFacility.getClass().getMethod(
                    "addRows", ctx.getClass(), long.class,
                    Class.forName("omero.gateway.model.TableData")
            );
            addRows.invoke(tablesFacility, ctx, tableId, tableData);

            logger.info("Added {} rows to OMERO table {}", records.size(), tableId);
            return true;

        } catch (Exception e) {
            logger.error("Failed to add rows to OMERO table {}: {}", tableId, e.getMessage(), e);
            return false;
        }
    }

    /**
     * Build OMERO TableData from annotation records using reflection.
     */
    private Object buildTableData(List<ImageAnnotationRecord> records) throws Exception {
        Class<?> tableDataClass = Class.forName("omero.gateway.model.TableData");
        Class<?> tableDataColumnClass = Class.forName("omero.gateway.model.TableDataColumn");

        // Define columns matching the Python schema
        Object[] columns = new Object[]{
                createColumn(tableDataColumnClass, "image_id", Long.class, 0),
                createColumn(tableDataColumnClass, "image_name", String.class, 1),
                createColumn(tableDataColumnClass, "train", Boolean.class, 2),
                createColumn(tableDataColumnClass, "validate", Boolean.class, 3),
                createColumn(tableDataColumnClass, "channel", Long.class, 4),
                createColumn(tableDataColumnClass, "z_slice", Long.class, 5),
                createColumn(tableDataColumnClass, "timepoint", Long.class, 6),
                createColumn(tableDataColumnClass, "label_id", String.class, 7),
                createColumn(tableDataColumnClass, "roi_id", String.class, 8),
                createColumn(tableDataColumnClass, "is_patch", Boolean.class, 9),
                createColumn(tableDataColumnClass, "is_volumetric", Boolean.class, 10),
                createColumn(tableDataColumnClass, "patch_x", Long.class, 11),
                createColumn(tableDataColumnClass, "patch_y", Long.class, 12),
                createColumn(tableDataColumnClass, "patch_width", Long.class, 13),
                createColumn(tableDataColumnClass, "patch_height", Long.class, 14),
                createColumn(tableDataColumnClass, "processed", Boolean.class, 15),
                createColumn(tableDataColumnClass, "annotation_type", String.class, 16),
                createColumn(tableDataColumnClass, "annotation_created_at", String.class, 17),
                createColumn(tableDataColumnClass, "annotation_updated_at", String.class, 18),
                createColumn(tableDataColumnClass, "schema_attachment_id", String.class, 19),
                createColumn(tableDataColumnClass, "label_input_id", String.class, 20),
                createColumn(tableDataColumnClass, "z_start", Long.class, 21),
                createColumn(tableDataColumnClass, "z_end", Long.class, 22),
                createColumn(tableDataColumnClass, "z_length", Long.class, 23),
        };

        // Build data arrays
        int nRows = records.size();
        Object[][] data = new Object[24][nRows];

        for (int i = 0; i < nRows; i++) {
            var r = records.get(i);
            data[0][i] = r.getImageId();
            data[1][i] = r.getImageName();
            data[2][i] = r.isTrain();
            data[3][i] = r.isValidate();
            data[4][i] = (long) r.getChannel();
            data[5][i] = (long) r.getZSlice();
            data[6][i] = (long) r.getTimepoint();
            data[7][i] = ImageAnnotationRecord.optionalLongToStr(r.getLabelId());
            data[8][i] = ImageAnnotationRecord.optionalLongToStr(r.getRoiId());
            data[9][i] = r.isPatch();
            data[10][i] = r.isVolumetric();
            data[11][i] = (long) r.getPatchX();
            data[12][i] = (long) r.getPatchY();
            data[13][i] = (long) r.getPatchWidth();
            data[14][i] = (long) r.getPatchHeight();
            data[15][i] = r.isProcessed();
            data[16][i] = r.getAnnotationType();
            data[17][i] = r.getAnnotationCreatedAt() != null ? r.getAnnotationCreatedAt() : "None";
            data[18][i] = r.getAnnotationUpdatedAt() != null ? r.getAnnotationUpdatedAt() : "None";
            data[19][i] = ImageAnnotationRecord.optionalLongToStr(r.getSchemaAttachmentId());
            data[20][i] = ImageAnnotationRecord.optionalLongToStr(r.getLabelInputId());
            data[21][i] = (long) r.getZStart();
            data[22][i] = (long) r.getZEnd();
            data[23][i] = (long) r.getZLength();
        }

        // Create TableData(columns, data)
        var columnsArray = java.lang.reflect.Array.newInstance(tableDataColumnClass, columns.length);
        for (int i = 0; i < columns.length; i++) {
            java.lang.reflect.Array.set(columnsArray, i, columns[i]);
        }

        return tableDataClass.getDeclaredConstructor(columnsArray.getClass(), Object[][].class)
                .newInstance(columnsArray, data);
    }

    private Object createColumn(Class<?> columnClass, String name, Class<?> type, int index) throws Exception {
        return columnClass.getDeclaredConstructor(String.class, Class.class, int.class)
                .newInstance(name, type, index);
    }

    /**
     * Parse OMERO TableData into annotation records.
     */
    private List<ImageAnnotationRecord> parseTableData(Object tableData) throws Exception {
        Method getData = tableData.getClass().getMethod("getData");
        Object[][] data = (Object[][]) getData.invoke(tableData);

        Method getColumns = tableData.getClass().getMethod("getColumns");
        Object[] columns = (Object[]) getColumns.invoke(tableData);

        Method getNumberOfRows = tableData.getClass().getMethod("getNumberOfRows");
        long nRows = (long) getNumberOfRows.invoke(tableData);

        var records = new ArrayList<ImageAnnotationRecord>();
        for (int i = 0; i < nRows; i++) {
            var record = new ImageAnnotationRecord();
            for (int j = 0; j < columns.length; j++) {
                Method getName = columns[j].getClass().getMethod("getName");
                String colName = (String) getName.invoke(columns[j]);
                Object value = data[j][i];

                setRecordField(record, colName, value);
            }
            records.add(record);
        }

        return records;
    }

    private void setRecordField(ImageAnnotationRecord record, String colName, Object value) {
        if (value == null) return;
        switch (colName) {
            case "image_id" -> record.setImageId((long) value);
            case "image_name" -> record.setImageName((String) value);
            case "train" -> { if ((boolean) value) record.setCategory("training"); }
            case "validate" -> { if ((boolean) value) record.setCategory("validation"); }
            case "channel" -> record.setChannel(((Long) value).intValue());
            case "z_slice" -> record.setZSlice(((Long) value).intValue());
            case "timepoint" -> record.setTimepoint(((Long) value).intValue());
            case "label_id" -> record.setLabelId(parseOptionalLong(value));
            case "roi_id" -> record.setRoiId(parseOptionalLong(value));
            case "is_patch" -> record.setIsPatch((boolean) value);
            case "is_volumetric" -> record.setIsVolumetric((boolean) value);
            case "patch_x" -> record.setPatchX(((Long) value).intValue());
            case "patch_y" -> record.setPatchY(((Long) value).intValue());
            case "patch_width" -> record.setPatchWidth(((Long) value).intValue());
            case "patch_height" -> record.setPatchHeight(((Long) value).intValue());
            case "processed" -> record.setProcessed((boolean) value);
            case "annotation_type" -> record.setAnnotationType((String) value);
            case "annotation_created_at" -> record.setAnnotationCreatedAt("None".equals(value) ? null : (String) value);
            case "annotation_updated_at" -> record.setAnnotationUpdatedAt("None".equals(value) ? null : (String) value);
            case "schema_attachment_id" -> record.setSchemaAttachmentId(parseOptionalLong(value));
            case "label_input_id" -> record.setLabelInputId(parseOptionalLong(value));
            case "z_start" -> record.setZStart(((Long) value).intValue());
            case "z_end" -> record.setZEnd(((Long) value).intValue());
            case "z_length" -> record.setZLength(((Long) value).intValue());
        }
    }

    private Long parseOptionalLong(Object value) {
        if (value == null) return null;
        if (value instanceof Long l) return l;
        String s = value.toString();
        if ("None".equals(s) || s.isEmpty()) return null;
        try { return Long.parseLong(s); }
        catch (NumberFormatException e) { return null; }
    }
}
