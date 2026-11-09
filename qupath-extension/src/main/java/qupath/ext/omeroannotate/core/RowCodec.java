package qupath.ext.omeroannotate.core;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

/**
 * The one mapping between a {@link Row} and a table row, used for both the local
 * {@code tracking_table.csv} and the OMERO tracking table. Column names, order and value
 * conventions are those of {@code omero_annotate_ai.rows} (Python).
 */
public final class RowCodec {

    /** Columns of the tracking table, in table order. */
    public static final List<String> COLUMNS = List.of(
            "image_id", "image_uri", "image_name", "annotation_id", "category", "channel",
            "z_slice", "timepoint", "label_id", "roi_id", "is_volumetric", "processed",
            "is_patch", "patch_x", "patch_y", "patch_width", "patch_height", "annotation_type",
            "annotation_created_at", "annotation_updated_at", "annotation_input_id", "session",
            "z_start", "z_end", "z_length", "level", "well", "field", "well_metadata",
            "input_digest", "mask_digest", "input_iscc", "mask_iscc", "claimed_by", "claimed_at");

    /** Columns holding true/false. */
    public static final Set<String> BOOL_COLUMNS = Set.of("processed", "is_patch", "is_volumetric");

    /** Columns holding whole numbers (OMERO IDs use 0 for "none"). */
    public static final Set<String> LONG_COLUMNS = Set.of(
            "image_id", "label_id", "roi_id", "annotation_input_id", "channel", "z_slice",
            "timepoint", "patch_x", "patch_y", "patch_width", "patch_height", "session",
            "z_start", "z_end", "z_length", "level");

    private static final Gson GSON = new GsonBuilder().disableHtmlEscaping().create();

    private RowCodec() {}

    // --- Row <-> map of cells -------------------------------------------------------

    /** The cells of {@code row} by column name: Long, Boolean or String values. */
    public static Map<String, Object> toCells(Row r) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("image_id", id(r.imageId));
        m.put("image_uri", r.imageUri);
        m.put("image_name", r.imageName);
        m.put("annotation_id", r.annotationId);
        m.put("category", r.category);
        m.put("channel", (long) r.channel);
        m.put("z_slice", (long) r.zSlice);
        m.put("timepoint", (long) r.timepoint);
        m.put("label_id", id(r.labelId));
        m.put("roi_id", id(r.roiId));
        m.put("is_volumetric", r.isVolumetric);
        m.put("processed", r.processed);
        m.put("is_patch", r.isPatch);
        m.put("patch_x", (long) r.patchX);
        m.put("patch_y", (long) r.patchY);
        m.put("patch_width", (long) r.patchWidth);
        m.put("patch_height", (long) r.patchHeight);
        m.put("annotation_type", r.annotationType);
        m.put("annotation_created_at", r.annotationCreatedAt == null ? "None" : r.annotationCreatedAt);
        m.put("annotation_updated_at", r.annotationUpdatedAt == null ? "None" : r.annotationUpdatedAt);
        m.put("annotation_input_id", id(r.annotationInputId));
        m.put("session", (long) r.session);
        m.put("z_start", (long) r.zStart);
        m.put("z_end", (long) r.zEnd);
        m.put("z_length", (long) r.zLength);
        m.put("level", (long) r.level);
        m.put("well", r.well);
        m.put("field", r.field);
        m.put("well_metadata", wellMetadataJson(r.wellMetadata));
        m.put("input_digest", str(r.inputDigest));
        m.put("mask_digest", str(r.maskDigest));
        m.put("input_iscc", str(r.inputIscc));
        m.put("mask_iscc", str(r.maskIscc));
        m.put("claimed_by", r.claimedBy);
        m.put("claimed_at", r.claimedAt);
        return m;
    }

    /** A row from its cells (any of Number, Boolean or String per cell; missing cells get defaults). */
    public static Row fromCells(Map<String, ?> m) {
        Row r = new Row();
        r.imageId = optionalId(m.get("image_id"));
        r.imageUri = string(m.get("image_uri"));
        r.imageName = string(m.get("image_name"));
        r.annotationId = string(m.get("annotation_id"));
        if (r.imageUri.isEmpty() && r.imageId != null)
            r.imageUri = Ids.omeroUri(r.imageId);
        String category = string(m.get("category"));
        r.category = List.of(Row.TRAINING, Row.VALIDATION, Row.TEST).contains(category) ? category : Row.TRAINING;
        r.channel = integer(m.get("channel"), -1);
        r.zSlice = integer(m.get("z_slice"), -1);
        r.timepoint = integer(m.get("timepoint"), -1);
        r.labelId = optionalId(m.get("label_id"));
        r.roiId = optionalId(m.get("roi_id"));
        r.isVolumetric = bool(m.get("is_volumetric"));
        r.processed = bool(m.get("processed"));
        r.isPatch = bool(m.get("is_patch"));
        r.patchX = integer(m.get("patch_x"), 0);
        r.patchY = integer(m.get("patch_y"), 0);
        r.patchWidth = integer(m.get("patch_width"), 0);
        r.patchHeight = integer(m.get("patch_height"), 0);
        String type = string(m.get("annotation_type"));
        r.annotationType = type.isEmpty() ? "segmentation_mask" : type;
        r.annotationCreatedAt = optionalString(m.get("annotation_created_at"));
        r.annotationUpdatedAt = optionalString(m.get("annotation_updated_at"));
        r.annotationInputId = optionalId(m.get("annotation_input_id"));
        r.session = integer(m.get("session"), -1);
        r.zStart = integer(m.get("z_start"), -1);
        r.zEnd = integer(m.get("z_end"), -1);
        r.zLength = integer(m.get("z_length"), 1);
        r.level = integer(m.get("level"), 0);
        r.well = string(m.get("well"));
        r.field = string(m.get("field"));
        r.wellMetadata = parseWellMetadata(string(m.get("well_metadata")));
        r.inputDigest = emptyToNull(string(m.get("input_digest")));
        r.maskDigest = emptyToNull(string(m.get("mask_digest")));
        r.inputIscc = emptyToNull(string(m.get("input_iscc")));
        r.maskIscc = emptyToNull(string(m.get("mask_iscc")));
        r.claimedBy = string(m.get("claimed_by"));
        r.claimedAt = string(m.get("claimed_at"));
        r.error = emptyToNull(string(m.get("error")));
        return r;
    }

    // --- CSV ------------------------------------------------------------------------

    /** The local {@code tracking_table.csv} holding {@code rows} (with the CSV-only error column). */
    public static String toCsv(List<Row> rows) {
        List<String> columns = new ArrayList<>(COLUMNS);
        columns.add("error");
        StringBuilder sb = new StringBuilder();
        sb.append(String.join(",", columns)).append('\n');
        for (Row row : rows) {
            Map<String, Object> cells = toCells(row);
            cells.put("error", str(row.error));
            boolean first = true;
            for (String column : columns) {
                if (!first)
                    sb.append(',');
                first = false;
                Object value = cells.get(column);
                if (value instanceof Boolean b)
                    sb.append(b ? "True" : "False"); // as pandas writes booleans
                else
                    sb.append(quote(String.valueOf(value)));
            }
            sb.append('\n');
        }
        return sb.toString();
    }

    /**
     * The rows of a {@code tracking_table.csv}.
     *
     * @throws IllegalArgumentException if the table has no {@code annotation_id} column
     */
    public static List<Row> fromCsv(String text) {
        List<List<String>> records = parseCsv(text);
        List<Row> rows = new ArrayList<>();
        if (records.isEmpty())
            return rows;
        List<String> header = records.get(0);
        if (!header.contains("annotation_id"))
            throw new IllegalArgumentException("This tracking table has no annotation_id column; "
                    + "tables made before omero-annotate-ai 0.5.0 are not supported");
        for (List<String> record : records.subList(1, records.size())) {
            if (record.size() == 1 && record.get(0).isEmpty())
                continue; // blank line
            Map<String, Object> cells = new LinkedHashMap<>();
            for (int i = 0; i < header.size() && i < record.size(); i++)
                cells.put(header.get(i), record.get(i));
            rows.add(fromCells(cells));
        }
        return rows;
    }

    private static String quote(String value) {
        if (value.indexOf(',') < 0 && value.indexOf('"') < 0 && value.indexOf('\n') < 0 && value.indexOf('\r') < 0)
            return value;
        return '"' + value.replace("\"", "\"\"") + '"';
    }

    static List<List<String>> parseCsv(String text) {
        List<List<String>> records = new ArrayList<>();
        List<String> record = new ArrayList<>();
        StringBuilder field = new StringBuilder();
        boolean quoted = false;
        boolean any = false;
        int n = text.length();
        for (int i = 0; i < n; i++) {
            char c = text.charAt(i);
            if (quoted) {
                if (c == '"') {
                    if (i + 1 < n && text.charAt(i + 1) == '"') {
                        field.append('"');
                        i++;
                    } else {
                        quoted = false;
                    }
                } else {
                    field.append(c);
                }
            } else if (c == '"') {
                quoted = true;
                any = true;
            } else if (c == ',') {
                record.add(field.toString());
                field.setLength(0);
                any = true;
            } else if (c == '\n' || c == '\r') {
                if (c == '\r' && i + 1 < n && text.charAt(i + 1) == '\n')
                    i++;
                record.add(field.toString());
                field.setLength(0);
                records.add(record);
                record = new ArrayList<>();
                any = false;
            } else {
                field.append(c);
                any = true;
            }
        }
        if (any || field.length() > 0 || !record.isEmpty()) {
            record.add(field.toString());
            records.add(record);
        }
        return records;
    }

    // --- Values ---------------------------------------------------------------------

    private static long id(Long value) {
        return value == null ? 0L : value;
    }

    private static String str(String value) {
        return value == null ? "" : value;
    }

    private static String emptyToNull(String value) {
        return value.isEmpty() ? null : value;
    }

    private static String string(Object value) {
        if (value == null)
            return "";
        if (value instanceof Double d && d.isNaN())
            return "";
        return String.valueOf(value);
    }

    private static String optionalString(Object value) {
        String s = string(value);
        return s.isEmpty() || s.equals("None") ? null : s;
    }

    private static int integer(Object value, int fallback) {
        if (value instanceof Number n) {
            if (n instanceof Double d && d.isNaN())
                return fallback;
            return n.intValue();
        }
        String s = string(value).strip();
        if (s.isEmpty())
            return fallback;
        try {
            return (int) Double.parseDouble(s);
        } catch (NumberFormatException e) {
            return fallback;
        }
    }

    private static Long optionalId(Object value) {
        long parsed;
        if (value instanceof Number n) {
            if (n instanceof Double d && d.isNaN())
                return null;
            parsed = n.longValue();
        } else {
            String s = string(value).strip();
            if (s.isEmpty())
                return null;
            try {
                parsed = (long) Double.parseDouble(s);
            } catch (NumberFormatException e) {
                return null;
            }
        }
        return parsed == 0 || parsed == -1 ? null : parsed;
    }

    private static boolean bool(Object value) {
        if (value instanceof Boolean b)
            return b;
        if (value instanceof Number n)
            return n.doubleValue() != 0;
        String s = string(value).strip().toLowerCase();
        return s.equals("true") || s.equals("1") || s.equals("yes");
    }

    /** JSON of well key-value pairs with sorted keys, as Python's {@code json.dumps(sort_keys=True)}. */
    static String wellMetadataJson(Map<String, String> metadata) {
        if (metadata == null || metadata.isEmpty())
            return "{}";
        StringBuilder sb = new StringBuilder("{");
        boolean first = true;
        for (Map.Entry<String, String> e : new TreeMap<>(metadata).entrySet()) {
            if (!first)
                sb.append(", ");
            first = false;
            sb.append(GSON.toJson(e.getKey())).append(": ").append(GSON.toJson(e.getValue()));
        }
        return sb.append('}').toString();
    }

    private static Map<String, String> parseWellMetadata(String json) {
        Map<String, String> result = new LinkedHashMap<>();
        if (json == null || json.isBlank())
            return result;
        try {
            Map<String, Object> parsed = GSON.fromJson(json, new TypeToken<Map<String, Object>>() {}.getType());
            if (parsed != null)
                parsed.forEach((k, v) -> result.put(k, v == null ? "" : String.valueOf(v)));
        } catch (RuntimeException e) {
            // an unreadable cell is treated as no metadata
        }
        return result;
    }
}
