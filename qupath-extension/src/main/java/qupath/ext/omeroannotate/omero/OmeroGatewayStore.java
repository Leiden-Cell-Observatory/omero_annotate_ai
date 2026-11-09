package qupath.ext.omeroannotate.omero;

import omero.api.IQueryPrx;
import omero.api.IUpdatePrx;
import omero.api.RawFileStorePrx;
import omero.cmd.CmdCallbackI;
import omero.cmd.ERR;
import omero.cmd.Response;
import omero.gateway.Gateway;
import omero.gateway.LoginCredentials;
import omero.gateway.SecurityContext;
import omero.gateway.model.ExperimenterData;
import omero.gateway.util.Requests;
import omero.grid.BoolColumn;
import omero.grid.Column;
import omero.grid.Data;
import omero.grid.DoubleColumn;
import omero.grid.ImageColumn;
import omero.grid.LongColumn;
import omero.grid.RoiColumn;
import omero.grid.SharedResourcesPrx;
import omero.grid.StringColumn;
import omero.grid.TablePrx;
import omero.log.SimpleLogger;
import omero.model.ChecksumAlgorithmI;
import omero.model.DatasetAnnotationLinkI;
import omero.model.DatasetI;
import omero.model.FileAnnotation;
import omero.model.FileAnnotationI;
import omero.model.IObject;
import omero.model.Image;
import omero.model.ImageAnnotationLinkI;
import omero.model.ImageI;
import omero.model.LengthI;
import omero.model.OriginalFile;
import omero.model.OriginalFileI;
import omero.model.Pixels;
import omero.model.PlateAnnotationLinkI;
import omero.model.PlateI;
import omero.model.PolygonI;
import omero.model.ProjectAnnotationLinkI;
import omero.model.ProjectI;
import omero.model.Roi;
import omero.model.RoiI;
import omero.model.ScreenAnnotationLinkI;
import omero.model.ScreenI;
import omero.model.enums.ChecksumAlgorithmSHA1160;
import omero.model.enums.UnitsLength;
import omero.sys.ParametersI;
import qupath.ext.omeroannotate.core.Bundle;
import qupath.ext.omeroannotate.core.RemoteStore;
import qupath.ext.omeroannotate.core.Row;
import qupath.ext.omeroannotate.core.RowCodec;

import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.Function;

import static omero.rtypes.rint;
import static omero.rtypes.rlong;
import static omero.rtypes.rstring;

/**
 * {@link RemoteStore} on a real OMERO server, through the OMERO Java gateway (Ice).
 * <p>
 * Tables, files and ROIs are written exactly as the Python package writes them, so a set
 * can be opened from either side. This is the only class that names OMERO types; it is
 * loaded only when the OMERO Java libraries are installed (see {@link OmeroAccess}).
 */
public final class OmeroGatewayStore implements RemoteStore {

    private static final String NS_BULK_ANNOTATIONS = "openmicroscopy.org/omero/bulk_annotations";
    private static final String TABLE_MIMETYPE = "OMERO.tables";
    private static final Map<String, String> ALL_GROUPS = Map.of("omero.group", "-1");

    /** Width of each string column of a new table (as the Python package creates them). */
    private static final Map<String, Integer> STRING_WIDTHS = Map.ofEntries(
            Map.entry("image_uri", 1024), Map.entry("image_name", 512), Map.entry("annotation_id", 256),
            Map.entry("category", 16), Map.entry("annotation_type", 64), Map.entry("annotation_created_at", 64),
            Map.entry("annotation_updated_at", 64), Map.entry("well", 32), Map.entry("field", 64),
            Map.entry("well_metadata", 4096), Map.entry("input_digest", 64), Map.entry("mask_digest", 64),
            Map.entry("input_iscc", 64), Map.entry("mask_iscc", 64), Map.entry("claimed_by", 160),
            Map.entry("claimed_at", 40));
    private static final int DEFAULT_STRING_WIDTH = 256;

    private final Gateway gateway;
    private final ExperimenterData user;
    private final String host;
    private final int port;
    private final boolean ownsSession;

    private OmeroGatewayStore(Gateway gateway, ExperimenterData user, String host, int port, boolean ownsSession) {
        this.gateway = gateway;
        this.user = user;
        this.host = host;
        this.port = port;
        this.ownsSession = ownsSession;
    }

    /**
     * Connect to an OMERO server.
     *
     * @param user     login name, or the ID of an existing session (e.g. the OMERO extension's)
     * @param password password, or the same session ID
     */
    public static OmeroGatewayStore connect(String host, int port, String user, String password) throws Exception {
        Gateway gateway = new Gateway(new SimpleLogger());
        try {
            ExperimenterData experimenter = gateway.connect(new LoginCredentials(user, password, host, port));
            if (experimenter == null)
                throw new IllegalStateException("OMERO did not accept the login");
            return new OmeroGatewayStore(gateway, experimenter, host, port, !user.equals(password));
        } catch (Exception e) {
            try {
                gateway.close();
            } catch (Exception ignored) {
                // nothing to close
            }
            throw e;
        }
    }

    @Override
    public String serverSlug() {
        return Bundle.serverSlug(host, port);
    }

    @Override
    public String userName() {
        return user.getUserName();
    }

    public long userId() {
        return user.getId();
    }

    public boolean isConnected() {
        try {
            return gateway.isConnected();
        } catch (Exception e) {
            return false;
        }
    }

    @Override
    public void close() {
        try {
            if (ownsSession)
                gateway.disconnect();
            else
                gateway.closeSessionOnExit(defaultContext(), false);
        } catch (Exception ignored) {
            // already closed
        }
        try {
            gateway.close();
        } catch (Exception ignored) {
            // already closed
        }
    }

    // --- Contexts and queries -------------------------------------------------------

    private SecurityContext defaultContext() {
        return new SecurityContext(user.getGroupId());
    }

    private IQueryPrx query() throws Exception {
        return gateway.getQueryService(defaultContext());
    }

    private List<IObject> find(String hql, ParametersI parameters) throws Exception {
        return query().findAllByQuery(hql, parameters, ALL_GROUPS);
    }

    private IObject findOne(String hql, ParametersI parameters) throws Exception {
        List<IObject> found = find(hql, parameters);
        return found.isEmpty() ? null : found.get(0);
    }

    private static long groupOf(IObject object) {
        return object.getDetails().getGroup().getId().getValue();
    }

    /** The context (group) an object of {@code type} lives in. */
    private SecurityContext contextOf(String type, long id) throws Exception {
        IObject object = findOne("select o from " + type + " o where o.id = :id", new ParametersI().addId(id));
        if (object == null)
            throw new IllegalArgumentException(type + " " + id + " was not found on OMERO (or you may not see it)");
        return new SecurityContext(groupOf(object));
    }

    private static String modelType(String containerType) {
        return switch (containerType.toLowerCase(Locale.ROOT)) {
            case "dataset" -> "Dataset";
            case "project" -> "Project";
            case "plate" -> "Plate";
            case "screen" -> "Screen";
            case "image" -> "Image";
            default -> throw new IllegalArgumentException("Unknown container type: " + containerType);
        };
    }

    private FileAnnotation tableAnnotation(long tableId) throws Exception {
        FileAnnotation annotation = (FileAnnotation) findOne(
                "select fa from FileAnnotation fa join fetch fa.file where fa.id = :id",
                new ParametersI().addId(tableId));
        if (annotation == null)
            throw new IllegalArgumentException("Tracking table " + tableId + " was not found (or you may not see it)");
        String description = annotation.getDescription() == null ? "" : annotation.getDescription().getValue();
        if (!description.contains(TABLE_MARKER))
            throw new UnsupportedOperationException("Table " + tableId + " was not written by omero-annotate-ai "
                    + "0.5 or later. Older tracking tables are not supported; start a new training set.");
        return annotation;
    }

    private interface TableAction<T> {
        T run(TablePrx table) throws Exception;
    }

    private <T> T withTable(long tableId, TableAction<T> action) throws Exception {
        FileAnnotation annotation = tableAnnotation(tableId);
        SecurityContext context = new SecurityContext(groupOf(annotation));
        SharedResourcesPrx resources = gateway.getSharedResources(context);
        TablePrx table = resources.openTable(new OriginalFileI(annotation.getFile().getId().getValue(), false));
        if (table == null)
            throw new IllegalStateException("Could not open tracking table " + tableId);
        try {
            return action.run(table);
        } finally {
            table.close();
        }
    }

    // --- Table columns --------------------------------------------------------------

    private static String omeroName(String column) {
        return column.equals("image_id") ? "Image" : column.equals("roi_id") ? "Roi" : column;
    }

    private static String cellName(String omeroName) {
        return omeroName.equals("Image") ? "image_id" : omeroName.equals("Roi") ? "roi_id" : omeroName;
    }

    private static Column[] newColumns(List<Row> rows) {
        List<Map<String, Object>> cells = new ArrayList<>();
        for (Row row : rows)
            cells.add(RowCodec.toCells(row));
        Column[] columns = new Column[RowCodec.COLUMNS.size()];
        for (int c = 0; c < columns.length; c++) {
            String name = RowCodec.COLUMNS.get(c);
            Column column;
            if (name.equals("image_id"))
                column = new ImageColumn("Image", "Annotated OMERO image", null);
            else if (name.equals("roi_id"))
                column = new RoiColumn("Roi", "ROI holding the annotation", null);
            else if (RowCodec.BOOL_COLUMNS.contains(name))
                column = new BoolColumn(name, "", null);
            else if (RowCodec.LONG_COLUMNS.contains(name))
                column = new LongColumn(name, "", null);
            else
                column = new StringColumn(name, "", STRING_WIDTHS.getOrDefault(name, DEFAULT_STRING_WIDTH), null);
            columns[c] = column;
        }
        fill(columns, cells);
        return columns;
    }

    /** Put the cells of some rows into table columns, checking string widths first. */
    private static void fill(Column[] columns, List<Map<String, Object>> rows) {
        int n = rows.size();
        for (Column column : columns) {
            String name = cellName(column.name);
            if (n > 0 && !rows.get(0).containsKey(name))
                throw new IllegalArgumentException("Rows lack the table column " + column.name);
            if (column instanceof StringColumn sc) {
                String[] values = new String[n];
                for (int i = 0; i < n; i++) {
                    Object value = rows.get(i).get(name);
                    values[i] = value == null ? "" : value.toString();
                    if (sc.size > 0 && values[i].length() > sc.size)
                        throw new IllegalArgumentException("Column '" + name + "' allows " + sc.size
                                + " characters, but row " + rows.get(i).get("annotation_id") + " has "
                                + values[i].length() + ": " + values[i].substring(0, Math.min(60, values[i].length())) + "...");
                }
                sc.values = values;
            } else if (column instanceof BoolColumn bc) {
                boolean[] values = new boolean[n];
                for (int i = 0; i < n; i++)
                    values[i] = Boolean.TRUE.equals(rows.get(i).get(name));
                bc.values = values;
            } else if (column instanceof DoubleColumn dc) {
                double[] values = new double[n];
                for (int i = 0; i < n; i++)
                    values[i] = ((Number) rows.get(i).get(name)).doubleValue();
                dc.values = values;
            } else {
                long[] values = new long[n];
                for (int i = 0; i < n; i++)
                    values[i] = ((Number) rows.get(i).get(name)).longValue();
                if (column instanceof LongColumn lc)
                    lc.values = values;
                else if (column instanceof ImageColumn ic)
                    ic.values = values;
                else if (column instanceof RoiColumn rc)
                    rc.values = values;
                else
                    throw new IllegalStateException("Unsupported table column type: " + column.getClass().getSimpleName());
            }
        }
    }

    private static Object cell(Column column, int i) {
        if (column instanceof StringColumn c)
            return c.values[i];
        if (column instanceof BoolColumn c)
            return c.values[i];
        if (column instanceof LongColumn c)
            return c.values[i];
        if (column instanceof ImageColumn c)
            return c.values[i];
        if (column instanceof RoiColumn c)
            return c.values[i];
        if (column instanceof DoubleColumn c)
            return c.values[i];
        return null;
    }

    private static void setCell(Column column, int i, Object value) {
        if (column instanceof StringColumn c) {
            String text = value == null ? "" : value.toString();
            if (c.size > 0 && text.length() > c.size)
                throw new IllegalArgumentException("Column '" + c.name + "' allows " + c.size + " characters");
            c.values[i] = text;
        } else if (column instanceof BoolColumn c)
            c.values[i] = Boolean.TRUE.equals(value);
        else if (column instanceof LongColumn c)
            c.values[i] = ((Number) value).longValue();
        else if (column instanceof ImageColumn c)
            c.values[i] = ((Number) value).longValue();
        else if (column instanceof RoiColumn c)
            c.values[i] = ((Number) value).longValue();
        else if (column instanceof DoubleColumn c)
            c.values[i] = ((Number) value).doubleValue();
    }

    private static List<Map<String, Object>> toRows(Data data) {
        List<Map<String, Object>> rows = new ArrayList<>();
        int n = data.rowNumbers == null ? 0 : data.rowNumbers.length;
        for (int i = 0; i < n; i++) {
            Map<String, Object> row = new LinkedHashMap<>();
            for (Column column : data.columns)
                row.put(cellName(column.name), cell(column, i));
            rows.add(row);
        }
        return rows;
    }

    private static long[] range(int n) {
        long[] indices = new long[n];
        for (int i = 0; i < n; i++)
            indices[i] = i;
        return indices;
    }

    /** Row number in the table of each annotation ID it holds. */
    private static Map<String, Long> rowNumbers(TablePrx table) throws Exception {
        Column[] headers = table.getHeaders();
        int key = -1;
        for (int i = 0; i < headers.length; i++)
            if (headers[i].name.equals("annotation_id"))
                key = i;
        if (key < 0)
            throw new UnsupportedOperationException("This tracking table has no annotation_id column");
        Map<String, Long> positions = new HashMap<>();
        long n = table.getNumberOfRows();
        if (n == 0)
            return positions;
        String[] ids = ((StringColumn) table.read(new long[]{key}, 0, n).columns[0]).values;
        for (int i = 0; i < ids.length; i++)
            positions.put(ids[i], (long) i);
        return positions;
    }

    // --- Tracking table -------------------------------------------------------------

    @Override
    public long createTable(String title, List<Row> rows, String containerType, List<Long> containerIds)
            throws Exception {
        if (containerIds.isEmpty())
            throw new IllegalArgumentException("A tracking table needs at least one container");
        String type = modelType(containerType);
        SecurityContext context = contextOf(type, containerIds.get(0));
        Column[] columns = newColumns(rows);
        SharedResourcesPrx resources = gateway.getSharedResources(context);
        long repository = resources.repositories().descriptions.get(0).getId().getValue();
        TablePrx table = resources.newTable(repository, title);
        if (table == null)
            throw new IllegalStateException("OMERO could not create a table (is OMERO.tables running?)");
        long fileId;
        try {
            table.initialize(columns);
            if (!rows.isEmpty())
                table.addData(columns);
            fileId = table.getOriginalFile().getId().getValue();
        } finally {
            table.close();
        }
        IUpdatePrx update = gateway.getUpdateService(context);
        FileAnnotationI annotation = new FileAnnotationI();
        annotation.setNs(rstring(NS_BULK_ANNOTATIONS));
        annotation.setDescription(rstring(TABLE_MARKER));
        // Reference the file unloaded: the table service already wrote it.
        annotation.setFile(new OriginalFileI(fileId, false));
        long tableId = update.saveAndReturnObject(annotation).getId().getValue();
        for (long containerId : containerIds)
            link(update, type, containerId, tableId);
        return tableId;
    }

    private static void link(IUpdatePrx update, String type, long parentId, long annotationId) throws Exception {
        update.saveAndReturnObject(newLink(type, parentId, new FileAnnotationI(annotationId, false)));
    }

    /** A link attaching a file annotation (saved already, or new) to an object. */
    private static IObject newLink(String type, long parentId, FileAnnotationI child) {
        switch (type) {
            case "Dataset" -> {
                DatasetAnnotationLinkI l = new DatasetAnnotationLinkI();
                l.setParent(new DatasetI(parentId, false));
                l.setChild(child);
                return l;
            }
            case "Project" -> {
                ProjectAnnotationLinkI l = new ProjectAnnotationLinkI();
                l.setParent(new ProjectI(parentId, false));
                l.setChild(child);
                return l;
            }
            case "Plate" -> {
                PlateAnnotationLinkI l = new PlateAnnotationLinkI();
                l.setParent(new PlateI(parentId, false));
                l.setChild(child);
                return l;
            }
            case "Screen" -> {
                ScreenAnnotationLinkI l = new ScreenAnnotationLinkI();
                l.setParent(new ScreenI(parentId, false));
                l.setChild(child);
                return l;
            }
            case "Image" -> {
                ImageAnnotationLinkI l = new ImageAnnotationLinkI();
                l.setParent(new ImageI(parentId, false));
                l.setChild(child);
                return l;
            }
            default -> throw new IllegalArgumentException("Cannot attach a file to a " + type);
        }
    }

    private static long childId(IObject link) {
        if (link instanceof DatasetAnnotationLinkI l)
            return l.getChild().getId().getValue();
        if (link instanceof ProjectAnnotationLinkI l)
            return l.getChild().getId().getValue();
        if (link instanceof PlateAnnotationLinkI l)
            return l.getChild().getId().getValue();
        if (link instanceof ScreenAnnotationLinkI l)
            return l.getChild().getId().getValue();
        if (link instanceof ImageAnnotationLinkI l)
            return l.getChild().getId().getValue();
        throw new IllegalStateException("Unexpected link type " + link.getClass().getSimpleName());
    }

    @Override
    public List<Map<String, Object>> readTable(long tableId) throws Exception {
        return withTable(tableId, table -> {
            long n = table.getNumberOfRows();
            if (n == 0)
                return new ArrayList<>();
            return toRows(table.read(range(table.getHeaders().length), 0, n));
        });
    }

    @Override
    public List<Map<String, Object>> readRows(long tableId, List<String> annotationIds) throws Exception {
        return withTable(tableId, table -> {
            Map<String, Long> positions = rowNumbers(table);
            List<Long> wanted = new ArrayList<>();
            for (String id : annotationIds)
                if (positions.containsKey(id))
                    wanted.add(positions.get(id));
            if (wanted.isEmpty())
                return new ArrayList<>();
            return toRows(table.readCoordinates(wanted.stream().mapToLong(Long::longValue).toArray()));
        });
    }

    @Override
    public void writeRows(long tableId, List<Row> rows) throws Exception {
        if (rows.isEmpty())
            return;
        withTable(tableId, table -> {
            Map<String, Long> positions = rowNumbers(table);
            List<Map<String, Object>> toUpdate = new ArrayList<>(), toAdd = new ArrayList<>();
            List<Long> numbers = new ArrayList<>();
            for (Row row : rows) {
                Map<String, Object> cells = RowCodec.toCells(row);
                Long position = positions.get(row.annotationId);
                if (position == null) {
                    toAdd.add(cells);
                } else {
                    toUpdate.add(cells);
                    numbers.add(position);
                }
            }
            if (!toUpdate.isEmpty()) {
                // readCoordinates returns the rows in the order asked for.
                Data data = table.readCoordinates(numbers.stream().mapToLong(Long::longValue).toArray());
                fill(data.columns, toUpdate);
                table.update(data);
            }
            if (!toAdd.isEmpty()) {
                Column[] headers = table.getHeaders();
                fill(headers, toAdd);
                table.addData(headers);
            }
            return null;
        });
    }

    @Override
    public List<String> updateCells(long tableId, List<String> annotationIds,
                                    Function<Map<String, Object>, Map<String, Object>> change) throws Exception {
        return withTable(tableId, table -> {
            Map<String, Long> positions = rowNumbers(table);
            List<String> keys = new ArrayList<>();
            List<Long> numbers = new ArrayList<>();
            for (String id : annotationIds) {
                if (positions.containsKey(id)) {
                    keys.add(id);
                    numbers.add(positions.get(id));
                }
            }
            List<String> changed = new ArrayList<>();
            if (keys.isEmpty())
                return changed;
            List<Map<String, Object>> seen = toRows(
                    table.readCoordinates(numbers.stream().mapToLong(Long::longValue).toArray()));
            List<Integer> planned = new ArrayList<>();
            List<Map<String, Object>> plannedCells = new ArrayList<>();
            java.util.TreeSet<String> names = new java.util.TreeSet<>();
            for (int i = 0; i < seen.size(); i++) {
                Map<String, Object> cells = change.apply(new LinkedHashMap<>(seen.get(i)));
                if (cells != null && !cells.isEmpty()) {
                    planned.add(i);
                    plannedCells.add(cells);
                    names.addAll(cells.keySet());
                }
            }
            if (planned.isEmpty())
                return changed;
            Column[] headers = table.getHeaders();
            long[] columnNumbers = new long[names.size()];
            List<String> ordered = new ArrayList<>(names);
            for (int c = 0; c < ordered.size(); c++) {
                columnNumbers[c] = -1;
                for (int h = 0; h < headers.length; h++)
                    if (cellName(headers[h].name).equals(ordered.get(c)))
                        columnNumbers[c] = h;
                if (columnNumbers[c] < 0)
                    throw new IllegalArgumentException("The table has no column " + ordered.get(c));
            }
            long[] rowNumbers = planned.stream().mapToLong(numbers::get).toArray();
            // Only the changed cells are written, and only if they still hold what was
            // read: a concurrent writer's cells are never overwritten with a stale copy.
            Data fresh = table.slice(columnNumbers, rowNumbers);
            Map<String, Column> byName = new HashMap<>();
            for (Column column : fresh.columns)
                byName.put(cellName(column.name), column);
            List<Integer> keep = new ArrayList<>();
            for (int i = 0; i < planned.size(); i++) {
                Map<String, Object> before = seen.get(planned.get(i));
                boolean unchanged = true;
                for (String name : ordered)
                    if (!java.util.Objects.equals(cell(byName.get(name), i), before.get(name)))
                        unchanged = false;
                if (!unchanged)
                    continue;
                for (String name : ordered)
                    if (plannedCells.get(i).containsKey(name))
                        setCell(byName.get(name), i, plannedCells.get(i).get(name));
                keep.add(i);
                changed.add(keys.get(planned.get(i)));
            }
            if (keep.isEmpty())
                return changed;
            if (keep.size() < planned.size()) {
                for (Column column : fresh.columns)
                    retain(column, keep);
                fresh.rowNumbers = keep.stream().mapToLong(i -> rowNumbers[i]).toArray();
            }
            table.update(fresh);
            return changed;
        });
    }

    private static void retain(Column column, List<Integer> keep) {
        int n = keep.size();
        if (column instanceof StringColumn c) {
            String[] values = new String[n];
            for (int i = 0; i < n; i++)
                values[i] = c.values[keep.get(i)];
            c.values = values;
        } else if (column instanceof BoolColumn c) {
            boolean[] values = new boolean[n];
            for (int i = 0; i < n; i++)
                values[i] = c.values[keep.get(i)];
            c.values = values;
        } else if (column instanceof LongColumn c) {
            c.values = keep.stream().mapToLong(i -> c.values[i]).toArray();
        } else if (column instanceof ImageColumn c) {
            c.values = keep.stream().mapToLong(i -> c.values[i]).toArray();
        } else if (column instanceof RoiColumn c) {
            c.values = keep.stream().mapToLong(i -> c.values[i]).toArray();
        }
    }

    // --- Files ----------------------------------------------------------------------

    private long uploadFile(SecurityContext context, byte[] bytes, String name, String mimetype, String ns,
                            String description, String parentType, long parentId) throws Exception {
        IUpdatePrx update = gateway.getUpdateService(context);
        OriginalFile file = new OriginalFileI();
        file.setName(rstring(name));
        file.setPath(rstring("/"));
        file.setSize(rlong(bytes.length));
        file.setMimetype(rstring(mimetype));
        ChecksumAlgorithmI hasher = new ChecksumAlgorithmI();
        hasher.setValue(rstring(ChecksumAlgorithmSHA1160.value));
        file.setHasher(hasher);
        file = (OriginalFile) update.saveAndReturnObject(file);
        RawFileStorePrx store = gateway.getRawFileService(context);
        try {
            store.setFileId(file.getId().getValue());
            int chunk = 1 << 20;
            for (int offset = 0; offset < bytes.length; offset += chunk) {
                int length = Math.min(chunk, bytes.length - offset);
                byte[] part = java.util.Arrays.copyOfRange(bytes, offset, offset + length);
                store.write(part, offset, length);
            }
            file = store.save();
        } finally {
            store.close();
        }
        FileAnnotationI annotation = new FileAnnotationI();
        annotation.setNs(rstring(ns));
        if (description != null)
            annotation.setDescription(rstring(description));
        annotation.setFile(new OriginalFileI(file.getId().getValue(), false));
        // The annotation is saved together with its link, so there is never an attached
        // file that hangs on nothing (and that a later search by tag could not find).
        IObject saved = update.saveAndReturnObject(newLink(parentType, parentId, annotation));
        return childId(saved);
    }

    /** File annotations with a namespace on an object, with their files loaded. */
    private List<FileAnnotation> fileAnnotations(String type, long id, String ns) throws Exception {
        List<FileAnnotation> result = new ArrayList<>();
        for (IObject object : find("select fa from FileAnnotation fa join fetch fa.file where fa.ns = :ns and fa.id in "
                + "(select l.child.id from " + type + "AnnotationLink l where l.parent.id = :id) order by fa.id",
                new ParametersI().addId(id).add("ns", rstring(ns))))
            result.add((FileAnnotation) object);
        return result;
    }

    @Override
    public byte[] downloadFile(long fileAnnotationId) throws Exception {
        FileAnnotation annotation = (FileAnnotation) findOne(
                "select fa from FileAnnotation fa join fetch fa.file where fa.id = :id",
                new ParametersI().addId(fileAnnotationId));
        if (annotation == null || annotation.getFile() == null)
            return null;
        RawFileStorePrx store = gateway.getRawFileService(new SecurityContext(groupOf(annotation)));
        try {
            store.setFileId(annotation.getFile().getId().getValue());
            long size = store.size();
            byte[] bytes = new byte[(int) size];
            int chunk = 1 << 20;
            for (int offset = 0; offset < size; offset += chunk) {
                byte[] part = store.read(offset, (int) Math.min(chunk, size - offset));
                System.arraycopy(part, 0, bytes, offset, part.length);
            }
            return bytes;
        } finally {
            store.close();
        }
    }

    private static String configFileName(long tableId) {
        return "annotation_config_table_" + tableId + ".yaml";
    }

    /** The containers a table is linked to: {@code (model type, id)}. */
    private List<Object[]> tableContainers(long tableId) throws Exception {
        List<Object[]> containers = new ArrayList<>();
        for (String type : List.of("Dataset", "Plate", "Project", "Screen"))
            for (IObject parent : find("select l.parent from " + type + "AnnotationLink l where l.child.id = :id",
                    new ParametersI().addId(tableId)))
                containers.add(new Object[]{type, parent.getId().getValue()});
        return containers;
    }

    @Override
    public String readConfig(long tableId) throws Exception {
        String name = configFileName(tableId);
        FileAnnotation newest = null;
        for (Object[] container : tableContainers(tableId))
            for (FileAnnotation annotation : fileAnnotations((String) container[0], (Long) container[1], CONFIG_NS))
                if (annotation.getFile() != null && name.equals(annotation.getFile().getName().getValue())
                        && (newest == null || annotation.getId().getValue() > newest.getId().getValue()))
                    newest = annotation;
        if (newest == null)
            return null;
        byte[] bytes = downloadFile(newest.getId().getValue());
        return bytes == null ? null : new String(bytes, StandardCharsets.UTF_8);
    }

    @Override
    public long replaceConfig(long tableId, String yaml, String containerType, long containerId) throws Exception {
        String type = modelType(containerType);
        String name = configFileName(tableId);
        SecurityContext context = contextOf(type, containerId);
        // The new file first; only older copies are deleted after, so the set is never
        // without a config and two computers publishing at once keep the newest.
        long newId = uploadFile(context, yaml.getBytes(StandardCharsets.UTF_8), name, "application/x-yaml",
                CONFIG_NS, null, type, containerId);
        for (FileAnnotation old : fileAnnotations(type, containerId, CONFIG_NS)) {
            long oldId = old.getId().getValue();
            if (oldId < newId && old.getFile() != null && name.equals(old.getFile().getName().getValue())) {
                try {
                    delete("FileAnnotation", List.of(oldId));
                } catch (Exception ignored) {
                    // already gone or not ours: harmless, readers take the newest
                }
            }
        }
        return newId;
    }

    // --- Uploads --------------------------------------------------------------------

    @Override
    public long uploadLabel(long imageId, byte[] tiff, String fileName, String description) throws Exception {
        return uploadFile(contextOf("Image", imageId), tiff, fileName, "image/tiff", LABELS_NS, description,
                "Image", imageId);
    }

    @Override
    public Long uploadRois(long imageId, List<Polygon> polygons, int z, int t, int channel, String name,
                           String description) throws Exception {
        if (polygons.isEmpty())
            return null;
        RoiI roi = new RoiI();
        roi.setImage(new ImageI(imageId, false));
        roi.setName(rstring(name));
        roi.setDescription(rstring(description));
        for (Polygon polygon : polygons) {
            StringBuilder points = new StringBuilder();
            double[] xy = polygon.xy();
            for (int i = 0; i + 1 < xy.length; i += 2) {
                if (i > 0)
                    points.append(' ');
                points.append(number(xy[i])).append(',').append(number(xy[i + 1]));
            }
            PolygonI shape = new PolygonI();
            shape.setPoints(rstring(points.toString()));
            shape.setTheZ(rint(Math.max(0, z)));
            shape.setTheT(rint(Math.max(0, t)));
            if (channel >= 0)
                shape.setTheC(rint(channel));
            shape.setTextValue(rstring("qupath.instance_segmentation"));
            shape.setStrokeColor(rint(0xFFFF00FF)); // yellow, opaque (RGBA)
            shape.setFillColor(rint(0xFFFF0020));
            shape.setStrokeWidth(new LengthI(1, UnitsLength.PIXEL));
            roi.addShape(shape);
        }
        IUpdatePrx update = gateway.getUpdateService(contextOf("Image", imageId));
        return update.saveAndReturnObject(roi).getId().getValue();
    }

    private static String number(double value) {
        return value == Math.rint(value) ? Long.toString((long) value) : String.format(Locale.ROOT, "%.2f", value);
    }

    private static String tagOf(String description) {
        return description == null ? "" : description.split(" \\| ", 2)[0].strip();
    }

    @Override
    public Tagged findTagged(long imageId, String tag) throws Exception {
        List<Long> labels = new ArrayList<>(), rois = new ArrayList<>();
        for (FileAnnotation annotation : fileAnnotations("Image", imageId, LABELS_NS))
            if (annotation.getDescription() != null && tag.equals(tagOf(annotation.getDescription().getValue())))
                labels.add(annotation.getId().getValue());
        for (IObject object : find("select r from Roi r where r.image.id = :id order by r.id",
                new ParametersI().addId(imageId))) {
            Roi roi = (Roi) object;
            if (roi.getDescription() != null && tag.equals(tagOf(roi.getDescription().getValue())))
                rois.add(roi.getId().getValue());
        }
        return new Tagged(labels, rois);
    }

    @Override
    public void delete(String kind, List<Long> ids) throws Exception {
        if (ids.isEmpty())
            return;
        String type = kind.equals("Roi") ? "Roi" : "Annotation";
        SecurityContext context = contextOf(type, ids.get(0));
        Response response;
        try {
            CmdCallbackI callback = gateway.submit(context, Requests.delete().target(type).id(ids).build());
            try {
                response = callback.loop(1200, 500);
            } finally {
                callback.close(true);
            }
        } catch (Exception e) {
            throw e;
        } catch (Throwable t) {
            throw new IllegalStateException("OMERO could not delete " + kind + " " + ids, t);
        }
        if (response instanceof ERR error)
            throw new IllegalStateException("OMERO could not delete " + kind + " " + ids + ": " + error.name);
    }

    // --- Browsing -------------------------------------------------------------------

    @Override
    public List<SetInfo> listSets(String containerType, long containerId) throws Exception {
        String type = modelType(containerType);
        List<SetInfo> sets = new ArrayList<>();
        for (IObject object : find("select fa from FileAnnotation fa join fetch fa.file join fetch fa.details.creationEvent "
                + "where fa.id in (select l.child.id from " + type + "AnnotationLink l where l.parent.id = :id) "
                + "order by fa.id desc", new ParametersI().addId(containerId))) {
            FileAnnotation annotation = (FileAnnotation) object;
            OriginalFile file = annotation.getFile();
            if (file == null || file.getMimetype() == null || !TABLE_MIMETYPE.equals(file.getMimetype().getValue()))
                continue;
            String description = annotation.getDescription() == null ? "" : annotation.getDescription().getValue();
            if (!description.contains(TABLE_MARKER))
                continue;
            String created = "";
            try {
                created = Instant.ofEpochMilli(annotation.getDetails().getCreationEvent().getTime().getValue())
                        .toString().substring(0, 10);
            } catch (RuntimeException ignored) {
                // the date is only shown
            }
            sets.add(new SetInfo(annotation.getId().getValue(), file.getName().getValue(), created));
        }
        return sets;
    }

    @Override
    public List<RemoteImage> listImages(String containerType, long containerId) throws Exception {
        String type = modelType(containerType);
        String where = switch (type) {
            case "Dataset" -> "i.id in (select l.child.id from DatasetImageLink l where l.parent.id = :id)";
            case "Project" -> "i.id in (select l.child.id from DatasetImageLink l where l.parent.id in "
                    + "(select pl.child.id from ProjectDatasetLink pl where pl.parent.id = :id))";
            case "Image" -> "i.id = :id";
            default -> throw new UnsupportedOperationException("New training sets in QuPath take their images "
                    + "from a dataset or project (plates and screens: use the Python package)");
        };
        List<RemoteImage> images = new ArrayList<>();
        for (IObject object : find("select i from Image i join fetch i.pixels where " + where + " order by i.id",
                new ParametersI().addId(containerId))) {
            Image image = (Image) object;
            Pixels pixels = image.getPrimaryPixels();
            images.add(new RemoteImage(image.getId().getValue(), image.getName().getValue(),
                    pixels.getSizeX().getValue(), pixels.getSizeY().getValue(), pixels.getSizeZ().getValue(),
                    pixels.getSizeT().getValue(), pixels.getSizeC().getValue()));
        }
        return images;
    }

    @Override
    public List<String[]> containersOfImage(long imageId) throws Exception {
        List<String[]> containers = new ArrayList<>();
        for (IObject dataset : find("select l.parent from DatasetImageLink l where l.child.id = :id",
                new ParametersI().addId(imageId))) {
            long datasetId = dataset.getId().getValue();
            containers.add(new String[]{"dataset", Long.toString(datasetId), ((omero.model.Dataset) dataset).getName().getValue()});
            for (IObject project : find("select l.parent from ProjectDatasetLink l where l.child.id = :id",
                    new ParametersI().addId(datasetId)))
                containers.add(new String[]{"project", Long.toString(project.getId().getValue()),
                        ((omero.model.Project) project).getName().getValue()});
        }
        return containers;
    }
}
