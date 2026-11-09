package qupath.ext.omeroannotate.core;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/** An in-memory OMERO: one tracking table per ID, uploads by ID. Any call can be made to fail. */
class FakeRemoteStore implements RemoteStore {

    final Map<Long, List<Map<String, Object>>> tables = new LinkedHashMap<>();
    final Map<Long, String> configs = new LinkedHashMap<>();
    final Map<Long, byte[]> files = new LinkedHashMap<>();
    final Map<Long, String> fileDescriptions = new LinkedHashMap<>();
    final Map<Long, Long> fileImages = new LinkedHashMap<>();
    final Map<Long, String> roiDescriptions = new LinkedHashMap<>();
    final Map<Long, Long> roiImages = new LinkedHashMap<>();
    final Map<Long, Integer> roiPolygons = new LinkedHashMap<>();
    long nextId = 100;
    String user = "annot";
    /** Name of a method that throws on its next call (then cleared), or null. */
    String failOnce;
    /** Name of a method that always throws, or null. */
    String failAlways;

    private void maybeFail(String method) throws Exception {
        if (method.equals(failAlways))
            throw new java.io.IOException("OMERO is not reachable (" + method + ")");
        if (method.equals(failOnce)) {
            failOnce = null;
            throw new java.io.IOException("OMERO is not reachable (" + method + ")");
        }
    }

    @Override public String serverSlug() { return "fake-4064"; }
    @Override public String userName() { return user; }

    @Override
    public long createTable(String title, List<Row> rows, String containerType, List<Long> containerIds) throws Exception {
        maybeFail("createTable");
        long id = nextId++;
        List<Map<String, Object>> table = new ArrayList<>();
        for (Row row : rows)
            table.add(RowCodec.toCells(row));
        tables.put(id, table);
        return id;
    }

    @Override
    public List<Map<String, Object>> readTable(long tableId) throws Exception {
        maybeFail("readTable");
        List<Map<String, Object>> copy = new ArrayList<>();
        for (Map<String, Object> row : tables.get(tableId))
            copy.add(new LinkedHashMap<>(row));
        return copy;
    }

    @Override
    public List<Map<String, Object>> readRows(long tableId, List<String> annotationIds) throws Exception {
        maybeFail("readRows");
        List<Map<String, Object>> result = new ArrayList<>();
        for (String id : annotationIds)
            for (Map<String, Object> row : tables.get(tableId))
                if (row.get("annotation_id").equals(id))
                    result.add(new LinkedHashMap<>(row));
        return result;
    }

    @Override
    public void writeRows(long tableId, List<Row> rows) throws Exception {
        maybeFail("writeRows");
        List<Map<String, Object>> table = tables.get(tableId);
        for (Row row : rows) {
            Map<String, Object> cells = RowCodec.toCells(row);
            boolean found = false;
            for (int i = 0; i < table.size(); i++) {
                if (table.get(i).get("annotation_id").equals(row.annotationId)) {
                    table.set(i, cells);
                    found = true;
                }
            }
            if (!found)
                table.add(cells);
        }
    }

    @Override
    public List<String> updateCells(long tableId, List<String> annotationIds,
                                    Function<Map<String, Object>, Map<String, Object>> change) throws Exception {
        maybeFail("updateCells");
        List<String> changed = new ArrayList<>();
        for (String id : annotationIds) {
            for (Map<String, Object> row : tables.get(tableId)) {
                if (!row.get("annotation_id").equals(id))
                    continue;
                Map<String, Object> cells = change.apply(new LinkedHashMap<>(row));
                if (cells != null && !cells.isEmpty()) {
                    row.putAll(cells);
                    changed.add(id);
                }
            }
        }
        return changed;
    }

    @Override public String readConfig(long tableId) throws Exception {
        maybeFail("readConfig");
        return configs.get(tableId);
    }

    @Override
    public long replaceConfig(long tableId, String yaml, String containerType, long containerId) throws Exception {
        maybeFail("replaceConfig");
        configs.put(tableId, yaml);
        return nextId++;
    }

    @Override
    public long uploadLabel(long imageId, byte[] tiff, String fileName, String description) throws Exception {
        maybeFail("uploadLabel");
        long id = nextId++;
        files.put(id, tiff);
        fileDescriptions.put(id, description);
        fileImages.put(id, imageId);
        return id;
    }

    @Override
    public Long uploadRois(long imageId, List<Polygon> polygons, int z, int t, int channel, String name,
                           String description) throws Exception {
        maybeFail("uploadRois");
        if (polygons.isEmpty())
            return null;
        long id = nextId++;
        roiDescriptions.put(id, description);
        roiImages.put(id, imageId);
        roiPolygons.put(id, polygons.size());
        return id;
    }

    @Override
    public Tagged findTagged(long imageId, String tag) throws Exception {
        maybeFail("findTagged");
        List<Long> labels = new ArrayList<>(), rois = new ArrayList<>();
        fileDescriptions.forEach((id, d) -> {
            if (fileImages.get(id) == imageId && d.split(" \\| ", 2)[0].strip().equals(tag))
                labels.add(id);
        });
        roiDescriptions.forEach((id, d) -> {
            if (roiImages.get(id) == imageId && d.split(" \\| ", 2)[0].strip().equals(tag))
                rois.add(id);
        });
        return new Tagged(labels, rois);
    }

    @Override public byte[] downloadFile(long fileAnnotationId) throws Exception {
        maybeFail("downloadFile");
        return files.get(fileAnnotationId);
    }

    @Override
    public void delete(String kind, List<Long> ids) throws Exception {
        maybeFail("delete");
        for (long id : ids) {
            if (kind.equals("Roi")) {
                roiDescriptions.remove(id);
                roiImages.remove(id);
                roiPolygons.remove(id);
            } else {
                files.remove(id);
                fileDescriptions.remove(id);
                fileImages.remove(id);
            }
        }
    }

    @Override public List<SetInfo> listSets(String containerType, long containerId) {
        List<SetInfo> sets = new ArrayList<>();
        tables.keySet().forEach(id -> sets.add(new SetInfo(id, "set " + id, "")));
        return sets;
    }

    @Override public List<RemoteImage> listImages(String containerType, long containerId) { return List.of(); }
    @Override public List<String[]> containersOfImage(long imageId) { return List.of(); }
    @Override public void close() {}

    Map<String, Object> tableRow(long tableId, String annotationId) {
        for (Map<String, Object> row : tables.get(tableId))
            if (row.get("annotation_id").equals(annotationId))
                return row;
        return null;
    }
}
