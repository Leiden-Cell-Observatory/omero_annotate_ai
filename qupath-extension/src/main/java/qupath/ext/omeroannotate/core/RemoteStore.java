package qupath.ext.omeroannotate.core;

import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * Everything a training set reads from and writes to OMERO: its tracking table, its
 * config file, and the label files and ROIs of annotated units.
 * <p>
 * The interface names no OMERO class, so the training-set logic can be tested with an
 * in-memory store and QuPath runs without the OMERO Java libraries for local sets.
 */
public interface RemoteStore extends AutoCloseable {

    /** Namespace of label files attached to images. */
    String LABELS_NS = "openmicroscopy.org/omero/annotate/labels";
    /** Namespace of the config file attached to the set's container. */
    String CONFIG_NS = "openmicroscopy.org/omero/annotate/config";
    /** Marker in the description of a tracking table's file annotation. */
    String TABLE_MARKER = "omero-annotate-ai table v3";

    /** A training set as an "open" dialog lists it. */
    record SetInfo(long tableId, String name, String created) {}

    /** An OMERO image a new set may include. */
    record RemoteImage(long id, String name, int width, int height, int sizeZ, int sizeT, int sizeC) {}

    /** A polygon of one object, in full-image pixel coordinates. */
    record Polygon(double[] xy) {}

    /** Label files and ROIs on an image carrying a unit's tag. */
    record Tagged(List<Long> labelIds, List<Long> roiIds) {}

    /** Folder name of the server, e.g. {@code omero.example.org-4064}. */
    String serverSlug();

    /** Login name of the connected user. */
    String userName();

    // --- Tracking table ---

    long createTable(String title, List<Row> rows, String containerType, List<Long> containerIds) throws Exception;

    /** All rows of a tracking table, as cells by column name (see {@link RowCodec}). */
    List<Map<String, Object>> readTable(long tableId) throws Exception;

    /** The current rows for some annotation IDs. */
    List<Map<String, Object>> readRows(long tableId, List<String> annotationIds) throws Exception;

    /** Update the rows that exist (by annotation ID) and append the others. */
    void writeRows(long tableId, List<Row> rows) throws Exception;

    /**
     * Change some cells of some rows in one read-modify-write. {@code change} gets each
     * row as it is in the table now and returns the cells to set (or null to leave the
     * row alone); cells another writer changed meanwhile are not overwritten.
     *
     * @return the annotation IDs of the rows changed
     */
    List<String> updateCells(long tableId, List<String> annotationIds,
                             Function<Map<String, Object>, Map<String, Object>> change) throws Exception;

    // --- Config ---

    /** The YAML text of the set's config on OMERO, or null if it has none. */
    String readConfig(long tableId) throws Exception;

    /** Attach the config, replacing older copies. */
    long replaceConfig(long tableId, String yaml, String containerType, long containerId) throws Exception;

    // --- Uploads ---

    long uploadLabel(long imageId, byte[] tiff, String fileName, String description) throws Exception;

    /** Post one ROI holding a polygon per object; null for no polygons. */
    Long uploadRois(long imageId, List<Polygon> polygons, int z, int t, int channel,
                    String name, String description) throws Exception;

    Tagged findTagged(long imageId, String tag) throws Exception;

    /** The bytes of an attached file, or null if it no longer exists. */
    byte[] downloadFile(long fileAnnotationId) throws Exception;

    /** Delete objects; {@code kind} is {@code "FileAnnotation"} or {@code "Roi"}. */
    void delete(String kind, List<Long> ids) throws Exception;

    // --- Browsing ---

    /** The training sets tracked on a container, newest first. */
    List<SetInfo> listSets(String containerType, long containerId) throws Exception;

    /** The images of a dataset or project. */
    List<RemoteImage> listImages(String containerType, long containerId) throws Exception;

    /** The datasets (and their projects) an image is in: {@code (type, id, name)}. */
    List<String[]> containersOfImage(long imageId) throws Exception;

    @Override
    void close();
}
