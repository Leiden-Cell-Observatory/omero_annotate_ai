package qupath.ext.omeroannotate.core;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.Locale;
import java.util.UUID;

/**
 * Where a training set's files live on disk. The layout is the Python package's:
 *
 * <pre>
 * &lt;home&gt;/&lt;server&gt;/&lt;table id&gt;_&lt;name&gt;/     a set tracked in an OMERO table
 * &lt;home&gt;/local/&lt;id&gt;_&lt;name&gt;/                a set kept on this computer only
 *     config.yaml
 *     tracking_table.csv
 *     annotation_input/&lt;annotation id&gt;.tif
 *     annotation_output/&lt;annotation id&gt;_mask.tif
 *     qupath/                               the QuPath project of the set (QuPath only)
 * </pre>
 */
public final class Bundle {

    public static final String HOME_ENV = "OMERO_ANNOTATE_AI_HOME";
    public static final String CONFIG = "config.yaml";
    public static final String TABLE = "tracking_table.csv";

    private Bundle() {}

    /** The folder holding all sets: {@code $OMERO_ANNOTATE_AI_HOME}, else {@code ~/omero_annotate_ai}. */
    public static Path defaultHome() {
        String env = System.getenv(HOME_ENV);
        if (env != null && !env.isBlank())
            return expand(env);
        return Path.of(System.getProperty("user.home"), "omero_annotate_ai");
    }

    private static Path expand(String path) {
        if (path.equals("~") || path.startsWith("~/") || path.startsWith("~\\"))
            return Path.of(System.getProperty("user.home") + path.substring(1));
        return Path.of(path);
    }

    /** Lower-case, filesystem-safe: anything but letters, digits, '.' and '_' becomes '-'. */
    public static String slug(String text) {
        String s = text.toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9._]+", "-");
        s = s.replaceAll("^[-.]+", "").replaceAll("[-.]+$", "");
        return s.isEmpty() ? "unnamed" : s;
    }

    /** A folder name for an OMERO server, e.g. {@code omero.example.org-4064}. */
    public static String serverSlug(String host, int port) {
        String h = host.strip().toLowerCase(Locale.ROOT).replaceAll("^[a-z]+://", "");
        while (h.endsWith("/"))
            h = h.substring(0, h.length() - 1);
        return slug(port > 0 ? h + "-" + port : h);
    }

    public static Path omeroBundle(Path home, String server, long tableId, String name) {
        return home.resolve(server).resolve(tableId + "_" + slug(name));
    }

    public static Path localBundle(Path home, String localId, String name) {
        return home.resolve("local").resolve(localId + "_" + slug(name));
    }

    /** ID of a new set kept only on disk. */
    public static String newLocalId() {
        return UUID.randomUUID().toString().replace("-", "").substring(0, 8);
    }

    public static Path inputPath(Path bundle, String annotationId) {
        return bundle.resolve("annotation_input").resolve(annotationId + ".tif");
    }

    public static Path maskPath(Path bundle, String annotationId) {
        return bundle.resolve("annotation_output").resolve(annotationId + "_mask.tif");
    }

    /** Where a mask is kept that lost to another session's upload of the same unit. */
    public static Path conflictPath(Path bundle, String annotationId, int session) {
        return bundle.resolve("annotation_output").resolve("conflicts")
                .resolve(annotationId + "_session" + session + ".tif");
    }

    /** The QuPath project folder of a set. */
    public static Path projectDir(Path bundle) {
        return bundle.resolve("qupath");
    }

    /** Write {@code bytes} to {@code path} through a temporary file, so readers never see half a file. */
    public static void atomicWrite(Path path, byte[] bytes) throws IOException {
        Files.createDirectories(path.getParent());
        Path temp = path.resolveSibling("." + path.getFileName() + ".tmp");
        try {
            try (FileChannel channel = FileChannel.open(temp, StandardOpenOption.CREATE,
                    StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)) {
                channel.write(java.nio.ByteBuffer.wrap(bytes));
                channel.force(true);
            }
            try {
                Files.move(temp, path, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            } catch (java.nio.file.AtomicMoveNotSupportedException e) {
                Files.move(temp, path, StandardCopyOption.REPLACE_EXISTING);
            }
        } finally {
            Files.deleteIfExists(temp);
        }
    }
}
