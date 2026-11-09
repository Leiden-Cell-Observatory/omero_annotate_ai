package qupath.ext.omeroannotate.core;

import java.net.InetAddress;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;

/**
 * Identifiers shared with the Python package: image URIs, annotation IDs and the
 * (hashed) ID of this computer.
 */
public final class Ids {

    private static final String OMERO_PREFIX = "omero:image:";

    private Ids() {}

    /** URI of an OMERO image: {@code omero:image:<id>}. */
    public static String omeroUri(long imageId) {
        return OMERO_PREFIX + imageId;
    }

    /** OMERO image ID of {@code uri}, or null when it is not an OMERO URI. */
    public static Long parseOmeroUri(String uri) {
        if (uri == null || !uri.startsWith(OMERO_PREFIX))
            return null;
        String rest = uri.substring(OMERO_PREFIX.length());
        if (rest.isEmpty() || !rest.chars().allMatch(c -> c >= '0' && c <= '9'))
            return null;
        try {
            return Long.parseLong(rest);
        } catch (NumberFormatException e) {
            return null;
        }
    }

    /** Short stable key of an image URI: the first 8 hex characters of its SHA-1. */
    public static String uriKey(String uri) {
        return hex("SHA-1", uri.getBytes(StandardCharsets.UTF_8)).substring(0, 8);
    }

    /**
     * Annotation ID of a 2D unit: {@code <key>_<t>_<z>}, plus {@code _<patch>} for patches.
     *
     * @param patch index of the patch within its plane, or -1 for the whole plane
     */
    public static String annotationId(String uri, int t, int z, int patch) {
        String base = uriKey(uri) + "_" + t + "_" + z;
        return patch < 0 ? base : base + "_" + patch;
    }

    private static String hostId;

    /** Short, stable ID of this computer (a hash: the host name itself is not stored). */
    public static synchronized String hostId() {
        if (hostId == null)
            hostId = hex("SHA-256", hostName().getBytes(StandardCharsets.UTF_8)).substring(0, 8);
        return hostId;
    }

    /**
     * The computer's own name, as Python's {@code socket.gethostname()} gives it, so that
     * QuPath and napari on one computer count as the same computer. Asked from the
     * operating system; a name-server lookup is only the last resort.
     */
    private static String hostName() {
        // The "hostname" command answers like gethostname() on Linux, macOS and Windows.
        try {
            Process process = new ProcessBuilder("hostname").redirectErrorStream(true).start();
            String value = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8).strip();
            if (process.waitFor(5, java.util.concurrent.TimeUnit.SECONDS) && process.exitValue() == 0
                    && !value.isEmpty() && !value.contains("\n"))
                return value;
        } catch (Exception e) {
            // try the next source
        }
        try {
            java.nio.file.Path file = java.nio.file.Path.of("/etc/hostname");
            if (java.nio.file.Files.isReadable(file)) {
                String value = java.nio.file.Files.readString(file).strip();
                if (!value.isEmpty())
                    return value;
            }
        } catch (Exception e) {
            // try the next source
        }
        for (String variable : new String[]{"HOSTNAME", "COMPUTERNAME"}) {
            String value = System.getenv(variable);
            if (value != null && !value.isBlank())
                return value.strip();
        }
        try {
            return InetAddress.getLocalHost().getHostName();
        } catch (Exception e) {
            return "unknown-" + System.getProperty("user.name", "");
        }
    }

    static String hex(String algorithm, byte[] bytes) {
        try {
            return HexFormat.of().formatHex(MessageDigest.getInstance(algorithm).digest(bytes));
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }
}
