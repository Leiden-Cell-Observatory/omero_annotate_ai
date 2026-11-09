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

    /** Short, stable ID of this computer (a hash: the host name itself is not stored). */
    public static String hostId() {
        return hex("SHA-256", hostName().getBytes(StandardCharsets.UTF_8)).substring(0, 8);
    }

    private static String hostName() {
        try {
            return InetAddress.getLocalHost().getHostName();
        } catch (Exception e) {
            String env = System.getenv("COMPUTERNAME");
            if (env == null)
                env = System.getenv("HOSTNAME");
            return env == null ? "unknown" : env;
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
