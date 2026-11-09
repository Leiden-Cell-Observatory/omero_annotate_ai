package qupath.ext.omeroannotate.omero;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.omeroannotate.core.RemoteStore;
import qupath.ext.omeroannotate.core.TimeLimitedStore;
import qupath.lib.images.servers.ImageServer;

import java.lang.reflect.Method;
import java.net.URI;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * The bridge to OMERO inside QuPath.
 * <p>
 * Images are opened by the official QuPath OMERO extension; this class finds the servers
 * it is logged in to and joins the same login for the things that extension does not do
 * (tracking tables, label files). Nothing here names a class of the OMERO extension or of
 * the OMERO Java libraries directly, so the plugin loads without either of them.
 */
public final class OmeroAccess {

    private static final Logger logger = LoggerFactory.getLogger(OmeroAccess.class);
    private static final String CLIENT_CLASS = "qupath.ext.omero.core.Client";
    private static final String GATEWAY_CLASS = "omero.gateway.Gateway";
    /** How long one call to OMERO may take before it counts as failed. */
    private static final long CALL_SECONDS = 60;

    private OmeroAccess() {}

    /** An OMERO server the OMERO extension is logged in to. */
    public record Login(String webUri, String sessionId, List<String[]> addresses, String label) {
        @Override
        public String toString() {
            return label;
        }
    }

    private static Class<?> find(String name) {
        try {
            return Class.forName(name, false, OmeroAccess.class.getClassLoader());
        } catch (ClassNotFoundException | LinkageError e) {
            return null;
        }
    }

    /** Whether the official QuPath OMERO extension is installed. */
    public static boolean hasOmeroExtension() {
        return find(CLIENT_CLASS) != null;
    }

    /** Whether the OMERO Java libraries (needed for training sets kept on OMERO) are installed. */
    public static boolean hasOmeroJava() {
        return find(GATEWAY_CLASS) != null;
    }

    /** What is missing to use training sets on OMERO, in plain words; null if nothing is. */
    public static String missing() {
        if (!hasOmeroExtension())
            return "The QuPath OMERO extension is not installed. Install it with Extensions ▸ Manage extensions.";
        if (!hasOmeroJava())
            return "The OMERO Java libraries are not installed. In Extensions ▸ Manage extensions, choose "
                    + "\"Install optional dependencies\" for the OMERO extension, then restart QuPath.";
        return null;
    }

    private static Object call(Object target, String method) throws Exception {
        Method m = target.getClass().getMethod(method);
        m.setAccessible(true);
        return m.invoke(target);
    }

    /** The servers the OMERO extension is logged in to with a personal account. */
    public static List<Login> logins() {
        List<Login> logins = new ArrayList<>();
        Class<?> clientClass = find(CLIENT_CLASS);
        if (clientClass == null)
            return logins;
        try {
            Collection<?> clients = (Collection<?>) clientClass.getMethod("getClients").invoke(null);
            for (Object client : clients) {
                try {
                    Object apis = call(client, "getApisHandler");
                    Object session = call(apis, "getSessionUuid");
                    if (!(session instanceof Optional<?> optional) || optional.isEmpty())
                        continue; // a public (not logged in) connection cannot write
                    URI web = (URI) call(apis, "getWebServerUri");
                    int port = ((Number) call(apis, "getServerPort")).intValue();
                    String address = String.valueOf(call(apis, "getServerAddress"));
                    Set<String> seen = new LinkedHashSet<>();
                    List<String[]> addresses = new ArrayList<>();
                    // The address chosen in the OMERO extension's Ice settings comes first.
                    try {
                        for (Object api : (Collection<?>) call(client, "getAllPixelApis")) {
                            if (!"Ice".equals(call(api, "getName")))
                                continue;
                            Map<?, ?> args = (Map<?, ?>) call(api, "getArgs");
                            Object iceHost = args.get("--serverAddress"), icePort = args.get("--serverPort");
                            if (iceHost != null && !iceHost.toString().isBlank() && icePort != null
                                    && Integer.parseInt(icePort.toString()) > 0 && seen.add(iceHost + ":" + icePort))
                                addresses.add(new String[]{iceHost.toString(), icePort.toString()});
                        }
                    } catch (Exception e) {
                        logger.debug("No Ice settings in the OMERO extension", e);
                    }
                    for (String host : new String[]{web.getHost(), address})
                        if (host != null && !host.isBlank() && seen.add(host + ":" + port))
                            addresses.add(new String[]{host, Integer.toString(port)});
                    logins.add(new Login(web.toString(), optional.get().toString(), addresses, web.getHost()));
                } catch (Exception e) {
                    logger.debug("Skipping an OMERO connection that could not be read", e);
                }
            }
        } catch (Exception e) {
            logger.warn("Could not read the connections of the OMERO extension: {}", e.toString());
        }
        return logins;
    }

    /**
     * Join the login of the OMERO extension. Call off the JavaFX thread.
     *
     * @param override {@code host:port} of the OMERO server to try first, or null
     * @throws Exception if the server cannot be reached at any of its addresses
     */
    public static RemoteStore connect(Login login, String override) throws Exception {
        String missing = missing();
        if (missing != null)
            throw new IllegalStateException(missing);
        List<String[]> addresses = new ArrayList<>();
        if (override != null && !override.isBlank()) {
            String[] parts = override.strip().split(":");
            addresses.add(new String[]{parts[0], parts.length > 1 ? parts[1] : "4064"});
        }
        addresses.addAll(login.addresses());
        Exception last = null;
        for (String[] address : addresses) {
            try {
                return TimeLimitedStore.wrap(
                        Connector.connect(address[0], Integer.parseInt(address[1]), login.sessionId()), CALL_SECONDS);
            } catch (Exception | LinkageError e) {
                logger.debug("OMERO not reached at {}:{}", address[0], address[1], e);
                last = e instanceof Exception ex ? ex : new IllegalStateException(e);
            }
        }
        StringBuilder tried = new StringBuilder();
        for (String[] address : addresses)
            tried.append(tried.length() == 0 ? "" : ", ").append(address[0]).append(':').append(address[1]);
        throw new IllegalStateException("The OMERO server could not be reached (tried " + tried + "). "
                + "If your server uses another address or port, enter it under \"OMERO server address\".", last);
    }

    /** Kept apart so the OMERO classes are only loaded when a connection is made. */
    private static final class Connector {
        static RemoteStore connect(String host, int port, String sessionId) throws Exception {
            return OmeroGatewayStore.connect(host, port, sessionId, sessionId);
        }
    }

    /** The OMERO image ID of an image opened through the OMERO extension, or null. */
    public static Long imageId(ImageServer<?> server) {
        if (server == null || !server.getClass().getName().equals("qupath.ext.omero.core.imageserver.OmeroImageServer"))
            return null;
        try {
            return ((Number) call(server, "getId")).longValue();
        } catch (Exception e) {
            return imageId(server.getURIs().isEmpty() ? null : server.getURIs().iterator().next());
        }
    }

    /** The OMERO image ID in a web link such as {@code .../webclient/?show=image-12}, or null. */
    public static Long imageId(URI uri) {
        if (uri == null || uri.getScheme() == null || !uri.getScheme().startsWith("http"))
            return null;
        String text = uri.toString();
        for (String pattern : new String[]{"show=image-(\\d+)", "/img_detail/(\\d+)", "[?&]images?=(\\d+)",
                "/iviewer/\\?images=(\\d+)", "/webgateway/img_detail/(\\d+)"}) {
            java.util.regex.Matcher m = java.util.regex.Pattern.compile(pattern).matcher(text);
            if (m.find())
                return Long.parseLong(m.group(1));
        }
        return null;
    }

    /** The web link the OMERO extension opens an image from. */
    public static String imageLink(String webUri, long imageId) {
        String base = webUri.endsWith("/") ? webUri.substring(0, webUri.length() - 1) : webUri;
        return base + "/webclient/?show=image-" + imageId;
    }
}
