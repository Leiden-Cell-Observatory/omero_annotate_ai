package qupath.ext.omero.annotate.omero;

import javafx.beans.property.BooleanProperty;
import javafx.beans.property.SimpleBooleanProperty;
import javafx.beans.property.SimpleStringProperty;
import javafx.beans.property.StringProperty;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Method;

/**
 * Manages OMERO connectivity with two strategies:
 * <ol>
 *     <li>Runtime detection of an existing qupath-extension-omero connection</li>
 *     <li>Direct connection via omero-gateway-java as fallback</li>
 * </ol>
 * <p>
 * If OMERO classes are not on the classpath, the manager gracefully
 * reports OMERO as unavailable, enabling local-only mode.
 */
public class OmeroConnectionManager {

    private static final Logger logger = LoggerFactory.getLogger(OmeroConnectionManager.class);

    private final BooleanProperty connected = new SimpleBooleanProperty(false);
    private final BooleanProperty omeroAvailable = new SimpleBooleanProperty(false);
    private final StringProperty serverInfo = new SimpleStringProperty("Not connected");

    // Hold OMERO objects as Object to avoid compile-time coupling
    private Object gateway;
    private Object securityContext;

    public OmeroConnectionManager() {
        omeroAvailable.set(checkOmeroOnClasspath());
    }

    /**
     * Check if OMERO gateway classes are available at runtime.
     */
    public boolean isOmeroAvailable() {
        return omeroAvailable.get();
    }

    public BooleanProperty omeroAvailableProperty() {
        return omeroAvailable;
    }

    public boolean isConnected() {
        return connected.get();
    }

    public BooleanProperty connectedProperty() {
        return connected;
    }

    public String getServerInfo() {
        return serverInfo.get();
    }

    public StringProperty serverInfoProperty() {
        return serverInfo;
    }

    /**
     * Attempt to detect an existing connection from qupath-extension-omero.
     *
     * @return true if an existing connection was found and reused
     */
    public boolean detectExistingConnection() {
        if (!isOmeroAvailable()) {
            logger.info("OMERO classes not on classpath, skipping connection detection");
            return false;
        }

        try {
            // Try to find the OMERO extension's active server connection via reflection
            // The qupath-extension-omero stores connections in OmeroWebClients or similar
            Class<?> omeroExtClass = Class.forName("qupath.ext.omero.OmeroExtension");
            Method getClients = omeroExtClass.getMethod("getOpenedClients");
            Object clients = getClients.invoke(null);

            if (clients != null) {
                // Check if there are any active clients
                Method isEmpty = clients.getClass().getMethod("isEmpty");
                boolean empty = (boolean) isEmpty.invoke(clients);
                if (!empty) {
                    logger.info("Detected existing OMERO connection from qupath-extension-omero");
                    connected.set(true);
                    serverInfo.set("Connected (via QuPath OMERO extension)");
                    return true;
                }
            }
        } catch (ClassNotFoundException e) {
            logger.debug("qupath-extension-omero not found on classpath");
        } catch (Exception e) {
            logger.debug("Could not detect existing OMERO connection: {}", e.getMessage());
        }

        return false;
    }

    /**
     * Connect directly to an OMERO server using the gateway API.
     *
     * @param host     OMERO server hostname
     * @param port     OMERO server port (typically 4064)
     * @param username OMERO username
     * @param password OMERO password
     * @return true if connection succeeded
     */
    public boolean connectDirect(String host, int port, String username, String password) {
        if (!isOmeroAvailable()) {
            logger.warn("Cannot connect: OMERO classes not available");
            return false;
        }

        try {
            // Use reflection to avoid compile-time dependency on OMERO
            Class<?> gatewayClass = Class.forName("omero.gateway.Gateway");
            Class<?> loginCredClass = Class.forName("omero.gateway.LoginCredentials");
            Class<?> simpleLoggerClass = Class.forName("omero.log.SimpleLogger");

            // Create Gateway
            Object simpleLogger = simpleLoggerClass.getDeclaredConstructor().newInstance();
            gateway = gatewayClass.getDeclaredConstructor(
                    Class.forName("omero.log.Logger")
            ).newInstance(simpleLogger);

            // Create LoginCredentials
            Object loginCreds = loginCredClass.getDeclaredConstructor(
                    String.class, String.class, String.class, int.class
            ).newInstance(username, password, host, port);

            // Connect
            Method connectMethod = gatewayClass.getMethod("connect", loginCredClass);
            Object user = connectMethod.invoke(gateway, loginCreds);

            if (user != null) {
                // Get SecurityContext
                Method getGroupId = user.getClass().getMethod("getGroupId");
                long groupId = (long) getGroupId.invoke(user);

                Class<?> secCtxClass = Class.forName("omero.gateway.SecurityContext");
                securityContext = secCtxClass.getDeclaredConstructor(long.class).newInstance(groupId);

                connected.set(true);
                serverInfo.set("Connected to " + host + ":" + port + " as " + username);
                logger.info("Connected to OMERO server at {}:{}", host, port);
                return true;
            }
        } catch (Exception e) {
            logger.error("Failed to connect to OMERO: {}", e.getMessage(), e);
            serverInfo.set("Connection failed: " + e.getMessage());
        }

        return false;
    }

    /**
     * Disconnect from OMERO server.
     */
    public void disconnect() {
        if (gateway != null) {
            try {
                Method disconnectMethod = gateway.getClass().getMethod("disconnect");
                disconnectMethod.invoke(gateway);
                logger.info("Disconnected from OMERO");
            } catch (Exception e) {
                logger.warn("Error disconnecting from OMERO: {}", e.getMessage());
            }
        }
        gateway = null;
        securityContext = null;
        connected.set(false);
        serverInfo.set("Not connected");
    }

    /**
     * Get the OMERO Gateway object (as Object to avoid compile-time dependency).
     *
     * @return the Gateway, or null if not connected
     */
    public Object getGateway() {
        return gateway;
    }

    /**
     * Get the OMERO SecurityContext (as Object to avoid compile-time dependency).
     *
     * @return the SecurityContext, or null if not connected
     */
    public Object getSecurityContext() {
        return securityContext;
    }

    private static boolean checkOmeroOnClasspath() {
        try {
            Class.forName("omero.gateway.Gateway");
            return true;
        } catch (ClassNotFoundException e) {
            return false;
        }
    }
}
