package qupath.ext.omero.annotate.ui;

import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.scene.paint.Color;
import javafx.scene.shape.Circle;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.omero.annotate.omero.OmeroConnectionManager;

/**
 * Tab 1: OMERO connection management and local mode toggle.
 * <p>
 * When "Work without OMERO" is checked, all OMERO fields are hidden
 * and the extension operates in local-only mode.
 */
public class ConnectionPane extends VBox {

    private static final Logger logger = LoggerFactory.getLogger(ConnectionPane.class);

    private final OmeroConnectionManager connectionManager;
    private final CheckBox localModeCheckBox;
    private final Circle statusIndicator;
    private final Label statusLabel;
    private final VBox omeroFieldsBox;

    // OMERO login fields
    private final TextField hostField;
    private final TextField portField;
    private final TextField usernameField;
    private final PasswordField passwordField;
    private final Button connectButton;
    private final Button disconnectButton;

    public ConnectionPane(OmeroConnectionManager connectionManager) {
        this.connectionManager = connectionManager;
        setSpacing(12);
        setPadding(new Insets(15));

        // Local mode toggle
        localModeCheckBox = new CheckBox("Work without OMERO (local only)");
        localModeCheckBox.setStyle("-fx-font-weight: bold;");
        localModeCheckBox.setSelected(!connectionManager.isOmeroAvailable());

        var localModeDescription = new Label(
                "Export training data to a local folder without requiring an OMERO server.");
        localModeDescription.setWrapText(true);
        localModeDescription.setStyle("-fx-text-fill: gray; -fx-font-size: 11px;");

        var localModeBox = new VBox(5, localModeCheckBox, localModeDescription);
        localModeBox.setPadding(new Insets(0, 0, 10, 0));

        // Connection status
        statusIndicator = new Circle(6);
        statusLabel = new Label("Not connected");
        var statusBox = new HBox(8, statusIndicator, statusLabel);
        statusBox.setAlignment(Pos.CENTER_LEFT);

        // OMERO login form
        hostField = new TextField("localhost");
        hostField.setPromptText("OMERO server hostname");
        portField = new TextField("4064");
        portField.setPromptText("Port");
        portField.setPrefWidth(80);
        usernameField = new TextField();
        usernameField.setPromptText("Username");
        passwordField = new PasswordField();
        passwordField.setPromptText("Password");

        var hostPortBox = new HBox(8, new Label("Host:"), hostField, new Label("Port:"), portField);
        hostPortBox.setAlignment(Pos.CENTER_LEFT);
        HBox.setHgrow(hostField, Priority.ALWAYS);

        connectButton = new Button("Connect");
        connectButton.setOnAction(e -> connect());
        connectButton.setDefaultButton(true);

        disconnectButton = new Button("Disconnect");
        disconnectButton.setOnAction(e -> disconnect());
        disconnectButton.setDisable(true);

        var buttonBox = new HBox(8, connectButton, disconnectButton);

        var detectButton = new Button("Detect Existing Connection");
        detectButton.setOnAction(e -> detectExisting());
        detectButton.setStyle("-fx-font-size: 11px;");

        omeroFieldsBox = new VBox(10,
                new Separator(),
                new Label("OMERO Server Connection"),
                statusBox,
                hostPortBox,
                new HBox(8, new Label("Username:"), usernameField),
                new HBox(8, new Label("Password:"), passwordField),
                buttonBox,
                detectButton
        );

        // Bind visibility
        localModeCheckBox.selectedProperty().addListener((obs, oldVal, newVal) -> {
            omeroFieldsBox.setVisible(!newVal);
            omeroFieldsBox.setManaged(!newVal);
        });
        omeroFieldsBox.setVisible(!localModeCheckBox.isSelected());
        omeroFieldsBox.setManaged(!localModeCheckBox.isSelected());

        // Update status indicator based on connection state
        connectionManager.connectedProperty().addListener((obs, oldVal, newVal) -> updateStatus());
        updateStatus();

        getChildren().addAll(localModeBox, omeroFieldsBox);
    }

    public boolean isLocalOnlyMode() {
        return localModeCheckBox.isSelected();
    }

    public CheckBox getLocalModeCheckBox() {
        return localModeCheckBox;
    }

    private void connect() {
        String host = hostField.getText().trim();
        int port;
        try {
            port = Integer.parseInt(portField.getText().trim());
        } catch (NumberFormatException e) {
            port = 4064;
        }
        String username = usernameField.getText().trim();
        String password = passwordField.getText();

        connectButton.setDisable(true);
        connectButton.setText("Connecting...");

        // Run connection in background thread
        var host_ = host;
        var port_ = port;
        new Thread(() -> {
            boolean success = connectionManager.connectDirect(host_, port_, username, password);
            javafx.application.Platform.runLater(() -> {
                connectButton.setDisable(false);
                connectButton.setText("Connect");
                if (!success) {
                    var alert = new Alert(Alert.AlertType.ERROR,
                            "Failed to connect to OMERO server at " + host_ + ":" + port_);
                    alert.showAndWait();
                }
            });
        }).start();
    }

    private void disconnect() {
        connectionManager.disconnect();
    }

    private void detectExisting() {
        boolean found = connectionManager.detectExistingConnection();
        if (!found) {
            var alert = new Alert(Alert.AlertType.INFORMATION,
                    "No existing OMERO connection detected.\n" +
                    "Make sure qupath-extension-omero is installed and connected.");
            alert.showAndWait();
        }
    }

    private void updateStatus() {
        boolean connected = connectionManager.isConnected();
        statusIndicator.setFill(connected ? Color.LIMEGREEN : Color.RED);
        statusLabel.setText(connectionManager.getServerInfo());
        connectButton.setDisable(connected);
        disconnectButton.setDisable(!connected);
    }
}
