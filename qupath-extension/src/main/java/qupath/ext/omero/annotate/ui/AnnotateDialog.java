package qupath.ext.omero.annotate.ui;

import javafx.scene.Scene;
import javafx.scene.control.Tab;
import javafx.scene.control.TabPane;
import javafx.stage.Stage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.omero.annotate.omero.OmeroConnectionManager;
import qupath.lib.gui.QuPathGUI;

/**
 * Main floating dialog for the OMERO Annotate AI extension.
 * <p>
 * Contains three tabs mirroring the napari plugin's UI pattern:
 * <ol>
 *     <li><b>Connection</b> - OMERO login / local mode toggle</li>
 *     <li><b>Configure</b> - Workflow settings (patches, split, channels)</li>
 *     <li><b>Export & Upload</b> - Run export, upload to OMERO, progress</li>
 * </ol>
 */
public class AnnotateDialog {

    private static final Logger logger = LoggerFactory.getLogger(AnnotateDialog.class);

    private final QuPathGUI qupath;
    private final Stage stage;
    private final TabPane tabPane;
    private final OmeroConnectionManager connectionManager;

    public AnnotateDialog(QuPathGUI qupath) {
        this.qupath = qupath;
        this.connectionManager = new OmeroConnectionManager();

        // Create tabs
        var connectionPane = new ConnectionPane(connectionManager);
        var configurePane = new ConfigurePane(qupath, connectionPane);
        var runExportPane = new RunExportPane(qupath, configurePane, connectionPane, connectionManager);

        var connectionTab = new Tab("Connection", connectionPane);
        connectionTab.setClosable(false);

        var configureTab = new Tab("Configure", configurePane);
        configureTab.setClosable(false);

        var exportTab = new Tab("Export & Upload", runExportPane);
        exportTab.setClosable(false);

        tabPane = new TabPane(connectionTab, configureTab, exportTab);

        // Create stage
        stage = new Stage();
        stage.setTitle("OMERO Annotate AI");
        stage.setScene(new Scene(tabPane, 550, 650));
        stage.initOwner(qupath.getStage());

        // Try to auto-detect existing OMERO connection
        connectionManager.detectExistingConnection();

        logger.info("AnnotateDialog created");
    }

    /**
     * Show the dialog and bring it to front.
     */
    public void show() {
        stage.show();
        stage.toFront();
    }

    /**
     * Show the dialog and select a specific tab.
     *
     * @param tabIndex 0=Connection, 1=Configure, 2=Export
     */
    public void showAndSelectTab(int tabIndex) {
        show();
        if (tabIndex >= 0 && tabIndex < tabPane.getTabs().size()) {
            tabPane.getSelectionModel().select(tabIndex);
        }
    }

    /**
     * Get the underlying stage for external management.
     */
    public Stage getStage() {
        return stage;
    }

    /**
     * Get the OMERO connection manager.
     */
    public OmeroConnectionManager getConnectionManager() {
        return connectionManager;
    }
}
