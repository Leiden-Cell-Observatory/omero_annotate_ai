package qupath.ext.omero.annotate.ui;

import javafx.application.Platform;
import javafx.geometry.Insets;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.omero.annotate.core.AnnotationConfig;
import qupath.ext.omero.annotate.core.TrainingDataExporter;
import qupath.ext.omero.annotate.omero.OmeroConnectionManager;
import qupath.ext.omero.annotate.omero.OmeroUploader;
import qupath.lib.gui.QuPathGUI;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;

/**
 * Tab 3: Export and upload actions with progress tracking.
 * <p>
 * Provides buttons for local export and OMERO upload, along with
 * a progress bar and log area showing current operations.
 */
public class RunExportPane extends VBox {

    private static final Logger logger = LoggerFactory.getLogger(RunExportPane.class);

    private final QuPathGUI qupath;
    private final ConfigurePane configurePane;
    private final ConnectionPane connectionPane;
    private final OmeroConnectionManager connectionManager;

    private final TextArea configPreview;
    private final Button exportButton;
    private final Button uploadButton;
    private final ProgressBar progressBar;
    private final TextArea logArea;
    private final Label summaryLabel;

    public RunExportPane(QuPathGUI qupath, ConfigurePane configurePane,
                         ConnectionPane connectionPane, OmeroConnectionManager connectionManager) {
        this.qupath = qupath;
        this.configurePane = configurePane;
        this.connectionPane = connectionPane;
        this.connectionManager = connectionManager;

        setSpacing(12);
        setPadding(new Insets(15));

        // Config preview
        configPreview = new TextArea();
        configPreview.setEditable(false);
        configPreview.setPrefRowCount(6);
        configPreview.setStyle("-fx-font-family: monospace; -fx-font-size: 11px;");
        var refreshPreview = new Button("Refresh Preview");
        refreshPreview.setOnAction(e -> updateConfigPreview());

        var previewSection = new VBox(5,
                createBoldLabel("Configuration Preview"),
                configPreview,
                refreshPreview
        );

        // Action buttons
        exportButton = new Button("Export Training Data");
        exportButton.setStyle("-fx-font-weight: bold; -fx-font-size: 14px;");
        exportButton.setPrefWidth(250);
        exportButton.setOnAction(e -> runExport());

        uploadButton = new Button("Upload to OMERO");
        uploadButton.setPrefWidth(250);
        uploadButton.setOnAction(e -> runUpload());

        // Bind upload button to OMERO availability
        connectionPane.getLocalModeCheckBox().selectedProperty().addListener((obs, oldVal, newVal) -> {
            uploadButton.setDisable(newVal);
        });
        uploadButton.setDisable(connectionPane.isLocalOnlyMode());

        var buttonBox = new HBox(12, exportButton, uploadButton);

        // Progress
        progressBar = new ProgressBar(0);
        progressBar.setPrefWidth(Double.MAX_VALUE);
        progressBar.setVisible(false);

        // Summary
        summaryLabel = new Label("");
        summaryLabel.setWrapText(true);
        summaryLabel.setStyle("-fx-font-weight: bold;");

        // Log area
        logArea = new TextArea();
        logArea.setEditable(false);
        logArea.setPrefRowCount(10);
        logArea.setStyle("-fx-font-family: monospace; -fx-font-size: 11px;");
        VBox.setVgrow(logArea, Priority.ALWAYS);

        getChildren().addAll(
                previewSection,
                new Separator(),
                buttonBox,
                progressBar,
                summaryLabel,
                createBoldLabel("Log"),
                logArea
        );
    }

    private void runExport() {
        var imageData = qupath.getImageData();
        if (imageData == null) {
            showError("No image is currently open in QuPath.");
            return;
        }

        String outputDir = configurePane.getOutputDirectory();
        if (outputDir.isEmpty()) {
            showError("Please set an output directory in the Configure tab.");
            return;
        }

        var annotations = imageData.getHierarchy().getAnnotationObjects();
        if (annotations.isEmpty()) {
            showError("No annotations found. Please draw annotations on the image first.");
            return;
        }

        // Build config from UI
        var config = configurePane.buildConfig();
        updateConfigPreview();

        // Set up exporter
        @SuppressWarnings("unchecked")
        var typedImageData = (qupath.lib.images.ImageData<BufferedImage>) imageData;
        var exporter = new TrainingDataExporter(typedImageData, config, Path.of(outputDir));

        // Run in background
        exportButton.setDisable(true);
        progressBar.setVisible(true);
        progressBar.setProgress(-1); // Indeterminate
        logArea.clear();
        appendLog("Starting export...");

        new Thread(() -> {
            try {
                var result = exporter.export(
                        (done, total) -> Platform.runLater(() -> {
                            if (total > 0) progressBar.setProgress((double) done / total);
                        }),
                        msg -> Platform.runLater(() -> appendLog(msg))
                );

                Platform.runLater(() -> {
                    progressBar.setProgress(1.0);
                    summaryLabel.setText(String.format(
                            "Export complete: %d patches (%d train, %d val) → %s",
                            result.totalPatches(), result.trainCount(), result.valCount(),
                            result.outputDirectory()
                    ));
                    appendLog("Export completed successfully!");
                    exportButton.setDisable(false);
                });

            } catch (IOException e) {
                Platform.runLater(() -> {
                    appendLog("ERROR: " + e.getMessage());
                    showError("Export failed: " + e.getMessage());
                    exportButton.setDisable(false);
                    progressBar.setVisible(false);
                });
            }
        }).start();
    }

    private void runUpload() {
        if (!connectionManager.isConnected()) {
            showError("Not connected to OMERO. Please connect in the Connection tab.");
            return;
        }

        var config = configurePane.buildConfig();
        if (config.getOmero().containerId == null) {
            showError("Please set an OMERO container ID in the Configure tab.");
            return;
        }

        String outputDir = configurePane.getOutputDirectory();
        if (outputDir.isEmpty()) {
            showError("Please set an output directory and export first.");
            return;
        }

        uploadButton.setDisable(true);
        appendLog("Starting OMERO upload...");

        new Thread(() -> {
            var uploader = new OmeroUploader(connectionManager);
            boolean success = uploader.uploadAll(
                    config.getOmero().containerId,
                    config,
                    config.getAnnotations(),
                    Path.of(outputDir)
            );

            Platform.runLater(() -> {
                if (success) {
                    appendLog("Upload to OMERO completed successfully!");
                    summaryLabel.setText("Upload complete. Table ID: " + config.getOmero().tableId);
                } else {
                    appendLog("ERROR: Upload to OMERO failed.");
                    showError("Upload to OMERO failed. Check the log for details.");
                }
                uploadButton.setDisable(connectionPane.isLocalOnlyMode());
            });
        }).start();
    }

    private void updateConfigPreview() {
        var config = configurePane.buildConfig();
        var summary = new StringBuilder();
        summary.append("name: ").append(config.getName()).append("\n");
        summary.append("mode: ").append(connectionPane.isLocalOnlyMode() ? "local-only" : "OMERO").append("\n");
        summary.append("patches: ").append(config.getSpatialCoverage().usePatches).append("\n");
        if (config.getSpatialCoverage().usePatches) {
            summary.append("patch_size: ").append(config.getSpatialCoverage().patchSize).append("\n");
        }
        summary.append("train_fraction: ").append(config.getTraining().trainFraction).append("\n");
        summary.append("channels: ").append(config.getSpatialCoverage().channels).append("\n");
        if (config.getOutput().outputDirectory != null) {
            summary.append("output: ").append(config.getOutput().outputDirectory).append("\n");
        }
        configPreview.setText(summary.toString());
    }

    private void appendLog(String message) {
        logArea.appendText(message + "\n");
        // Auto-scroll to bottom
        logArea.setScrollTop(Double.MAX_VALUE);
    }

    private void showError(String message) {
        var alert = new Alert(Alert.AlertType.ERROR, message);
        alert.showAndWait();
    }

    private Label createBoldLabel(String text) {
        var label = new Label(text);
        label.setStyle("-fx-font-weight: bold; -fx-font-size: 13px;");
        return label;
    }
}
