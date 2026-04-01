package qupath.ext.omero.annotate.core;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.omero.annotate.omero.OmeroConnectionManager;
import qupath.ext.omero.annotate.omero.OmeroTableManager;
import qupath.ext.omero.annotate.omero.OmeroUploader;
import qupath.lib.images.ImageData;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * Orchestrates the full annotation workflow, supporting both local-only
 * and OMERO modes.
 * <p>
 * Coordinates between the TrainingDataExporter, LocalTrackingTable,
 * and OMERO components (when available).
 */
public class WorkflowManager {

    private static final Logger logger = LoggerFactory.getLogger(WorkflowManager.class);

    private final OmeroConnectionManager connectionManager;
    private AnnotationConfig config;
    private LocalTrackingTable localTable;
    private OmeroUploader omeroUploader;

    public WorkflowManager(OmeroConnectionManager connectionManager) {
        this.connectionManager = connectionManager;
        if (connectionManager.isConnected()) {
            this.omeroUploader = new OmeroUploader(connectionManager);
        }
    }

    /**
     * Initialize a new workflow or resume an existing one.
     *
     * @param config    the workflow configuration
     * @param outputDir output directory for training data
     * @return the initialized config (may be updated with resumed data)
     */
    public AnnotationConfig initializeWorkflow(AnnotationConfig config, Path outputDir) throws IOException {
        this.config = config;
        config.getOutput().outputDirectory = outputDir.toString();

        Files.createDirectories(outputDir);
        Files.createDirectories(outputDir.resolve("input"));
        Files.createDirectories(outputDir.resolve("output"));

        // Initialize local tracking table
        Path csvPath = outputDir.resolve("tracking_table.csv");
        if (Files.exists(csvPath) && config.getWorkflow().resumeFromTable) {
            logger.info("Resuming from existing tracking table: {}", csvPath);
            localTable = LocalTrackingTable.loadFromCsv(csvPath);

            // Merge existing records into config
            for (var record : localTable.getRecords()) {
                if (config.getAnnotations().stream()
                        .noneMatch(a -> a.getImageId() == record.getImageId()
                                && a.getPatchX() == record.getPatchX()
                                && a.getPatchY() == record.getPatchY())) {
                    config.addAnnotation(record);
                }
            }
        } else {
            localTable = new LocalTrackingTable(outputDir);
        }

        // If OMERO mode, try to sync with OMERO table
        if (connectionManager.isConnected() && config.getOmero().tableId != null) {
            var tableManager = new OmeroTableManager(connectionManager);
            var omeroRecords = tableManager.readTable(config.getOmero().tableId);
            if (!omeroRecords.isEmpty()) {
                logger.info("Loaded {} records from OMERO table {}", omeroRecords.size(), config.getOmero().tableId);
                // Merge OMERO records
                for (var record : omeroRecords) {
                    if (config.getAnnotations().stream()
                            .noneMatch(a -> a.getImageId() == record.getImageId()
                                    && a.getPatchX() == record.getPatchX()
                                    && a.getPatchY() == record.getPatchY())) {
                        config.addAnnotation(record);
                    }
                }
            }
        }

        logger.info("Workflow initialized: {} existing annotations, output: {}",
                config.getAnnotations().size(), outputDir);
        return config;
    }

    /**
     * Run the export phase: extract patches and masks from QuPath annotations.
     *
     * @param imageData the QuPath image data with annotations
     * @return export result with statistics
     */
    @SuppressWarnings("unchecked")
    public TrainingDataExporter.ExportResult runExport(ImageData<?> imageData) throws IOException {
        if (config == null) {
            throw new IllegalStateException("Workflow not initialized. Call initializeWorkflow() first.");
        }

        Path outputDir = Path.of(config.getOutput().outputDirectory);
        var exporter = new TrainingDataExporter(
                (ImageData<BufferedImage>) imageData, config, outputDir
        );

        var result = exporter.export();

        // Update local tracking table
        localTable.addRecords(config.getAnnotations());
        localTable.saveToCsv();

        // Save config
        config.saveToYaml(outputDir.resolve("annotation_config.yaml"));

        return result;
    }

    /**
     * Run the upload phase: push results to OMERO.
     *
     * @return true if upload was successful
     */
    public boolean runUpload() {
        if (config == null) {
            throw new IllegalStateException("Workflow not initialized.");
        }

        if (!connectionManager.isConnected()) {
            logger.warn("Cannot upload: not connected to OMERO");
            return false;
        }

        if (omeroUploader == null) {
            omeroUploader = new OmeroUploader(connectionManager);
        }

        Long containerId = config.getOmero().containerId;
        if (containerId == null) {
            logger.error("No OMERO container ID configured");
            return false;
        }

        Path outputDir = Path.of(config.getOutput().outputDirectory);
        return omeroUploader.uploadAll(containerId, config, config.getAnnotations(), outputDir);
    }

    /**
     * Get the current progress summary.
     */
    public java.util.Map<String, Object> getProgressSummary() {
        if (config == null) {
            return java.util.Map.of("total_units", 0, "completed_units", 0,
                    "pending_units", 0, "progress_percent", 0.0);
        }
        return config.getProgressSummary();
    }

    public AnnotationConfig getConfig() {
        return config;
    }

    public LocalTrackingTable getLocalTable() {
        return localTable;
    }
}
