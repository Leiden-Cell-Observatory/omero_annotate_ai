package qupath.ext.omero.annotate.core;

import javafx.concurrent.Task;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.omero.annotate.util.MaskGenerator;
import qupath.ext.omero.annotate.util.PatchCoordinateGenerator;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.regions.RegionRequest;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Exports training data (image patches + label masks) from QuPath annotations.
 * <p>
 * Creates the consolidated output directory structure matching the napari plugin:
 * <pre>
 *   output_dir/
 *     input/                    # Source image patches
 *     output/                   # Label masks
 *     tracking_table.csv        # Local tracking
 *     annotation_config.yaml    # Workflow config
 * </pre>
 * <p>
 * Category (train/val/test) is stored in the config YAML, not as folder names.
 */
public class TrainingDataExporter {

    private static final Logger logger = LoggerFactory.getLogger(TrainingDataExporter.class);

    private final ImageData<BufferedImage> imageData;
    private final AnnotationConfig config;
    private final Path outputDir;

    public TrainingDataExporter(ImageData<BufferedImage> imageData, AnnotationConfig config, Path outputDir) {
        this.imageData = imageData;
        this.config = config;
        this.outputDir = outputDir;
    }

    /**
     * Create a JavaFX Task for background export execution.
     */
    public Task<ExportResult> createExportTask() {
        return new Task<>() {
            @Override
            protected ExportResult call() throws Exception {
                return export(this::updateProgress, this::updateMessage);
            }
        };
    }

    /**
     * Export training data synchronously.
     */
    public ExportResult export() throws IOException {
        return export((done, total) -> {}, msg -> {});
    }

    /**
     * Export training data with progress callbacks.
     */
    public ExportResult export(ProgressCallback progressCallback, MessageCallback messageCallback) throws IOException {
        var inputDir = outputDir.resolve("input");
        var outputMaskDir = outputDir.resolve("output");
        Files.createDirectories(inputDir);
        Files.createDirectories(outputMaskDir);

        messageCallback.update("Collecting annotations...");
        ImageServer<BufferedImage> server = imageData.getServer();

        // Get all annotation objects
        var annotations = new ArrayList<>(imageData.getHierarchy().getAnnotationObjects());
        if (annotations.isEmpty()) {
            logger.warn("No annotations found in the current image");
            return new ExportResult(0, 0, 0, outputDir);
        }

        // Build class mapping
        var classMapping = MaskGenerator.buildClassMapping(annotations);

        // Determine patch parameters
        var spatialCoverage = config.getSpatialCoverage();
        int patchWidth = spatialCoverage.patchSize.get(1);
        int patchHeight = spatialCoverage.patchSize.get(0);
        int imageWidth = server.getWidth();
        int imageHeight = server.getHeight();

        // Generate patches
        messageCallback.update("Generating patch coordinates...");
        List<RegionRequest> patches;
        if (spatialCoverage.usePatches) {
            patches = PatchCoordinateGenerator.generateCentroidPatches(
                    annotations, patchWidth, patchHeight,
                    imageWidth, imageHeight,
                    server.getPath(), 1.0
            );
        } else {
            // Use bounding boxes for non-patch mode
            patches = PatchCoordinateGenerator.generateBoundingBoxPatches(
                    annotations, server.getPath(), 1.0
            );
        }

        // Determine train/val split
        double trainFraction = config.getTraining().trainFraction;
        int totalPatches = patches.size();
        int trainCount = (int) Math.round(totalPatches * trainFraction);

        // Shuffle for random split
        var indices = new ArrayList<Integer>();
        for (int i = 0; i < totalPatches; i++) indices.add(i);
        Collections.shuffle(indices);

        // Get image name (strip extension)
        String imageName = server.getMetadata().getName();
        if (imageName == null) imageName = "image";
        imageName = imageName.replaceAll("\\.[^.]+$", "");

        // Determine the image ID (use hash if not from OMERO)
        long imageId = Math.abs(imageName.hashCode());

        // Export each patch
        var trackingTable = new LocalTrackingTable(outputDir);
        int exportedCount = 0;

        messageCallback.update("Exporting patches...");
        for (int idx = 0; idx < totalPatches; idx++) {
            int patchIdx = indices.get(idx);
            var region = patches.get(patchIdx);
            boolean isTrain = idx < trainCount;
            String category = isTrain ? "training" : "validation";

            progressCallback.update(idx, totalPatches);
            messageCallback.update(String.format("Exporting patch %d/%d (%s)...", idx + 1, totalPatches, category));

            // Read image region
            BufferedImage patchImage = server.readRegion(region);
            if (patchImage == null) {
                logger.warn("Could not read region: {}", region);
                continue;
            }

            // Generate label mask for this region
            BufferedImage mask = MaskGenerator.generateMask(imageData, region, classMapping);

            // File naming: {imageName}_{channel}_{index}.tif
            String fileName = String.format("%s_%d_%d.tif", imageName, 0, patchIdx);

            // Save image patch
            Path inputPath = inputDir.resolve(fileName);
            ImageIO.write(patchImage, "tiff", inputPath.toFile());

            // Save mask
            Path maskPath = outputMaskDir.resolve(fileName);
            MaskGenerator.saveMaskAsTiff(mask, maskPath);

            // Create tracking record
            var record = new ImageAnnotationRecord(imageId, imageName);
            record.setCategory(category);
            record.setIsPatch(spatialCoverage.usePatches);
            record.setPatchX(region.getX());
            record.setPatchY(region.getY());
            record.setPatchWidth(region.getWidth());
            record.setPatchHeight(region.getHeight());
            record.setAnnotationType("qupath_manual");
            record.markProcessed();

            trackingTable.addRecord(record);
            config.addAnnotation(record);
            exportedCount++;
        }

        // Save tracking table and config
        messageCallback.update("Saving tracking table and config...");
        trackingTable.saveToCsv();

        config.getOutput().outputDirectory = outputDir.toString();
        config.saveToYaml(outputDir.resolve("annotation_config.yaml"));

        int finalTrainCount = (int) config.getAnnotations().stream()
                .filter(r -> "training".equals(r.getCategory())).count();
        int finalValCount = (int) config.getAnnotations().stream()
                .filter(r -> "validation".equals(r.getCategory())).count();

        progressCallback.update(totalPatches, totalPatches);
        messageCallback.update(String.format("Export complete: %d patches (%d train, %d val)",
                exportedCount, finalTrainCount, finalValCount));

        logger.info("Exported {} patches to {}", exportedCount, outputDir);
        return new ExportResult(exportedCount, finalTrainCount, finalValCount, outputDir);
    }

    /**
     * Result of an export operation.
     */
    public record ExportResult(
            int totalPatches,
            int trainCount,
            int valCount,
            Path outputDirectory
    ) {}

    @FunctionalInterface
    public interface ProgressCallback {
        void update(long done, long total);
    }

    @FunctionalInterface
    public interface MessageCallback {
        void update(String message);
    }
}
