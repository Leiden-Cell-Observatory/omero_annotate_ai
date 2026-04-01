package qupath.ext.omero.annotate.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.objects.PathObject;
import qupath.lib.regions.ImagePlane;
import qupath.lib.regions.RegionRequest;
import qupath.lib.roi.interfaces.ROI;

import java.util.ArrayList;
import java.util.List;

/**
 * Generates centroid-centered patch coordinates from QuPath annotations.
 * <p>
 * Mirrors the Python {@code generate_patch_coordinates()} from
 * {@code omero_annotate_ai.processing.image_functions}.
 */
public class PatchCoordinateGenerator {

    private static final Logger logger = LoggerFactory.getLogger(PatchCoordinateGenerator.class);

    /**
     * Generate centroid-centered patch regions for a list of annotations.
     *
     * @param annotations list of QuPath PathObjects with ROIs
     * @param patchWidth  desired patch width in pixels
     * @param patchHeight desired patch height in pixels
     * @param imageWidth  full image width for bounds clipping
     * @param imageHeight full image height for bounds clipping
     * @param serverPath  image server path for creating RegionRequests
     * @param downsample  downsample factor (1.0 for full resolution)
     * @return list of RegionRequests for each patch
     */
    public static List<RegionRequest> generateCentroidPatches(
            List<PathObject> annotations,
            int patchWidth,
            int patchHeight,
            int imageWidth,
            int imageHeight,
            String serverPath,
            double downsample) {

        var patches = new ArrayList<RegionRequest>();

        for (var annotation : annotations) {
            ROI roi = annotation.getROI();
            if (roi == null) {
                logger.warn("Skipping annotation without ROI");
                continue;
            }

            // Get centroid of the annotation
            double centroidX = roi.getCentroidX();
            double centroidY = roi.getCentroidY();

            // Calculate patch top-left corner (centered on centroid)
            int x = (int) Math.round(centroidX - patchWidth / 2.0);
            int y = (int) Math.round(centroidY - patchHeight / 2.0);

            // Clip to image bounds
            x = Math.max(0, Math.min(x, imageWidth - patchWidth));
            y = Math.max(0, Math.min(y, imageHeight - patchHeight));

            // Ensure patch doesn't exceed image dimensions
            int actualWidth = Math.min(patchWidth, imageWidth - x);
            int actualHeight = Math.min(patchHeight, imageHeight - y);

            if (actualWidth <= 0 || actualHeight <= 0) {
                logger.warn("Skipping annotation at ({}, {}) - patch outside image bounds",
                        centroidX, centroidY);
                continue;
            }

            var regionRequest = RegionRequest.createInstance(
                    serverPath, downsample, x, y, actualWidth, actualHeight
            );

            patches.add(regionRequest);
            logger.debug("Generated patch at ({}, {}) size {}x{} for annotation centroid ({}, {})",
                    x, y, actualWidth, actualHeight, centroidX, centroidY);
        }

        logger.info("Generated {} patches from {} annotations", patches.size(), annotations.size());
        return patches;
    }

    /**
     * Generate patches using bounding boxes of annotations instead of centroids.
     * Useful when annotations are larger than the patch size.
     *
     * @param annotations list of QuPath PathObjects with ROIs
     * @param serverPath  image server path
     * @param downsample  downsample factor
     * @return list of RegionRequests matching annotation bounding boxes
     */
    public static List<RegionRequest> generateBoundingBoxPatches(
            List<PathObject> annotations,
            String serverPath,
            double downsample) {

        var patches = new ArrayList<RegionRequest>();

        for (var annotation : annotations) {
            ROI roi = annotation.getROI();
            if (roi == null) continue;

            int x = (int) Math.round(roi.getBoundsX());
            int y = (int) Math.round(roi.getBoundsY());
            int w = (int) Math.round(roi.getBoundsWidth());
            int h = (int) Math.round(roi.getBoundsHeight());

            if (w > 0 && h > 0) {
                patches.add(RegionRequest.createInstance(serverPath, downsample, x, y, w, h));
            }
        }

        return patches;
    }
}
