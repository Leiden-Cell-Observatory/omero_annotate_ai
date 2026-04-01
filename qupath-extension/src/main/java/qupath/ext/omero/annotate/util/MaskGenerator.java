package qupath.ext.omero.annotate.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.images.ImageData;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.regions.RegionRequest;
import qupath.lib.roi.interfaces.ROI;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.IOException;
import java.nio.file.Path;
import java.util.*;
import java.util.List;

/**
 * Generates label masks from QuPath annotations.
 * <p>
 * Each PathClass is assigned a unique integer label (background=0, class1=1, class2=2, ...).
 * Annotations are rendered as filled polygons onto a 16-bit grayscale image.
 */
public class MaskGenerator {

    private static final Logger logger = LoggerFactory.getLogger(MaskGenerator.class);

    /**
     * Build a mapping from PathClass to integer label values.
     * Classes are sorted alphabetically for deterministic assignment.
     * Background (null PathClass) maps to 0.
     *
     * @param annotations list of PathObjects to extract classes from
     * @return mapping from PathClass to label integer
     */
    public static Map<PathClass, Integer> buildClassMapping(Collection<PathObject> annotations) {
        var classSet = new TreeSet<PathClass>(Comparator.comparing(PathClass::toString));
        for (var annotation : annotations) {
            if (annotation.getPathClass() != null) {
                classSet.add(annotation.getPathClass());
            }
        }

        var mapping = new LinkedHashMap<PathClass, Integer>();
        int label = 1;
        for (var pathClass : classSet) {
            mapping.put(pathClass, label++);
        }

        logger.info("Built class mapping with {} classes: {}", mapping.size(), mapping);
        return mapping;
    }

    /**
     * Generate a label mask for a specific image region.
     *
     * @param imageData    the QuPath image data containing annotations
     * @param region       the region to generate a mask for
     * @param classMapping mapping from PathClass to label value
     * @return 16-bit grayscale BufferedImage with label values
     */
    public static BufferedImage generateMask(
            ImageData<?> imageData,
            RegionRequest region,
            Map<PathClass, Integer> classMapping) {

        int width = (int) Math.round(region.getWidth() / region.getDownsample());
        int height = (int) Math.round(region.getHeight() / region.getDownsample());

        // Create 16-bit grayscale image (TYPE_USHORT_GRAY)
        var mask = new BufferedImage(width, height, BufferedImage.TYPE_USHORT_GRAY);
        var g2d = mask.createGraphics();
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF);

        // Get annotations that intersect this region
        var hierarchy = imageData.getHierarchy();
        var regionROI = qupath.lib.roi.ROIs.createRectangleROI(
                region.getX(), region.getY(), region.getWidth(), region.getHeight(),
                region.getImagePlane()
        );
        var annotations = hierarchy.getObjectsForRegion(
                PathObject.class, regionROI.toImageRegion(), null
        );

        double downsample = region.getDownsample();
        double offsetX = region.getX();
        double offsetY = region.getY();

        for (var annotation : annotations) {
            if (annotation.getROI() == null) continue;

            // Get label value for this class
            int labelValue;
            if (annotation.getPathClass() != null && classMapping.containsKey(annotation.getPathClass())) {
                labelValue = classMapping.get(annotation.getPathClass());
            } else {
                labelValue = 1; // Default label for unclassified annotations
            }

            // Convert ROI to AWT Shape, translated to region coordinates
            var roi = annotation.getROI();
            var shape = roiToShape(roi, offsetX, offsetY, downsample);

            if (shape != null) {
                // For 16-bit, we need to write directly to the raster
                fillShapeOnRaster(mask.getRaster(), shape, labelValue, width, height);
            }
        }

        g2d.dispose();
        return mask;
    }

    /**
     * Save a mask image as a 16-bit TIFF file.
     *
     * @param mask     the mask BufferedImage
     * @param filepath output file path
     */
    public static void saveMaskAsTiff(BufferedImage mask, Path filepath) throws IOException {
        logger.debug("Saving mask to {}", filepath);
        java.nio.file.Files.createDirectories(filepath.getParent());

        // Use ImageIO with TIFF writer
        if (!ImageIO.write(mask, "tiff", filepath.toFile())) {
            // Fallback: try PNG if TIFF writer not available
            logger.warn("TIFF writer not available, falling back to PNG");
            ImageIO.write(mask, "png", filepath.toFile().toPath()
                    .resolveSibling(filepath.getFileName().toString().replace(".tif", ".png")).toFile());
        }
    }

    /**
     * Convert a QuPath ROI to an AWT Shape in region-local coordinates.
     */
    private static Shape roiToShape(ROI roi, double offsetX, double offsetY, double downsample) {
        try {
            var geometry = roi.getGeometry();
            if (geometry == null || geometry.isEmpty()) return null;

            // Get the shape from the ROI and transform coordinates
            var shape = roi.getShape();
            if (shape == null) return null;

            // Transform to region-local coordinates
            var transform = new java.awt.geom.AffineTransform();
            transform.scale(1.0 / downsample, 1.0 / downsample);
            transform.translate(-offsetX, -offsetY);

            return transform.createTransformedShape(shape);
        } catch (Exception e) {
            logger.warn("Could not convert ROI to shape: {}", e.getMessage());
            return null;
        }
    }

    /**
     * Fill a shape on a WritableRaster with a given label value.
     * Works correctly for 16-bit images where Graphics2D color mapping is unreliable.
     */
    private static void fillShapeOnRaster(WritableRaster raster, Shape shape, int labelValue, int width, int height) {
        // Create a temporary 8-bit mask for the shape
        var tempMask = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        var g = tempMask.createGraphics();
        g.setColor(Color.WHITE);
        g.fill(shape);
        g.dispose();

        // Copy non-zero pixels to the raster with the correct label value
        var tempRaster = tempMask.getRaster();
        int[] pixel = new int[1];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                tempRaster.getPixel(x, y, pixel);
                if (pixel[0] > 0) {
                    raster.setPixel(x, y, new int[]{labelValue});
                }
            }
        }
    }
}
