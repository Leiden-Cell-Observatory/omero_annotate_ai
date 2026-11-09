package qupath.ext.omeroannotate.qp;

import org.locationtech.jts.geom.Coordinate;
import org.locationtech.jts.geom.Geometry;
import org.locationtech.jts.geom.Polygon;
import qupath.ext.omeroannotate.core.PixelArray;
import qupath.ext.omeroannotate.core.RemoteStore;
import qupath.ext.omeroannotate.core.Row;
import qupath.ext.omeroannotate.core.TrainingSet;
import qupath.lib.analysis.images.ContourTracing;
import qupath.lib.analysis.images.SimpleImages;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.images.servers.PixelType;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjects;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.objects.hierarchy.PathObjectHierarchy;
import qupath.lib.regions.ImagePlane;
import qupath.lib.regions.RegionRequest;
import qupath.lib.roi.ROIs;
import qupath.lib.roi.RoiTools;
import qupath.lib.roi.interfaces.ROI;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.Shape;
import java.awt.geom.AffineTransform;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.awt.image.WritableRaster;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

/**
 * How an annotation unit appears in QuPath: a locked rectangle (the <i>frame</i>) marks
 * the plane or patch to annotate, and every object inside it becomes one label of the
 * unit's mask. It does not matter how the objects were made: by hand, with Cellpose,
 * InstanSeg, StarDist or SAM, as annotations, detections or cells.
 */
public final class Units {

    /** Metadata key on a frame: the annotation ID of its unit. */
    public static final String META_UNIT = "omero-annotate-ai unit";
    /** Metadata key on a frame: which training set it belongs to. */
    public static final String META_SET = "omero-annotate-ai set";
    /** Metadata key on a frame: digest of the mask its objects were last saved as or loaded from. */
    public static final String META_DIGEST = "omero-annotate-ai mask";

    /** Largest unit (in pixels) that is rasterised in one piece. */
    public static final long MAX_UNIT_PIXELS = 100_000_000L;

    private Units() {}

    /** The unit's region in full-resolution pixels: {@code x, y, width, height}. */
    public static int[] region(Row row, ImageServer<?> server) {
        if (row.isPatch) {
            int x = Math.max(0, Math.min(row.patchX, server.getWidth() - 1));
            int y = Math.max(0, Math.min(row.patchY, server.getHeight() - 1));
            return new int[]{x, y, Math.min(row.patchWidth, server.getWidth() - x),
                    Math.min(row.patchHeight, server.getHeight() - y)};
        }
        return new int[]{0, 0, server.getWidth(), server.getHeight()};
    }

    public static ImagePlane plane(Row row) {
        return ImagePlane.getPlane(Math.max(0, row.zSlice), Math.max(0, row.timepoint));
    }

    /** The PathClass of a frame: Training, Validation or Test (as Cellpose training in QuPath expects). */
    public static PathClass frameClass(String category) {
        return switch (category) {
            case Row.VALIDATION -> PathClass.fromString("Validation", 0xFFA500);
            case Row.TEST -> PathClass.fromString("Test", 0x9370DB);
            default -> PathClass.fromString("Training", 0x00B4D8);
        };
    }

    public static boolean isFrame(PathObject object) {
        return object.getMetadata().containsKey(META_UNIT);
    }

    /** The frame of a unit of the set {@code setKey} in a hierarchy, or null. */
    public static PathObject findFrame(PathObjectHierarchy hierarchy, String setKey, Row row) {
        for (PathObject object : hierarchy.getAnnotationObjects()) {
            Map<String, String> metadata = object.getMetadata();
            if (row.annotationId.equals(metadata.get(META_UNIT)) && setKey.equals(metadata.get(META_SET)))
                return object;
        }
        return null;
    }

    /** The unit's frame, created (locked, named and classified by split) if the image has none yet. */
    public static PathObject ensureFrame(PathObjectHierarchy hierarchy, String setKey, Row row, ImageServer<?> server) {
        PathObject frame = findFrame(hierarchy, setKey, row);
        if (frame != null)
            return frame;
        int[] r = region(row, server);
        frame = PathObjects.createAnnotationObject(
                ROIs.createRectangleROI(r[0], r[1], r[2], r[3], plane(row)), frameClass(row.category));
        frame.setName("Annotate here (" + row.category + ")");
        frame.getMetadata().put(META_UNIT, row.annotationId);
        frame.getMetadata().put(META_SET, setKey);
        frame.setLocked(true);
        hierarchy.addObject(frame);
        return frame;
    }

    /**
     * The objects that make up a unit's mask: every object with an area on the frame's
     * plane whose centre lies inside the frame. Frames themselves, objects with an
     * "ignored" class (name ending in *) and regions enclosing the frame are left out.
     */
    public static List<PathObject> objectsInFrame(PathObjectHierarchy hierarchy, PathObject frame) {
        ROI frameRoi = frame.getROI();
        double x0 = frameRoi.getBoundsX(), y0 = frameRoi.getBoundsY();
        double x1 = x0 + frameRoi.getBoundsWidth(), y1 = y0 + frameRoi.getBoundsHeight();
        List<PathObject> ancestors = new ArrayList<>();
        for (PathObject p = frame.getParent(); p != null; p = p.getParent())
            ancestors.add(p);
        List<PathObject> result = new ArrayList<>();
        // Everything touching the frame; the centre rule below decides (QuPath's own
        // "objects for ROI" would drop hand-drawn annotations that cross the border).
        Collection<PathObject> candidates = hierarchy.getAllObjectsForRegion(qupath.lib.regions.ImageRegion
                .createInstance((int) Math.floor(x0), (int) Math.floor(y0), (int) Math.ceil(x1 - x0),
                        (int) Math.ceil(y1 - y0), frameRoi.getZ(), frameRoi.getT()));
        for (PathObject object : candidates) {
            if (object == frame || object.isRootObject() || object.isTMACore() || isFrame(object))
                continue;
            ROI roi = object.getROI();
            if (roi == null || !roi.isArea() || roi.isEmpty())
                continue;
            if (roi.getZ() != frameRoi.getZ() || roi.getT() != frameRoi.getT())
                continue;
            if (ancestors.contains(object))
                continue;
            PathClass pathClass = object.getPathClass();
            if (pathClass != null && pathClass.getName() != null && pathClass.getName().endsWith("*"))
                continue;
            double cx = roi.getCentroidX(), cy = roi.getCentroidY();
            if (cx < x0 || cx >= x1 || cy < y0 || cy >= y1)
                continue;
            // A region covering the whole frame (tissue outline) is context, not an object.
            if (roi.getBoundsX() <= x0 && roi.getBoundsY() <= y0 && roi.getBoundsX() + roi.getBoundsWidth() >= x1
                    && roi.getBoundsY() + roi.getBoundsHeight() >= y1 && object.isAnnotation()
                    && roi.getArea() >= 0.98 * frameRoi.getArea())
                continue;
            result.add(object);
        }
        return result;
    }

    /**
     * The label mask of some objects within a region: 0 is background, each object one
     * number from 1. Where objects overlap the smaller one stays visible; objects that
     * end up without a pixel get no label.
     */
    public static PixelArray rasterize(List<PathObject> objects, int x, int y, int width, int height) {
        if ((long) width * height > MAX_UNIT_PIXELS)
            throw new IllegalArgumentException("This region is too large to annotate in one piece (" + width
                    + " x " + height + " pixels). Make the training set with patches instead.");
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = image.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF);
        g.setRenderingHint(RenderingHints.KEY_STROKE_CONTROL, RenderingHints.VALUE_STROKE_PURE);
        g.translate(-x, -y);
        List<PathObject> ordered = new ArrayList<>(objects);
        ordered.sort(Comparator.comparingDouble((PathObject o) -> o.getROI().getArea()).reversed());
        if (ordered.size() >= 0xFFFFFF)
            throw new IllegalArgumentException("Too many objects in one region: " + ordered.size());
        int label = 0;
        for (PathObject object : ordered) {
            Shape shape = RoiTools.getShape(object.getROI());
            g.setColor(new Color(++label));
            g.fill(shape);
        }
        g.dispose();
        int[] data = ((DataBufferInt) image.getRaster().getDataBuffer()).getData();
        int[] labels = new int[data.length];
        // Renumber 1..N in the order labels first appear, dropping objects without pixels.
        int[] mapping = new int[label + 1];
        int next = 0;
        for (int i = 0; i < data.length; i++) {
            int value = data[i] & 0xFFFFFF;
            if (value == 0)
                continue;
            if (mapping[value] == 0)
                mapping[value] = ++next;
            labels[i] = mapping[value];
        }
        return PixelArray.ofLabels(labels, width, height);
    }

    /** How many objects a label mask holds. */
    public static int countLabels(PixelArray mask) {
        java.util.BitSet seen = new java.util.BitSet();
        for (int value : mask.toInts())
            if (value > 0)
                seen.set(value);
        return seen.cardinality();
    }

    private static Map<Number, Geometry> trace(PixelArray mask, int x, int y, ImagePlane plane) {
        int[] values = mask.toInts();
        float[] data = new float[values.length];
        for (int i = 0; i < values.length; i++)
            data[i] = values[i];
        RegionRequest request = RegionRequest.createInstance("mask", 1.0, x, y, mask.width(), mask.height(),
                plane.getZ(), plane.getT());
        return ContourTracing.createGeometries(SimpleImages.createFloatImage(data, mask.width(), mask.height()),
                request, 1, -1);
    }

    /** One annotation per label of a mask placed at {@code (x, y)} on {@code plane}. */
    public static List<PathObject> objectsFromMask(PixelArray mask, int x, int y, ImagePlane plane) {
        List<PathObject> objects = new ArrayList<>();
        for (Geometry geometry : trace(mask, x, y, plane).values()) {
            if (geometry == null || geometry.isEmpty())
                continue;
            objects.add(PathObjects.createAnnotationObject(
                    qupath.lib.roi.GeometryTools.geometryToROI(geometry, plane)));
        }
        return objects;
    }

    /** Outlines of a mask's objects for OMERO: one polygon per connected piece, in full-image pixels. */
    public static final TrainingSet.MaskTracer TRACER = (mask, offsetX, offsetY) -> {
        List<RemoteStore.Polygon> polygons = new ArrayList<>();
        for (Geometry geometry : trace(mask, offsetX, offsetY, ImagePlane.getDefaultPlane()).values()) {
            if (geometry == null)
                continue;
            for (int i = 0; i < geometry.getNumGeometries(); i++) {
                if (!(geometry.getGeometryN(i) instanceof Polygon polygon) || polygon.isEmpty())
                    continue;
                Coordinate[] ring = polygon.getExteriorRing().getCoordinates();
                int n = ring.length - 1; // the ring repeats its first point
                if (n < 3)
                    continue;
                double[] xy = new double[n * 2];
                for (int k = 0; k < n; k++) {
                    xy[2 * k] = ring[k].x;
                    xy[2 * k + 1] = ring[k].y;
                }
                polygons.add(new RemoteStore.Polygon(xy));
            }
        }
        return polygons;
    };

    /**
     * The pixels the annotator sees for a unit: the annotated channel of its plane or
     * patch at full resolution (all three colours for RGB images such as H&amp;E).
     */
    public static PixelArray readInput(ImageServer<BufferedImage> server, Row row) throws IOException {
        int[] r = region(row, server);
        if ((long) r[2] * r[3] > MAX_UNIT_PIXELS)
            throw new IOException("This region is too large to annotate in one piece (" + r[2] + " x " + r[3]
                    + " pixels). Make the training set with patches instead.");
        BufferedImage image = server.readRegion(RegionRequest.createInstance(server.getPath(), 1.0, r[0], r[1],
                r[2], r[3], Math.max(0, row.zSlice), Math.max(0, row.timepoint)));
        if (image == null)
            throw new IOException("The image could not be read");
        int width = image.getWidth(), height = image.getHeight();
        if (server.isRGB()) {
            int[] argb = image.getRGB(0, 0, width, height, null, 0, width);
            byte[] bytes = new byte[argb.length * 3];
            for (int i = 0; i < argb.length; i++) {
                bytes[3 * i] = (byte) (argb[i] >> 16);
                bytes[3 * i + 1] = (byte) (argb[i] >> 8);
                bytes[3 * i + 2] = (byte) argb[i];
            }
            return new PixelArray("u1", width, height, 3, bytes);
        }
        WritableRaster raster = image.getRaster();
        int band = Math.max(0, Math.min(row.channel, raster.getNumBands() - 1));
        PixelType type = server.getPixelType();
        String dtype = switch (type) {
            case UINT8 -> "u1";
            case INT8 -> "i1";
            case UINT16 -> "u2";
            case INT16 -> "i2";
            case UINT32 -> "u4";
            case INT32 -> "i4";
            case FLOAT32 -> "f4";
            case FLOAT64 -> "f8";
        };
        int n = width * height;
        ByteBuffer buffer = ByteBuffer.allocate(n * PixelArray.bytesPerSample(dtype)).order(ByteOrder.LITTLE_ENDIAN);
        switch (dtype) {
            case "f4" -> {
                float[] samples = raster.getSamples(0, 0, width, height, band, (float[]) null);
                for (float v : samples)
                    buffer.putFloat(v);
            }
            case "f8" -> {
                double[] samples = raster.getSamples(0, 0, width, height, band, (double[]) null);
                for (double v : samples)
                    buffer.putDouble(v);
            }
            default -> {
                int[] samples = raster.getSamples(0, 0, width, height, band, (int[]) null);
                for (int v : samples) {
                    switch (dtype) {
                        case "u1", "i1" -> buffer.put((byte) v);
                        case "u2", "i2" -> buffer.putShort((short) v);
                        default -> buffer.putInt(v);
                    }
                }
            }
        }
        return new PixelArray(dtype, width, height, 1, buffer.array());
    }

    /** Bounds of a frame as whole pixels: {@code x, y, width, height}. */
    public static int[] bounds(PathObject frame) {
        Rectangle2D b = AffineTransform.getTranslateInstance(0, 0)
                .createTransformedShape(RoiTools.getShape(frame.getROI())).getBounds2D();
        return new int[]{(int) Math.round(b.getX()), (int) Math.round(b.getY()), (int) Math.round(b.getWidth()),
                (int) Math.round(b.getHeight())};
    }
}
