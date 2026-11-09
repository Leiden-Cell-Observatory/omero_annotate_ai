package qupath.ext.omeroannotate.qp;

import org.junit.jupiter.api.Test;
import qupath.ext.omeroannotate.core.PixelArray;
import qupath.ext.omeroannotate.core.RemoteStore;
import qupath.ext.omeroannotate.core.Row;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjects;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.objects.hierarchy.PathObjectHierarchy;
import qupath.lib.regions.ImagePlane;
import qupath.lib.roi.ROIs;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/** Objects to label masks and back. */
class UnitsTest {

    private static PathObject rect(double x, double y, double w, double h) {
        return PathObjects.createAnnotationObject(ROIs.createRectangleROI(x, y, w, h, ImagePlane.getDefaultPlane()));
    }

    private static PathObject frame(PathObjectHierarchy hierarchy, double x, double y, double w, double h) {
        PathObject frame = rect(x, y, w, h);
        frame.getMetadata().put(Units.META_UNIT, "unit");
        frame.getMetadata().put(Units.META_SET, "set");
        hierarchy.addObject(frame);
        return frame;
    }

    @Test
    void objectsBecomeLabelsAndComeBack() {
        List<PathObject> objects = List.of(rect(10, 10, 20, 10), rect(50, 40, 8, 8),
                PathObjects.createDetectionObject(ROIs.createEllipseROI(60, 10, 20, 20, ImagePlane.getDefaultPlane())));
        PixelArray mask = Units.rasterize(objects, 0, 0, 100, 80);
        assertEquals("u2", mask.dtype());
        assertEquals(3, Units.countLabels(mask));
        int[] labels = mask.toInts();
        assertEquals(0, labels[0]);
        assertTrue(labels[15 * 100 + 15] > 0);
        long rectPixels = java.util.Arrays.stream(labels).filter(v -> v == labels[15 * 100 + 15]).count();
        assertEquals(200, rectPixels, "a 20 x 10 rectangle covers exactly 200 pixels");

        List<PathObject> back = Units.objectsFromMask(mask, 0, 0, ImagePlane.getDefaultPlane());
        assertEquals(3, back.size());
        PixelArray again = Units.rasterize(back, 0, 0, 100, 80);
        assertEquals(mask.digest(), again.digest(), "mask -> objects -> mask is lossless");
    }

    @Test
    void masksAreCutToTheirRegion() {
        // A patch at (100, 200): objects are in full-image coordinates, the mask in patch coordinates.
        List<PathObject> objects = List.of(rect(110, 210, 10, 10), rect(90, 195, 20, 20));
        PixelArray mask = Units.rasterize(objects, 100, 200, 64, 64);
        int[] labels = mask.toInts();
        assertTrue(labels[15 * 64 + 15] > 0, "the object at (110..120, 210..220) lands at (10..20, 10..20)");
        assertTrue(labels[0] > 0, "an object crossing the border is cut at the border");
        assertEquals(2, Units.countLabels(mask));
        List<PathObject> back = Units.objectsFromMask(mask, 100, 200, ImagePlane.getPlane(2, 1));
        assertEquals(2, back.size());
        for (PathObject object : back) {
            assertTrue(object.getROI().getBoundsX() >= 100 && object.getROI().getBoundsY() >= 200);
            assertEquals(2, object.getROI().getZ());
            assertEquals(1, object.getROI().getT());
        }
        List<RemoteStore.Polygon> polygons = Units.TRACER.trace(mask, 100, 200);
        assertEquals(2, polygons.size());
        for (RemoteStore.Polygon polygon : polygons)
            for (int i = 0; i < polygon.xy().length; i += 2)
                assertTrue(polygon.xy()[i] >= 100 && polygon.xy()[i] <= 164 && polygon.xy()[i + 1] >= 200);
    }

    @Test
    void smallerObjectsStayOnTopAndHiddenOnesGetNoLabel() {
        PixelArray mask = Units.rasterize(List.of(rect(10, 10, 30, 30), rect(20, 20, 5, 5), rect(500, 500, 5, 5)),
                0, 0, 60, 60);
        int[] labels = mask.toInts();
        assertEquals(2, Units.countLabels(mask), "the object outside the region has no label");
        assertNotEquals(labels[12 * 60 + 12], labels[22 * 60 + 22]);
        assertEquals(java.util.stream.IntStream.of(labels).max().getAsInt(), 2, "labels are 1..N without gaps");
    }

    @Test
    void anEmptyRegionGivesAnEmptyMask() {
        PixelArray mask = Units.rasterize(List.of(), 0, 0, 32, 16);
        assertEquals(0, Units.countLabels(mask));
        assertEquals(32, mask.width());
        assertEquals(16, mask.height());
        assertTrue(Units.objectsFromMask(mask, 0, 0, ImagePlane.getDefaultPlane()).isEmpty());
        assertTrue(Units.TRACER.trace(mask, 0, 0).isEmpty());
    }

    @Test
    void onlyObjectsInsideTheFrameCount() {
        PathObjectHierarchy hierarchy = new PathObjectHierarchy();
        PathObject tissue = rect(0, 0, 1000, 1000); // a region around everything
        hierarchy.addObject(tissue);
        PathObject frame = frame(hierarchy, 100, 100, 200, 200);
        PathObject inside = rect(150, 150, 20, 20);
        PathObject crossing = rect(280, 150, 30, 20); // centre at 295: inside
        PathObject outside = rect(400, 400, 20, 20);
        PathObject otherPlane = PathObjects.createAnnotationObject(ROIs.createRectangleROI(150, 200, 20, 20, ImagePlane.getPlane(3, 0)));
        PathObject ignored = rect(200, 200, 20, 20);
        ignored.setPathClass(PathClass.fromString("Ignore*"));
        PathObject cell = PathObjects.createCellObject(ROIs.createEllipseROI(120, 250, 30, 30, ImagePlane.getDefaultPlane()),
                ROIs.createEllipseROI(130, 260, 10, 10, ImagePlane.getDefaultPlane()), null, null);
        PathObject point = PathObjects.createAnnotationObject(ROIs.createPointsROI(160, 160, ImagePlane.getDefaultPlane()));
        PathObject otherFrame = frame(hierarchy, 120, 120, 50, 50);
        hierarchy.addObjects(List.of(inside, crossing, outside, otherPlane, ignored, cell, point));

        List<PathObject> found = Units.objectsInFrame(hierarchy, frame);
        assertTrue(found.contains(inside));
        assertTrue(found.contains(crossing));
        assertTrue(found.contains(cell));
        assertEquals(3, found.size(), found.toString());
        assertFalse(found.contains(otherFrame));
    }

    @Test
    void framesAreFoundAgainAndNamedBySplit() {
        PathObjectHierarchy hierarchy = new PathObjectHierarchy();
        Row row = new Row();
        row.annotationId = "abc_0_0_1";
        row.category = Row.VALIDATION;
        row.isPatch = true;
        row.patchX = 10;
        row.patchY = 20;
        row.patchWidth = 64;
        row.patchHeight = 32;
        row.zSlice = 0;
        row.timepoint = 0;
        assertNull(Units.findFrame(hierarchy, "set-1", row));
        assertEquals("Validation", Units.frameClass(row.category).getName());
        assertEquals("Training", Units.frameClass(Row.TRAINING).getName());
        assertEquals("Test", Units.frameClass(Row.TEST).getName());
    }

    @Test
    void thousandsOfObjectsAreFastEnough() {
        // A dense 2048 x 2048 field with 5000 nuclei: saving must not make the annotator wait.
        Random random = new Random(3);
        List<PathObject> objects = new ArrayList<>();
        for (int i = 0; i < 5000; i++)
            objects.add(PathObjects.createDetectionObject(ROIs.createEllipseROI(random.nextInt(2000),
                    random.nextInt(2000), 14 + random.nextInt(20), 14 + random.nextInt(20), ImagePlane.getDefaultPlane())));
        long start = System.nanoTime();
        PixelArray mask = Units.rasterize(objects, 0, 0, 2048, 2048);
        long rasterMs = (System.nanoTime() - start) / 1_000_000;
        start = System.nanoTime();
        List<RemoteStore.Polygon> polygons = Units.TRACER.trace(mask, 0, 0);
        long traceMs = (System.nanoTime() - start) / 1_000_000;
        start = System.nanoTime();
        String digest = mask.digest();
        byte[] tiff = qupath.ext.omeroannotate.core.TiffIO.encode(mask, true);
        long encodeMs = (System.nanoTime() - start) / 1_000_000;
        System.out.println("PERF objects=5000 labels=" + Units.countLabels(mask) + " rasterize=" + rasterMs
                + "ms trace=" + traceMs + "ms polygons=" + polygons.size() + " digest+tiff=" + encodeMs + "ms tiff="
                + tiff.length / 1024 + "kB " + digest.substring(0, 8));
        assertTrue(Units.countLabels(mask) > 3000);
        assertTrue(rasterMs < 5000, "rasterising took " + rasterMs + " ms");
        assertTrue(traceMs < 20000, "tracing took " + traceMs + " ms");
    }

    @Test
    void seriesNumbersAreReadFromServerPaths() {
        assertEquals(0, Images.seriesOf("BioFormatsImageServer: file:/data/a.tif[--series, 0]"));
        assertEquals(3, Images.seriesOf("BioFormatsImageServer: file:/data/a.lif[--series, 3]"));
        assertEquals(0, Images.seriesOf("ImageJServer: file:/data/a.tif"));
        assertEquals(0, Images.seriesOf(null));
    }
}
