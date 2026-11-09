package qupath.ext.omeroannotate.omero;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.junit.jupiter.api.io.TempDir;
import qupath.ext.omeroannotate.core.AnnotationSession;
import qupath.ext.omeroannotate.core.Ids;
import qupath.ext.omeroannotate.core.PixelArray;
import qupath.ext.omeroannotate.core.Planning;
import qupath.ext.omeroannotate.core.RemoteStore;
import qupath.ext.omeroannotate.core.Row;
import qupath.ext.omeroannotate.core.RowCodec;
import qupath.ext.omeroannotate.core.SetConfig;
import qupath.ext.omeroannotate.core.TiffIO;
import qupath.ext.omeroannotate.core.TrainingSet;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * The OMERO backend against a real server. Runs only with
 * {@code -Domero.test.host=localhost} (see the README for the Docker test server); the
 * server needs a dataset with at least four images, given as {@code -Domero.test.dataset}.
 */
@EnabledIfSystemProperty(named = "omero.test.host", matches = ".+")
class OmeroIntegrationTest {

    static OmeroGatewayStore omero;
    static long datasetId;

    static final TrainingSet.MaskTracer BOXES = (mask, dx, dy) -> {
        List<RemoteStore.Polygon> polygons = new ArrayList<>();
        int[] labels = mask.toInts();
        java.util.Map<Integer, int[]> boxes = new java.util.TreeMap<>();
        for (int i = 0; i < labels.length; i++) {
            if (labels[i] == 0)
                continue;
            int x = i % mask.width(), y = i / mask.width();
            int[] b = boxes.computeIfAbsent(labels[i], k -> new int[]{x, y, x, y});
            b[0] = Math.min(b[0], x);
            b[1] = Math.min(b[1], y);
            b[2] = Math.max(b[2], x);
            b[3] = Math.max(b[3], y);
        }
        for (int[] b : boxes.values())
            polygons.add(new RemoteStore.Polygon(new double[]{dx + b[0], dy + b[1], dx + b[2] + 1, dy + b[1],
                    dx + b[2] + 1, dy + b[3] + 1, dx + b[0], dy + b[3] + 1}));
        return polygons;
    };

    @BeforeAll
    static void connect() throws Exception {
        omero = OmeroGatewayStore.connect(System.getProperty("omero.test.host"),
                Integer.parseInt(System.getProperty("omero.test.port", "6064")),
                System.getProperty("omero.test.user", "annot"),
                System.getProperty("omero.test.password", "annot-pw"));
        datasetId = Long.parseLong(System.getProperty("omero.test.dataset", "1"));
    }

    @AfterAll
    static void disconnect() {
        if (omero != null)
            omero.close();
    }

    static PixelArray squares(int width, int height, int n) {
        int[] labels = new int[width * height];
        for (int k = 0; k < n; k++)
            for (int y = 10 + 30 * k; y < 30 + 30 * k; y++)
                for (int x = 20; x < 60; x++)
                    labels[y * width + x] = k + 1;
        return PixelArray.ofLabels(labels, width, height);
    }

    @Test
    void aSetLivesOnOmero(@TempDir Path home, @TempDir Path otherComputer) throws Exception {
        List<RemoteStore.RemoteImage> images = omero.listImages("dataset", datasetId);
        assertTrue(images.size() >= 4, "the test dataset needs at least four images");
        RemoteStore.RemoteImage any = images.get(0);
        assertTrue(any.width() > 0 && any.height() > 0 && any.sizeC() > 0);
        assertFalse(omero.containersOfImage(any.id()).isEmpty());

        // Create
        SetConfig.NewSet spec = new SetConfig.NewSet();
        spec.name = "qupath java test " + System.currentTimeMillis();
        spec.local = false;
        spec.containerIds = List.of(datasetId);
        spec.trainN = 2;
        spec.validateN = 1;
        spec.testN = 1;
        List<Planning.ImageRef> refs = new ArrayList<>();
        for (RemoteStore.RemoteImage image : images)
            refs.add(new Planning.ImageRef(Ids.omeroUri(image.id()), image.id(), image.name(), image.width(),
                    image.height(), image.sizeZ(), image.sizeT()));
        List<Row> rows = new ArrayList<>();
        Planning.planSplit(refs, spec).forEach((uri, category) -> refs.stream().filter(r -> r.uri().equals(uri))
                .forEach(r -> rows.addAll(Planning.unitsForImage(r, category, spec))));
        TrainingSet set = TrainingSet.create(SetConfig.create(spec), rows, omero, home, "test");
        long tableId = set.tableId();
        System.out.println("JAVA_TABLE_ID=" + tableId);
        assertTrue(omero.listSets("dataset", datasetId).stream().anyMatch(s -> s.tableId() == tableId
                && s.name().equals(spec.name)));
        assertEquals(4, omero.readTable(tableId).size());
        SetConfig published = SetConfig.parse(omero.readConfig(tableId));
        assertEquals(tableId, published.tableId());
        assertEquals(omero.serverSlug(), published.server());

        // Annotate one unit
        AnnotationSession session = new AnnotationSession(set, false, "QuPath test", BOXES);
        Row row = session.next();
        Map<String, Object> claimed = omero.readRows(tableId, List.of(row.annotationId)).get(0);
        assertEquals(set.token(session.id()), claimed.get("claimed_by"));
        RemoteStore.RemoteImage image = images.stream().filter(i -> i.id() == row.imageId).findFirst().orElseThrow();
        PixelArray mask = squares(image.width(), image.height(), 3);
        AnnotationSession.Result result = session.commit(row, mask,
                new PixelArray("u2", image.width(), image.height(), 1, new byte[image.width() * image.height() * 2]));
        assertEquals(AnnotationSession.Outcome.SAVED, result.outcome(), result.message());
        Row stored = RowCodec.fromCells(omero.readRows(tableId, List.of(row.annotationId)).get(0));
        assertTrue(stored.processed);
        assertEquals(row.labelId, stored.labelId);
        assertEquals(row.roiId, stored.roiId);
        assertEquals(mask.digest(), stored.maskDigest);
        assertEquals("", stored.claimedBy);
        assertEquals(session.id(), stored.session);
        RemoteStore.Tagged tagged = omero.findTagged(row.imageId, set.tag(row.annotationId));
        assertEquals(List.of(row.labelId), tagged.labelIds());
        assertEquals(List.of(row.roiId), tagged.roiIds());
        assertEquals(mask.digest(), TiffIO.decode(omero.downloadFile(row.labelId)).digest());

        // Exclude another, skip a third
        Row excluded = session.next();
        assertEquals(AnnotationSession.Outcome.EXCLUDED, session.exclude(excluded).outcome());
        Row skipped = session.next();
        session.skip(skipped);
        assertEquals("", omero.readRows(tableId, List.of(skipped.annotationId)).get(0).get("claimed_by"));
        session.finish("first pass", false);

        // A second computer opens the same set from OMERO alone.
        TrainingSet theirs = TrainingSet.open(tableId, omero, otherComputer, "test");
        assertEquals(4, theirs.rows().size());
        assertTrue(theirs.row(row.annotationId).isAnnotated());
        assertTrue(theirs.row(excluded.annotationId).isExcluded());
        assertEquals("finished", theirs.config().session(1).get("status"));
        assertEquals("first pass", theirs.config().session(1).get("note"));

        // Review there: a changed mask replaces the uploads.
        AnnotationSession review = new AnnotationSession(theirs, true, "QuPath test", BOXES);
        Row again = review.next();
        assertEquals(row.annotationId, again.annotationId);
        long oldLabel = again.labelId, oldRoi = again.roiId;
        assertEquals(AnnotationSession.Outcome.REVISED,
                review.commit(again, squares(image.width(), image.height(), 4), null).outcome());
        review.finish(null, false);
        RemoteStore.Tagged after = omero.findTagged(row.imageId, set.tag(row.annotationId));
        assertEquals(List.of(again.labelId), after.labelIds(), "the old label file is gone");
        assertEquals(List.of(again.roiId), after.roiIds());
        assertNotEquals(oldLabel, again.labelId);
        assertNotEquals(oldRoi, again.roiId);
        assertNull(omero.downloadFile(oldLabel));

        // The first computer sees it on its next refresh.
        set.refresh();
        assertEquals(again.labelId, set.row(row.annotationId).labelId);
        assertEquals(2, SetConfig.parse(omero.readConfig(tableId)).sessions().size());
    }

    @Test
    void oldOrForeignTablesAreRefused() {
        assertThrows(IllegalArgumentException.class, () -> omero.readTable(999_999_999L));
    }
}
