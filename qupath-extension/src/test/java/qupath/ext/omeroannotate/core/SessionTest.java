package qupath.ext.omeroannotate.core;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/** The life of a training set: sessions, saving, review, exclusion, failures, several computers. */
class SessionTest {

    @TempDir
    Path home;
    FakeRemoteStore omero;

    /** One rectangle outline per label value (enough to count objects). */
    static final TrainingSet.MaskTracer TRACER = (mask, dx, dy) -> {
        List<RemoteStore.Polygon> polygons = new ArrayList<>();
        java.util.Set<Integer> seen = new java.util.TreeSet<>();
        for (int v : mask.toInts())
            if (v > 0)
                seen.add(v);
        for (int ignored : seen)
            polygons.add(new RemoteStore.Polygon(new double[]{dx, dy, dx + 1, dy, dx + 1, dy + 1}));
        return polygons;
    };

    @BeforeEach
    void setUp() {
        omero = new FakeRemoteStore();
    }

    static PixelArray mask(int... labels) {
        int[] data = new int[16];
        System.arraycopy(labels, 0, data, 0, labels.length);
        return PixelArray.ofLabels(data, 4, 4);
    }

    static PixelArray input() {
        return new PixelArray("u1", 4, 4, 1, new byte[16]);
    }

    private List<Row> plan(boolean omeroImages, int n) {
        List<Planning.ImageRef> images = new ArrayList<>();
        for (int i = 1; i <= n; i++)
            images.add(omeroImages
                    ? new Planning.ImageRef(Ids.omeroUri(i), (long) i, "img" + i, 4, 4, 1, 1)
                    : new Planning.ImageRef("/data/img" + i + ".tif", null, "img" + i + ".tif", 4, 4, 1, 1));
        SetConfig.NewSet spec = spec(!omeroImages, n);
        List<Row> rows = new ArrayList<>();
        Planning.planSplit(images, spec).forEach((uri, category) -> images.stream()
                .filter(i -> i.uri().equals(uri))
                .forEach(i -> rows.addAll(Planning.unitsForImage(i, category, spec))));
        return rows;
    }

    private SetConfig.NewSet spec(boolean local, int n) {
        SetConfig.NewSet spec = new SetConfig.NewSet();
        spec.name = "Nuclei set";
        spec.local = local;
        spec.seed = 1;
        spec.trainN = n - 1;
        spec.validateN = 1;
        spec.containerIds = local ? List.of() : List.of(5L);
        return spec;
    }

    private TrainingSet localSet(int n) throws Exception {
        return TrainingSet.create(SetConfig.create(spec(true, n)), plan(false, n), null, home, "test");
    }

    private TrainingSet omeroSet(int n) throws Exception {
        return TrainingSet.create(SetConfig.create(spec(false, n)), plan(true, n), omero, home, "test");
    }

    // --- Local sets -----------------------------------------------------------------

    @Test
    void localSetIsSavedAndReopened() throws Exception {
        TrainingSet set = localSet(3);
        assertTrue(set.isLocal());
        assertTrue(set.bundle().startsWith(home.resolve("local")));
        assertTrue(set.bundle().getFileName().toString().endsWith("_nuclei-set"));
        assertTrue(Files.exists(set.bundle().resolve("config.yaml")));
        assertTrue(Files.exists(set.bundle().resolve("tracking_table.csv")));
        assertEquals(new TrainingSet.Progress(0, 0, 2), set.progress().get(Row.TRAINING));
        assertEquals(new TrainingSet.Progress(0, 0, 1), set.progress().get(Row.VALIDATION));

        AnnotationSession session = new AnnotationSession(set, false, "QuPath", TRACER);
        assertEquals(1, session.id());
        assertEquals(3, session.remaining());
        Row first = session.next();
        assertEquals(Row.TRAINING, first.category);
        assertEquals(AnnotationSession.Outcome.SAVED, session.commit(first, mask(1, 1, 2), input()).outcome());
        assertTrue(Files.exists(Bundle.maskPath(set.bundle(), first.annotationId)));
        assertTrue(Files.exists(Bundle.inputPath(set.bundle(), first.annotationId)));
        assertEquals(mask(1, 1, 2).digest(), first.maskDigest);
        assertEquals(input().digest(), first.inputDigest);
        assertEquals(1, first.session);

        Row second = session.next();
        session.skip(second);
        Row third = session.next();
        assertEquals(AnnotationSession.Outcome.EXCLUDED, session.exclude(third).outcome());
        assertNull(session.next(), "every unit is handed out once per session");
        session.finish("nuclei only", false);
        assertThrows(IllegalStateException.class, session::next);

        TrainingSet again = TrainingSet.open(set.bundle(), null, "test");
        assertEquals(3, again.rows().size());
        assertTrue(again.row(first.annotationId).isAnnotated());
        assertFalse(again.row(second.annotationId).processed);
        assertTrue(again.row(third.annotationId).isExcluded());
        Map<String, Object> record = again.config().session(1);
        assertEquals("finished", record.get("status"));
        assertEquals(1, record.get("n_annotated"));
        assertEquals("nuclei only", record.get("note"));
        assertEquals("annotate", record.get("kind"));

        // The next session offers only the skipped unit.
        AnnotationSession next = new AnnotationSession(again, false, "QuPath", TRACER);
        assertEquals(2, next.id());
        assertEquals(second.annotationId, next.next().annotationId);
        assertNull(next.next());
        next.finish(null, false);
    }

    @Test
    void reviewReplacesOnlyChangedMasks() throws Exception {
        TrainingSet set = localSet(2);
        AnnotationSession first = new AnnotationSession(set, false, "QuPath", TRACER);
        Row row = first.next();
        first.commit(row, mask(1, 2), input());
        first.finish(null, false);

        AnnotationSession review = new AnnotationSession(set, true, "QuPath", TRACER);
        assertEquals(1, review.remaining());
        Row again = review.next();
        assertEquals(row.annotationId, again.annotationId);
        assertEquals(AnnotationSession.Outcome.UNCHANGED, review.commit(again, mask(1, 2), null).outcome());
        assertEquals(1, again.session, "an unchanged unit keeps its session");
        assertEquals(AnnotationSession.Outcome.REVISED, review.commit(again, mask(1, 2, 3), null).outcome());
        assertEquals(2, again.session);
        assertEquals(mask(1, 2, 3).digest(), again.maskDigest);
        assertArrayEquals(mask(1, 2, 3).toInts(),
                TiffIO.read(Bundle.maskPath(set.bundle(), row.annotationId)).toInts());
        assertThrows(IllegalStateException.class, () -> review.exclude(again));
        review.finish(null, false);
        assertEquals(1, set.config().session(2).get("n_annotated"));
        assertEquals("review", set.config().session(2).get("kind"));
        try (var files = Files.list(Bundle.maskPath(set.bundle(), "x").getParent())) {
            assertEquals(1, files.count(), "no .previous file is left behind");
        }
    }

    @Test
    void excludedUnitCanBeAcceptedAfterAll() throws Exception {
        TrainingSet set = localSet(2);
        AnnotationSession session = new AnnotationSession(set, false, "QuPath", TRACER);
        Row row = session.next();
        session.exclude(row);
        assertTrue(row.isExcluded());
        assertEquals(AnnotationSession.Outcome.SAVED, session.commit(row, mask(1), input()).outcome());
        assertTrue(row.isAnnotated());
        assertEquals("segmentation_mask", row.annotationType);
    }

    @Test
    void anInterruptedSessionIsClosedOnReopen() throws Exception {
        TrainingSet set = localSet(2);
        AnnotationSession session = new AnnotationSession(set, false, "QuPath", TRACER);
        session.commit(session.next(), mask(1), input());
        // no finish(): QuPath crashed
        TrainingSet again = TrainingSet.open(set.bundle(), null, "test");
        assertEquals("interrupted", again.config().session(1).get("status"));
        assertEquals(1, again.config().session(1).get("n_annotated"));
    }

    @Test
    void threeDSetsAreRefused() throws Exception {
        TrainingSet set = localSet(2);
        ((Map<String, Object>) set.config().root().get("spatial_coverage")).put("three_d", true);
        set.saveConfig();
        Exception e = assertThrows(UnsupportedOperationException.class,
                () -> TrainingSet.open(set.bundle(), null, "test"));
        assertTrue(e.getMessage().contains("3D"));
    }

    // --- OMERO sets -----------------------------------------------------------------

    @Test
    void omeroSetUploadsAndUpdatesTheTable() throws Exception {
        TrainingSet set = omeroSet(3);
        long tableId = set.tableId();
        assertFalse(set.isLocal());
        assertEquals(home.resolve("fake-4064").resolve(tableId + "_nuclei-set"), set.bundle());
        assertEquals(3, omero.tables.get(tableId).size());
        assertNotNull(omero.configs.get(tableId));
        assertEquals(tableId, SetConfig.parse(omero.configs.get(tableId)).tableId());

        AnnotationSession session = new AnnotationSession(set, false, "QuPath", TRACER);
        Row row = session.next();
        assertEquals(session.id() + "@annot/" + Ids.hostId(), omero.tableRow(tableId, row.annotationId).get("claimed_by"));
        AnnotationSession.Result result = session.commit(row, mask(1, 1, 2), input());
        assertEquals(AnnotationSession.Outcome.SAVED, result.outcome());
        assertNotNull(row.labelId);
        assertNotNull(row.roiId);
        Map<String, Object> cells = omero.tableRow(tableId, row.annotationId);
        assertEquals(true, cells.get("processed"));
        assertEquals(row.labelId, cells.get("label_id"));
        assertEquals("", cells.get("claimed_by"), "saving releases the claim");
        String description = omero.fileDescriptions.get(row.labelId);
        assertEquals("omero-annotate-ai:" + tableId + ":" + row.annotationId + " | Training set: Nuclei set | t=0 | z=0 | c=0",
                description);
        assertEquals(description, omero.roiDescriptions.get(row.roiId));
        assertEquals(2, omero.roiPolygons.get(row.roiId));
        assertEquals(mask(1, 1, 2).digest(), TiffIO.decode(omero.files.get(row.labelId)).digest());
        assertEquals("Nuclei_set_" + row.annotationId + "_mask.tif", set.labelFileName(row));

        // An empty mask is a valid annotation: a label file, no ROI.
        Row empty = session.next();
        assertEquals(AnnotationSession.Outcome.SAVED, session.commit(empty, mask(), input()).outcome());
        assertNotNull(empty.labelId);
        assertNull(empty.roiId);

        Row held = session.next();
        session.finish("done", false);
        assertEquals("", omero.tableRow(tableId, held.annotationId).get("claimed_by"), "finishing hands units back");
        SetConfig published = SetConfig.parse(omero.configs.get(tableId));
        assertEquals("finished", published.session(1).get("status"));
        assertEquals(2, published.session(1).get("n_annotated"));
        assertEquals("annot", published.session(1).get("user"));
    }

    @Test
    void aFailedUploadKeepsTheMaskAndIsRepairedLater() throws Exception {
        TrainingSet set = omeroSet(2);
        long tableId = set.tableId();
        AnnotationSession session = new AnnotationSession(set, false, "QuPath", TRACER);
        Row row = session.next();
        omero.failAlways = "uploadLabel";
        AnnotationSession.Result result = session.commit(row, mask(1, 2), input());
        assertEquals(AnnotationSession.Outcome.UPLOAD_FAILED, result.outcome());
        assertTrue(result.message().contains("not reachable"));
        assertFalse(row.processed);
        assertTrue(Files.exists(Bundle.maskPath(set.bundle(), row.annotationId)), "the mask is never lost");
        assertEquals(false, omero.tableRow(tableId, row.annotationId).get("processed"));
        assertEquals("", omero.tableRow(tableId, row.annotationId).get("claimed_by"));
        assertEquals(0, session.savedCount());
        assertEquals(1, set.unfinished().size());
        session.finish(null, true);

        // Next day: OMERO is back. Reopening and repairing uploads the saved mask once.
        omero.failAlways = null;
        TrainingSet again = TrainingSet.open(tableId, omero, home, "test");
        assertEquals(1, again.unfinished().size());
        AnnotationSession next = new AnnotationSession(again, false, "QuPath", TRACER);
        assertEquals(1, next.remaining(), "the unit waiting for its upload is not offered for annotation again");
        assertEquals(1, again.repair(TRACER));
        Row repaired = again.row(row.annotationId);
        assertTrue(repaired.isAnnotated());
        assertNull(repaired.error);
        assertEquals(true, omero.tableRow(tableId, row.annotationId).get("processed"));
        assertEquals(1, omero.files.size());
        assertEquals(0, again.unfinished().size());
        assertEquals(0, again.repair(TRACER));
        assertEquals(1, omero.files.size(), "repairing twice uploads nothing twice");
    }

    @Test
    void aCrashAfterUploadAdoptsTheTaggedUpload() throws Exception {
        TrainingSet set = omeroSet(2);
        long tableId = set.tableId();
        AnnotationSession session = new AnnotationSession(set, false, "QuPath", TRACER);
        Row row = session.next();
        omero.failAlways = "writeRows"; // label and ROI go up, the table update fails
        AnnotationSession.Result result = session.commit(row, mask(1, 2), input());
        assertEquals(AnnotationSession.Outcome.UPLOAD_FAILED, result.outcome());
        assertEquals(1, omero.files.size());
        assertEquals(false, omero.tableRow(tableId, row.annotationId).get("processed"));
        omero.failAlways = null;

        // Even if the local table lost the IDs (crash before it was written), the tag finds the uploads.
        TrainingSet again = TrainingSet.open(tableId, omero, home, "test");
        Row reopened = again.row(row.annotationId);
        assertTrue(reopened.processed, "the local folder still knows the unit is saved");
        reopened.labelId = null;
        reopened.roiId = null;
        assertEquals(1, again.repair(TRACER));
        assertEquals(1, omero.files.size(), "the tagged upload is adopted, not repeated");
        assertEquals(1, omero.roiDescriptions.size());
        assertEquals(true, omero.tableRow(tableId, row.annotationId).get("processed"));
        assertEquals(reopened.labelId, omero.tableRow(tableId, row.annotationId).get("label_id"));
    }

    @Test
    void twoComputersNeverAnnotateOrUploadTheSameUnit() throws Exception {
        TrainingSet mine = omeroSet(2);
        long tableId = mine.tableId();
        // A colleague on another computer opens the same set.
        FakeRemoteStore theirOmero = omero;
        Path theirHome = home.resolve("other-computer");
        TrainingSet theirs = TrainingSet.open(tableId, theirOmero, theirHome, "test");
        AnnotationSession mySession = new AnnotationSession(mine, false, "QuPath", TRACER);
        Row myRow = mySession.next();

        // Their claim, as another user/computer would write it.
        String foreign = "9@colleague/ffffffff";
        Row second = mine.rows().stream().filter(r -> r != myRow).findFirst().orElseThrow();
        omero.tableRow(tableId, second.annotationId).put("claimed_by", foreign);
        omero.tableRow(tableId, second.annotationId).put("claimed_at", TrainingSet.claimNow());
        assertNull(mySession.next(), "a unit claimed elsewhere is skipped");
        assertEquals(1, mySession.heldElsewhere());

        // An old claim (a crashed session) expires.
        omero.tableRow(tableId, second.annotationId).put("claimed_at", "2020-01-01T00:00:00+00:00");
        assertEquals(second.annotationId, mySession.next().annotationId);

        // The colleague saves my unit first; my commit must not upload a second mask.
        Row theirRow = theirs.row(myRow.annotationId);
        theirRow.markProcessed("segmentation_mask");
        theirRow.labelId = 555L;
        theirRow.session = 9;
        omero.writeRows(tableId, List.of(theirRow));
        int filesBefore = omero.files.size();
        AnnotationSession.Result result = mySession.commit(myRow, mask(1), input());
        assertEquals(AnnotationSession.Outcome.SAVED_ELSEWHERE, result.outcome());
        assertEquals(filesBefore, omero.files.size());
        assertEquals(555L, myRow.labelId, "their row is taken over");
        assertTrue(Files.exists(Bundle.conflictPath(mine.bundle(), myRow.annotationId, mySession.id())));
        assertEquals(0, mySession.savedCount());
    }

    @Test
    void ownLeftoverClaimsNeverBlock() throws Exception {
        TrainingSet set = omeroSet(2);
        AnnotationSession crashed = new AnnotationSession(set, false, "QuPath", TRACER);
        Row row = crashed.next();
        // no finish(): the claim of session 1 stays in the table
        AnnotationSession next = new AnnotationSession(set, false, "QuPath", TRACER);
        assertEquals(row.annotationId, next.next().annotationId);
    }

    @Test
    void reviewOnOmeroReplacesUploads() throws Exception {
        TrainingSet set = omeroSet(2);
        long tableId = set.tableId();
        AnnotationSession first = new AnnotationSession(set, false, "QuPath", TRACER);
        Row row = first.next();
        first.commit(row, mask(1, 2), input());
        first.finish(null, false);
        long oldLabel = row.labelId, oldRoi = row.roiId;

        AnnotationSession review = new AnnotationSession(set, true, "QuPath", TRACER);
        Row again = review.next();
        assertEquals(AnnotationSession.Outcome.UNCHANGED, review.commit(again, mask(1, 2), null).outcome());
        assertEquals(oldLabel, again.labelId);

        assertEquals(AnnotationSession.Outcome.REVISED, review.commit(again, mask(1, 2, 3), null).outcome());
        assertNotEquals(oldLabel, again.labelId);
        assertFalse(omero.files.containsKey(oldLabel), "the old label file is deleted");
        assertFalse(omero.roiDescriptions.containsKey(oldRoi));
        assertEquals(1, omero.files.size());
        assertEquals(again.labelId, omero.tableRow(tableId, row.annotationId).get("label_id"));
        assertEquals(3, omero.roiPolygons.get(again.roiId));

        // A failed replacement leaves the earlier mask and uploads in place.
        long keptLabel = again.labelId;
        omero.failAlways = "uploadRois";
        AnnotationSession.Result failed = review.commit(again, mask(4), null);
        assertEquals(AnnotationSession.Outcome.UPLOAD_FAILED, failed.outcome());
        assertEquals(keptLabel, again.labelId);
        assertEquals(1, omero.files.size(), "the half-uploaded new label is removed");
        assertArrayEquals(mask(1, 2, 3).toInts(),
                TiffIO.read(Bundle.maskPath(set.bundle(), row.annotationId)).toInts());
        assertTrue(Files.exists(Bundle.conflictPath(set.bundle(), row.annotationId, review.id())));
    }

    @Test
    void sessionsOfOtherComputersAreKeptInTheConfig() throws Exception {
        TrainingSet set = omeroSet(2);
        long tableId = set.tableId();
        // Another computer publishes session 1 at the same moment.
        SetConfig theirs = SetConfig.parse(omero.configs.get(tableId));
        Map<String, Object> other = theirs.addSession(1, "annotate", "colleague", "0.6.0", "vit_b_lm");
        other.put("host", "ffffffff");
        omero.configs.put(tableId, theirs.toYaml());

        AnnotationSession session = new AnnotationSession(set, false, "QuPath", TRACER);
        assertEquals(2, session.id(), "a taken session number is not reused");
        session.finish(null, false);
        SetConfig published = SetConfig.parse(omero.configs.get(tableId));
        assertEquals(2, published.sessions().size());
        assertEquals("colleague", published.session(1).get("user"));
        assertEquals("annot", published.session(2).get("user"));
    }

    @Test
    void omeroSetsNeedOmeroImagesAndAContainer() {
        SetConfig.NewSet spec = spec(false, 2);
        assertThrows(IllegalArgumentException.class,
                () -> TrainingSet.create(SetConfig.create(spec), plan(false, 2), omero, home, "test"));
        spec.containerIds = List.of();
        assertThrows(IllegalArgumentException.class,
                () -> TrainingSet.create(SetConfig.create(spec), plan(true, 2), omero, home, "test"));
        assertThrows(IllegalArgumentException.class,
                () -> TrainingSet.create(SetConfig.create(spec(false, 2)), plan(true, 2), null, home, "test"));
    }

    @Test
    void workIsSavedLocallyWhenOmeroDoesNotAnswerAtAll() throws Exception {
        TrainingSet set = omeroSet(3);
        long tableId = set.tableId();
        AnnotationSession session = new AnnotationSession(set, false, "QuPath", TRACER);
        Row row = session.next();
        Row other = session.next();
        omero.failAlways = "readRows"; // the very first call of a save fails: the server is gone
        AnnotationSession.Result result = session.commit(row, mask(1, 2), input());
        assertEquals(AnnotationSession.Outcome.UPLOAD_FAILED, result.outcome());
        assertTrue(Files.exists(Bundle.maskPath(set.bundle(), row.annotationId)), "the mask is on disk");
        assertTrue(Files.exists(Bundle.inputPath(set.bundle(), row.annotationId)));
        assertEquals(0, omero.files.size(), "nothing was uploaded");
        AnnotationSession.Result excluded = session.exclude(other);
        assertEquals(AnnotationSession.Outcome.UPLOAD_FAILED, excluded.outcome());
        assertTrue(other.isExcluded(), "the exclusion is kept locally");
        assertEquals(2, set.unfinished().size());

        omero.failAlways = null; // the server is back
        assertEquals(2, set.repair(TRACER));
        assertEquals(true, omero.tableRow(tableId, row.annotationId).get("processed"));
        assertEquals("excluded", omero.tableRow(tableId, other.annotationId).get("annotation_type"));
        assertEquals(1, omero.files.size());
        assertTrue(set.unfinished().isEmpty());
    }

    @Test
    void aServerThatNeverAnswersFailsAfterItsTimeLimit() throws Exception {
        FakeRemoteStore frozen = new FakeRemoteStore() {
            @Override
            public java.util.List<java.util.Map<String, Object>> readTable(long tableId) throws Exception {
                Thread.sleep(60_000);
                return super.readTable(tableId);
            }
        };
        RemoteStore limited = TimeLimitedStore.wrap(frozen, 1);
        long start = System.nanoTime();
        Exception e = assertThrows(java.io.IOException.class, () -> limited.readTable(1));
        assertTrue(e.getMessage().contains("did not answer"), e.getMessage());
        assertTrue((System.nanoTime() - start) / 1_000_000 < 5_000);
        assertEquals("fake-4064", limited.serverSlug());
        // Right after that, further calls fail at once instead of waiting again.
        start = System.nanoTime();
        assertThrows(java.io.IOException.class, () -> limited.readConfig(1));
        assertTrue((System.nanoTime() - start) / 1_000_000 < 500);

        // A server that answers: errors keep their own message, ordinary calls pass through.
        FakeRemoteStore healthy = new FakeRemoteStore();
        RemoteStore wrapped = TimeLimitedStore.wrap(healthy, 5);
        healthy.failAlways = "readConfig";
        Exception inner = assertThrows(java.io.IOException.class, () -> wrapped.readConfig(1));
        assertTrue(inner.getMessage().contains("not reachable"), inner.getMessage());
        assertTrue(wrapped.listSets("dataset", 1).isEmpty());
    }

    @Test
    void aChangedMaskAfterAHalfUploadReplacesTheHalfUpload() throws Exception {
        TrainingSet set = omeroSet(2);
        long tableId = set.tableId();
        AnnotationSession session = new AnnotationSession(set, false, "QuPath", TRACER);
        Row row = session.next();
        omero.failOnce = "uploadRois"; // the label file goes up, the ROI does not
        assertEquals(AnnotationSession.Outcome.UPLOAD_FAILED, session.commit(row, mask(1, 2), input()).outcome());
        Long firstLabel = row.labelId;
        assertNotNull(firstLabel);
        assertNull(row.roiId);

        // The annotator corrects the objects and accepts again.
        assertEquals(AnnotationSession.Outcome.SAVED, session.commit(row, mask(1, 2, 3), null).outcome());
        assertNotEquals(firstLabel, row.labelId, "the label of the old mask is not kept");
        assertFalse(omero.files.containsKey(firstLabel));
        assertEquals(1, omero.files.size());
        assertEquals(mask(1, 2, 3).digest(), TiffIO.decode(omero.files.get(row.labelId)).digest());
        assertEquals(3, omero.roiPolygons.get(row.roiId));
        assertEquals(row.labelId, omero.tableRow(tableId, row.annotationId).get("label_id"));
    }

    @Test
    void theSameMaskAfterAHalfUploadOnlyAddsWhatIsMissing() throws Exception {
        TrainingSet set = omeroSet(2);
        AnnotationSession session = new AnnotationSession(set, false, "QuPath", TRACER);
        Row row = session.next();
        omero.failOnce = "uploadRois";
        session.commit(row, mask(1, 2), input());
        Long firstLabel = row.labelId;
        assertEquals(AnnotationSession.Outcome.SAVED, session.commit(row, mask(1, 2), null).outcome());
        assertEquals(firstLabel, row.labelId, "the label already uploaded is used");
        assertEquals(1, omero.files.size());
        assertNotNull(row.roiId);
    }

    @Test
    void aCorrectionWhoseTableUpdateFailsIsCompletedNotReverted() throws Exception {
        TrainingSet set = omeroSet(2);
        long tableId = set.tableId();
        AnnotationSession first = new AnnotationSession(set, false, "QuPath", TRACER);
        Row row = first.next();
        first.commit(row, mask(1, 2), input());
        first.finish(null, false);
        long oldLabel = row.labelId, oldRoi = row.roiId;

        AnnotationSession review = new AnnotationSession(set, true, "QuPath", TRACER);
        Row again = review.next();
        omero.failAlways = "writeRows";
        assertEquals(AnnotationSession.Outcome.UPLOAD_FAILED, review.commit(again, mask(1, 2, 3), null).outcome());
        omero.failAlways = null;
        long newLabel = again.labelId;
        assertNotEquals(oldLabel, newLabel);
        assertTrue(omero.files.containsKey(oldLabel), "the label the table still names is kept");
        assertEquals(oldLabel, omero.tableRow(tableId, row.annotationId).get("label_id"));

        // Reopening (QuPath was closed in between) keeps the correction and completes it.
        TrainingSet reopened = TrainingSet.open(tableId, omero, home, "test");
        Row pending = reopened.row(row.annotationId);
        assertEquals(newLabel, pending.labelId, "the folder's newer row is not thrown away");
        assertEquals(1, reopened.unfinished().size());
        assertEquals(1, reopened.repair(TRACER));
        assertEquals(newLabel, omero.tableRow(tableId, row.annotationId).get("label_id"));
        assertEquals(mask(1, 2, 3).digest(), omero.tableRow(tableId, row.annotationId).get("mask_digest"));
        assertFalse(omero.files.containsKey(oldLabel), "the replaced label is removed once the table is right");
        assertFalse(omero.roiDescriptions.containsKey(oldRoi));
        assertEquals(1, omero.files.size());
        assertTrue(reopened.unfinished().isEmpty());
    }

    @Test
    void aReviewDoesNotOverwriteANewerCorrectionFromElsewhere() throws Exception {
        TrainingSet set = omeroSet(2);
        long tableId = set.tableId();
        AnnotationSession first = new AnnotationSession(set, false, "QuPath", TRACER);
        Row row = first.next();
        first.commit(row, mask(1, 2), input());
        first.finish(null, false);

        AnnotationSession review = new AnnotationSession(set, true, "QuPath", TRACER);
        Row mine = review.next();
        // Meanwhile a colleague corrects the same unit (their label has a higher ID).
        Row theirs = mine.copy();
        theirs.labelId = omero.uploadLabel(mine.imageId, TiffIO.encode(mask(5), true), "x.tif", set.uploadDescription(mine));
        theirs.maskDigest = mask(5).digest();
        theirs.session = 9;
        omero.writeRows(tableId, List.of(theirs));
        int files = omero.files.size();

        AnnotationSession.Result result = review.commit(mine, mask(1, 2, 3), null);
        assertEquals(AnnotationSession.Outcome.SAVED_ELSEWHERE, result.outcome());
        assertEquals(files, omero.files.size(), "nothing was uploaded over their correction");
        assertEquals(theirs.labelId, mine.labelId);
        assertEquals(theirs.labelId, omero.tableRow(tableId, row.annotationId).get("label_id"));
        assertTrue(Files.exists(Bundle.conflictPath(set.bundle(), row.annotationId, review.id())));
    }

    @Test
    void aMaskWithOtherLabelNumbersIsTheSameMask() throws Exception {
        // Cellpose and micro-SAM number objects in their own order; QuPath from the top left.
        TrainingSet set = localSet(2);
        AnnotationSession first = new AnnotationSession(set, false, "QuPath", TRACER);
        Row row = first.next();
        first.commit(row, mask(7, 7, 3, 0, 9), input());
        first.finish(null, false);
        AnnotationSession review = new AnnotationSession(set, true, "QuPath", TRACER);
        Row again = review.next();
        assertEquals(AnnotationSession.Outcome.UNCHANGED, review.commit(again, mask(1, 1, 2, 0, 3), null).outcome());
        assertEquals(1, again.session);
        // Two objects merged into one, or one split in two, is a change.
        assertEquals(AnnotationSession.Outcome.REVISED, review.commit(again, mask(1, 1, 1, 0, 3), null).outcome());
        assertEquals(AnnotationSession.Outcome.REVISED, review.commit(again, mask(1, 2, 3, 0, 4), null).outcome());
    }

    @Test
    void leftoverRoisOfACutOffUploadAreNotDuplicated() throws Exception {
        TrainingSet set = omeroSet(2);
        AnnotationSession session = new AnnotationSession(set, false, "QuPath", TRACER);
        Row row = session.next();
        // The ROI upload reaches OMERO, but its answer never arrives: the ID is not known here.
        FakeRemoteStore real = omero;
        omero.failOnce = "uploadRois";
        session.commit(row, mask(1, 2), input());
        real.roiDescriptions.put(900L, set.uploadDescription(row));
        real.roiImages.put(900L, row.imageId);
        real.roiPolygons.put(900L, 2);
        assertEquals(AnnotationSession.Outcome.SAVED, session.commit(row, mask(1, 2), null).outcome());
        assertEquals(1, real.roiDescriptions.size(), "one set of ROIs, not two");
        assertFalse(real.roiDescriptions.containsKey(900L));
    }

    @Test
    void aFolderOfAnotherServerIsRefused() throws Exception {
        TrainingSet set = omeroSet(2);
        long tableId = set.tableId();
        // A second server (a test copy) with a table of the same ID holding another set.
        FakeRemoteStore staging = new FakeRemoteStore() {
            @Override public String serverSlug() { return "staging-4064"; }
        };
        staging.tables.put(tableId, omero.readTable(tableId));
        SetConfig other = SetConfig.parse(omero.configs.get(tableId));
        other.root().put("name", "something else");
        staging.configs.put(tableId, other.toYaml());
        Exception e = assertThrows(java.io.IOException.class, () -> TrainingSet.open(set.bundle(), staging, "test"));
        assertTrue(e.getMessage().contains("staging-4064"), e.getMessage());
        // The same server under another name (same set on it) is fine.
        staging.configs.put(tableId, omero.configs.get(tableId));
        assertEquals(2, TrainingSet.open(set.bundle(), staging, "test").rows().size());
    }

    @Test
    void ownClaimsLeftByACrashAreReleasedOnOpen() throws Exception {
        TrainingSet set = omeroSet(2);
        long tableId = set.tableId();
        AnnotationSession crashed = new AnnotationSession(set, false, "QuPath", TRACER);
        Row row = crashed.next();
        assertNotEquals("", omero.tableRow(tableId, row.annotationId).get("claimed_by"));
        TrainingSet.open(tableId, omero, home, "test");
        assertEquals("", omero.tableRow(tableId, row.annotationId).get("claimed_by"));
    }
}
