package qupath.ext.omeroannotate.core;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/** File formats and identifiers shared with the Python package. Reference values come from Python. */
class FormatsTest {

    private static PixelArray ramp(String dtype, int width, int height, int samples) {
        int n = width * height * samples;
        ByteBuffer b = ByteBuffer.allocate(n * PixelArray.bytesPerSample(dtype)).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < n; i++) {
            switch (dtype) {
                case "u1" -> b.put((byte) i);
                case "u2" -> b.putShort((short) i);
                case "u4" -> b.putInt(i * 70000);
                case "f4" -> b.putFloat(i / 4f);
                default -> throw new IllegalArgumentException(dtype);
            }
        }
        return new PixelArray(dtype, width, height, samples, b.array());
    }

    @Test
    void pixelDigestEqualsPython() {
        // omero_annotate_ai.bundle.pixel_digest of np.arange(12).reshape(3, 4) etc.
        assertEquals("8385dd413fb519b54cec289c8e112a08e60c3025f70a4b5efe6f7006847f1b8b", ramp("u2", 4, 3, 1).digest());
        assertEquals("b1afa0e8157b9f1941e20b83b86af8d7c62a49a692ee411a1bae0d41666a53b3", ramp("u1", 4, 3, 1).digest());
        assertEquals("1532caf094768c99c343c903cd88d62d50f12a86bbb09abb76b09aca44cc06c5", ramp("u4", 4, 3, 1).digest());
        assertEquals("6c6c97fd125de453859707abb76210ee0338156e4fab34684441d99103c0baba", ramp("f4", 4, 3, 1).digest());
        assertEquals("f53bd83888f6895d11170a90d9346aba2197e7125fb87cfdff5f86d970bf8cf2", ramp("u1", 4, 2, 3).digest());
    }

    @Test
    void identifiersEqualPython() {
        assertEquals("omero:image:42", Ids.omeroUri(42));
        assertEquals(42L, Ids.parseOmeroUri("omero:image:42"));
        assertNull(Ids.parseOmeroUri("/data/image.tif"));
        assertNull(Ids.parseOmeroUri("omero:image:x"));
        assertEquals("e32e7635", Ids.uriKey("omero:image:42")); // hashlib.sha1(...).hexdigest()[:8]
        assertEquals("e32e7635_0_5", Ids.annotationId("omero:image:42", 0, 5, -1));
        assertEquals("e32e7635_2_0_3", Ids.annotationId("omero:image:42", 2, 0, 3));
        assertEquals("a5638e33", Ids.hex("SHA-256", "myhost".getBytes()).substring(0, 8));
    }

    @Test
    void slugsEqualPython() {
        assertEquals("nuclei-set_1", Bundle.slug("Nuclei Set_1"));
        assertEquals("unnamed", Bundle.slug("  !! "));
        assertEquals("a.b", Bundle.slug(".a.b."));
        assertEquals("omero.example.org-4064", Bundle.serverSlug("Omero.Example.org", 4064));
        assertEquals("example.org-omero-ws", Bundle.serverSlug("wss://example.org/omero-ws/", 0));
    }

    @Test
    void tiffRoundTrip() throws Exception {
        for (boolean compress : new boolean[]{false, true}) {
            for (PixelArray image : List.of(ramp("u1", 7, 5, 1), ramp("u2", 7, 5, 1), ramp("u4", 7, 5, 1),
                    ramp("f4", 7, 5, 1), ramp("u1", 7, 5, 3))) {
                PixelArray back = TiffIO.decode(TiffIO.encode(image, compress));
                assertEquals(image.dtype(), back.dtype());
                assertEquals(image.width(), back.width());
                assertEquals(image.height(), back.height());
                assertEquals(image.samples(), back.samples());
                assertArrayEquals(image.bytes(), back.bytes());
                assertEquals(image.digest(), back.digest());
            }
        }
    }

    @Test
    void tiffCompressionIsSmallerForMasks() {
        int[] labels = new int[512 * 512];
        for (int i = 0; i < labels.length; i++)
            labels[i] = (i / 512 > 100 && i / 512 < 200 && i % 512 > 100 && i % 512 < 200) ? 1 : 0;
        PixelArray mask = PixelArray.ofLabels(labels, 512, 512);
        assertEquals("u2", mask.dtype());
        assertTrue(TiffIO.encode(mask, true).length < TiffIO.encode(mask, false).length / 20);
        assertArrayEquals(labels, mask.toInts());
    }

    @Test
    void labelsAbove16BitUse32Bit() {
        int[] labels = {0, 1, 70000, 5};
        PixelArray mask = PixelArray.ofLabels(labels, 2, 2);
        assertEquals("u4", mask.dtype());
        assertArrayEquals(labels, mask.toInts());
    }

    @Test
    void tiffRefusesWhatItCannotRead() {
        assertThrows(java.io.IOException.class, () -> TiffIO.decode(new byte[]{1, 2, 3}));
        assertThrows(java.io.IOException.class, () -> TiffIO.decode("not a tiff file at all".getBytes()));
    }

    @Test
    void csvRoundTrip() {
        Row row = new Row();
        row.imageId = 12L;
        row.imageUri = "omero:image:12";
        row.imageName = "plate 1, well \"B2\"\nfield 3";
        row.annotationId = "abc_0_0";
        row.category = Row.VALIDATION;
        row.channel = 1;
        row.timepoint = 0;
        row.zSlice = 3;
        row.zStart = 3;
        row.zEnd = 3;
        row.isPatch = true;
        row.patchX = 10;
        row.patchY = 20;
        row.patchWidth = 256;
        row.patchHeight = 128;
        row.wellMetadata.put("treatment", "DMSO, 0.1%");
        row.wellMetadata.put("cell line", "HeLa");
        row.markProcessed("segmentation_mask");
        row.session = 2;
        row.labelId = 99L;
        row.maskDigest = "ab".repeat(32);
        row.error = "IOException: boom, again";
        Row open = new Row();
        open.imageUri = "/data/images/a.tif";
        open.imageName = "a.tif";
        open.annotationId = "def_0_0";

        String csv = RowCodec.toCsv(List.of(row, open));
        assertTrue(csv.startsWith("image_id,image_uri,image_name,annotation_id,category,"));
        List<Row> back = RowCodec.fromCsv(csv);
        assertEquals(2, back.size());
        Row r = back.get(0);
        assertEquals(RowCodec.toCells(row), RowCodec.toCells(r));
        assertEquals(row.error, r.error);
        assertEquals(Map.of("treatment", "DMSO, 0.1%", "cell line", "HeLa"), r.wellMetadata);
        Row o = back.get(1);
        assertNull(o.imageId);
        assertNull(o.labelId);
        assertNull(o.annotationCreatedAt);
        assertNull(o.maskDigest);
        assertFalse(o.processed);
        assertEquals("/data/images/a.tif", o.imageUri);
    }

    @Test
    void csvWrittenByPandasIsRead() {
        // tracking_table.csv as pandas writes it (booleans True/False, missing times "None").
        String csv = "image_id,image_uri,image_name,annotation_id,category,channel,z_slice,timepoint,label_id,"
                + "roi_id,is_volumetric,processed,is_patch,patch_x,patch_y,patch_width,patch_height,"
                + "annotation_type,annotation_created_at,annotation_updated_at,annotation_input_id,session,"
                + "z_start,z_end,z_length,level,well,field,well_metadata,input_digest,mask_digest,input_iscc,"
                + "mask_iscc,claimed_by,claimed_at,error\r\n"
                + "5,omero:image:5,img.tif,4ba674d4_0_0,training,0,0,0,31,7,False,True,False,0,0,0,0,"
                + "segmentation_mask,2026-01-02T03:04:05.000006,2026-01-02T03:04:05.000006,0,1,0,0,1,0,B2,0,"
                + "\"{\"\"treatment\"\": \"\"DMSO\"\"}\",aa,bb,,,,,\r\n"
                + "6,omero:image:6,img2.tif,5ba674d4_0_0,test,0,0,0,0,0,False,False,False,0,0,0,0,"
                + "segmentation_mask,None,None,0,-1,0,0,1,0,,,{},,,,,3@annot/abcd1234,2026-01-02T03:04:05+00:00,\r\n";
        List<Row> rows = RowCodec.fromCsv(csv);
        assertEquals(2, rows.size());
        assertTrue(rows.get(0).processed);
        assertEquals(31L, rows.get(0).labelId);
        assertEquals(7L, rows.get(0).roiId);
        assertEquals("DMSO", rows.get(0).wellMetadata.get("treatment"));
        assertEquals("B2", rows.get(0).well);
        assertFalse(rows.get(1).processed);
        assertEquals(Row.TEST, rows.get(1).category);
        assertNull(rows.get(1).labelId);
        assertEquals("3@annot/abcd1234", rows.get(1).claimedBy);
        assertEquals(-1, rows.get(1).session);
    }

    @Test
    void oldTablesAreRefused() {
        assertThrows(IllegalArgumentException.class, () -> RowCodec.fromCsv("image_id,processed\n1,True\n"));
    }

    @Test
    void configRoundTripKeepsUnknownKeysAndDates() {
        String yaml = """
                schema_version: 3
                name: nuclei
                created: 2025-01-14 10:30:00.123456
                study:
                  title: 'yes'
                  description: Über-study
                  keywords:
                  - nuclei
                ai_model:
                  framework: micro_sam
                  pretrained_from: vit_b_lm
                spatial_coverage:
                  channels:
                  - 2
                  three_d: false
                workflow:
                  batch_size: 0
                  claim_hours: 6.0
                storage: omero
                identity:
                  server: omero.example.org-4064
                  table_id: 123
                  local_uuid: null
                sessions:
                - id: 1
                  kind: annotate
                  started: 2025-01-14 10:31:00.000001
                  ended: null
                  status: finished
                  host: abcd1234
                  n_annotated: 4
                  note: null
                """;
        SetConfig config = SetConfig.parse(yaml);
        assertEquals("nuclei", config.name());
        assertEquals(123L, config.tableId());
        assertFalse(config.isLocal());
        assertEquals(2, config.labelChannel());
        assertEquals(6.0, config.claimHours());
        assertEquals(1, config.sessions().size());
        String out = config.toYaml();
        assertTrue(out.contains("created: 2025-01-14 10:30:00.123456"), out);
        assertTrue(out.contains("started: 2025-01-14 10:31:00.000001"), out);
        assertTrue(out.contains("title: 'yes'"), out);
        assertTrue(out.contains("Über-study"), out);
        assertTrue(out.contains("pretrained_from: vit_b_lm"), out);
        assertEquals(config.root(), SetConfig.parse(out).root());
    }

    @Test
    void configOfOtherSchemaIsRefused() {
        assertThrows(IllegalArgumentException.class, () -> SetConfig.parse("schema_version: 2\nname: x\n"));
    }

    @Test
    void newConfigHasWhatPythonNeeds() {
        SetConfig.NewSet spec = new SetConfig.NewSet();
        spec.name = "My set";
        spec.usePatches = true;
        spec.patchWidth = 256;
        spec.patchHeight = 128;
        SetConfig config = SetConfig.create(spec);
        SetConfig back = SetConfig.parse(config.toYaml());
        assertEquals("My set", back.name());
        assertTrue(back.isLocal());
        assertTrue(back.usesPatches());
        assertEquals(spec.seed, back.seed());
        assertEquals(List.of(128, 256), ((Map<?, ?>) back.root().get("spatial_coverage")).get("patch_size"));
    }

    @Test
    void splitIsReproducibleAndComplete() {
        List<Planning.ImageRef> images = new java.util.ArrayList<>();
        for (int i = 0; i < 10; i++)
            images.add(new Planning.ImageRef("omero:image:" + i, (long) i, "img" + i, 100, 100, 1, 1));
        SetConfig.NewSet spec = new SetConfig.NewSet();
        spec.seed = 7;
        spec.trainN = 4;
        spec.validateN = 2;
        spec.testN = 1;
        Map<String, String> plan = Planning.planSplit(images, spec);
        assertEquals(7, plan.size());
        assertEquals(4, plan.values().stream().filter(Row.TRAINING::equals).count());
        assertEquals(2, plan.values().stream().filter(Row.VALIDATION::equals).count());
        assertEquals(1, plan.values().stream().filter(Row.TEST::equals).count());
        assertEquals(Row.TRAINING, plan.values().iterator().next());
        assertEquals(plan, Planning.planSplit(images, spec));
        spec.seed = 8;
        assertNotEquals(plan.keySet().toString(), Planning.planSplit(images, spec).keySet().toString());

        spec.trainN = 20;
        assertThrows(Planning.SelectionException.class, () -> Planning.planSplit(images, spec));

        spec.useAllImages = true;
        spec.validationFraction = 0.2;
        spec.testFraction = 0.1;
        Map<String, String> all = Planning.planSplit(images, spec);
        assertEquals(10, all.size());
        assertEquals(2, all.values().stream().filter(Row.VALIDATION::equals).count());
        assertEquals(1, all.values().stream().filter(Row.TEST::equals).count());
    }

    @Test
    void patchesNeverOverlapAndStayInside() {
        Random rng = new Random(1);
        for (int round = 0; round < 50; round++) {
            List<int[]> origins = Planning.patchOrigins(300, 400, 128, 128, 4, true, rng);
            assertEquals(4, origins.size());
            for (int i = 0; i < origins.size(); i++) {
                int[] a = origins.get(i);
                assertTrue(a[0] >= 0 && a[1] >= 0 && a[0] + 128 <= 400 && a[1] + 128 <= 300);
                for (int j = i + 1; j < origins.size(); j++) {
                    int[] b = origins.get(j);
                    assertFalse(a[0] < b[0] + 128 && b[0] < a[0] + 128 && a[1] < b[1] + 128 && b[1] < a[1] + 128);
                }
            }
        }
        assertEquals(6, Planning.patchOrigins(300, 400, 128, 128, 10, false, rng).size());
        assertEquals(6, Planning.patchOrigins(300, 400, 128, 128, 10, true, rng).size());
    }

    @Test
    void unitsFollowTheSpec() {
        Planning.ImageRef image = new Planning.ImageRef("omero:image:42", 42L, "img", 600, 500, 5, 3);
        SetConfig.NewSet spec = new SetConfig.NewSet();
        spec.channel = 1;
        spec.timepoints = List.of(0, 2, 9);
        spec.zSlices = List.of(4);
        List<Row> whole = Planning.unitsForImage(image, Row.TRAINING, spec);
        assertEquals(2, whole.size());
        assertEquals("e32e7635_0_4", whole.get(0).annotationId);
        assertEquals("e32e7635_2_4", whole.get(1).annotationId);
        assertFalse(whole.get(0).isPatch);
        assertEquals(1, whole.get(0).channel);
        assertEquals(4, whole.get(0).zStart);

        spec.timepoints = List.of(0);
        spec.usePatches = true;
        spec.patchWidth = 200;
        spec.patchHeight = 100;
        spec.patchesPerImage = 3;
        List<Row> patches = Planning.unitsForImage(image, Row.TEST, spec);
        assertEquals(3, patches.size());
        assertEquals("e32e7635_0_4_1", patches.get(1).annotationId);
        assertEquals(200, patches.get(0).patchWidth);
        assertEquals(100, patches.get(0).patchHeight);
        assertEquals(patches.get(0).patchX, Planning.unitsForImage(image, Row.TEST, spec).get(0).patchX);

        spec.patchWidth = 1000; // larger than the image: the whole image is the patch
        List<Row> big = Planning.unitsForImage(image, Row.TEST, spec);
        assertEquals(1, big.size());
        assertEquals(600, big.get(0).patchWidth);
        assertEquals(500, big.get(0).patchHeight);
    }

    @Test
    void atomicWriteReplaces(@TempDir Path dir) throws Exception {
        Path file = dir.resolve("sub").resolve("a.txt");
        Bundle.atomicWrite(file, "one".getBytes());
        Bundle.atomicWrite(file, "two".getBytes());
        assertEquals("two", Files.readString(file));
        try (var list = Files.list(file.getParent())) {
            assertEquals(1, list.count());
        }
    }
}
