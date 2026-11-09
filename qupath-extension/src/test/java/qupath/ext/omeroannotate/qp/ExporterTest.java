package qupath.ext.omeroannotate.qp;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import qupath.ext.omeroannotate.core.AnnotationSession;
import qupath.ext.omeroannotate.core.PixelArray;
import qupath.ext.omeroannotate.core.Planning;
import qupath.ext.omeroannotate.core.Row;
import qupath.ext.omeroannotate.core.SetConfig;
import qupath.ext.omeroannotate.core.TiffIO;
import qupath.ext.omeroannotate.core.TrainingSet;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/** Training data export: the folder layouts tools train from. */
class ExporterTest {

    private TrainingSet annotatedSet(Path home) throws Exception {
        SetConfig.NewSet spec = new SetConfig.NewSet();
        spec.name = "export me";
        spec.seed = 5;
        spec.trainN = 2;
        spec.validateN = 1;
        spec.testN = 1;
        List<Planning.ImageRef> images = new ArrayList<>();
        for (int i = 0; i < 5; i++)
            images.add(new Planning.ImageRef("/data/img" + i + ".tif", null, "img" + i + ".tif", 8, 6, 1, 1));
        List<Row> rows = new ArrayList<>();
        Planning.planSplit(images, spec).forEach((uri, category) -> images.stream()
                .filter(i -> i.uri().equals(uri)).forEach(i -> rows.addAll(Planning.unitsForImage(i, category, spec))));
        TrainingSet set = TrainingSet.create(SetConfig.create(spec), rows, null, home, "test");
        AnnotationSession session = new AnnotationSession(set, false, "test", Units.TRACER);
        int n = 0;
        for (Row row = session.next(); row != null; row = session.next()) {
            if (n++ == 0) { // the first training image is excluded
                session.exclude(row);
                continue;
            }
            int[] labels = new int[48];
            labels[n] = 1;
            session.commit(row, PixelArray.ofLabels(labels, 8, 6), new PixelArray("u2", 8, 6, 1, new byte[96]));
        }
        session.finish(null, false);
        return set;
    }

    @Test
    void splitFoldersPairImagesAndLabelsByName(@TempDir Path home, @TempDir Path out) throws Exception {
        TrainingSet set = annotatedSet(home);
        Exporter.Report report = Exporter.export(set, out, Exporter.Layout.SPLIT_FOLDERS, null, null, (a, b) -> { });
        assertTrue(report.skipped().isEmpty(), report.skipped().toString());
        assertEquals(1, report.written().get(Row.TRAINING), "the excluded image is not exported");
        assertEquals(1, report.written().get(Row.VALIDATION));
        assertEquals(1, report.written().get(Row.TEST));
        for (Row row : set.rows()) {
            String prefix = row.category.equals(Row.TRAINING) ? "train" : row.category.equals(Row.VALIDATION) ? "val" : "test";
            Path image = out.resolve(prefix + "_input").resolve(row.annotationId + ".tif");
            Path label = out.resolve(prefix + "_label").resolve(row.annotationId + ".tif");
            assertEquals(row.isAnnotated(), Files.exists(image), row.toString());
            assertEquals(row.isAnnotated(), Files.exists(label), row.toString());
            if (row.isAnnotated()) {
                assertEquals(row.maskDigest, TiffIO.read(label).digest());
                assertEquals(row.inputDigest, TiffIO.read(image).digest());
            }
        }
    }

    @Test
    void cellposeLayoutPutsMasksNextToImages(@TempDir Path home, @TempDir Path out) throws Exception {
        TrainingSet set = annotatedSet(home);
        Exporter.export(set, out, Exporter.Layout.CELLPOSE, null, null, (a, b) -> { });
        try (var train = Files.list(out.resolve("train")); var test = Files.list(out.resolve("test"))) {
            List<String> trainFiles = train.map(p -> p.getFileName().toString()).sorted().toList();
            assertEquals(2, trainFiles.size());
            assertTrue(trainFiles.get(1).equals(trainFiles.get(0).replace(".tif", "_masks.tif")), trainFiles.toString());
            assertEquals(4, test.count(), "validation and test images go to test/");
        }
    }

    @Test
    void aSetWithoutAnnotationsCannotBeExported(@TempDir Path home, @TempDir Path out) throws Exception {
        SetConfig.NewSet spec = new SetConfig.NewSet();
        spec.trainN = 1;
        spec.validateN = 0;
        List<Row> rows = Planning.unitsForImage(new Planning.ImageRef("/a.tif", null, "a.tif", 4, 4, 1, 1), Row.TRAINING, spec);
        TrainingSet set = TrainingSet.create(SetConfig.create(spec), rows, null, home, "test");
        assertThrows(java.io.IOException.class,
                () -> Exporter.export(set, out, Exporter.Layout.SPLIT_FOLDERS, null, null, (a, b) -> { }));
    }
}
