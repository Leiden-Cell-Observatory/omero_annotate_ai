package qupath.ext.omeroannotate.qp;

import qupath.ext.omeroannotate.core.Bundle;
import qupath.ext.omeroannotate.core.PixelArray;
import qupath.ext.omeroannotate.core.Row;
import qupath.ext.omeroannotate.core.TiffIO;
import qupath.ext.omeroannotate.core.TrainingSet;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.projects.Project;
import qupath.lib.projects.ProjectImageEntry;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiConsumer;

/**
 * Writes a training set as folders of image and label pairs, ready for training.
 * Excluded and unannotated images are left out.
 */
public final class Exporter {

    /** How the files are laid out. */
    public enum Layout {
        /**
         * {@code train_input/ID.tif} with {@code train_label/ID.tif}, and the same for
         * {@code val_} and {@code test_}: the layout of the Python package, read by
         * micro-SAM, BiaPy and DL4MicEverywhere.
         */
        SPLIT_FOLDERS("Image and label folders (micro-SAM, BiaPy, DL4MicEverywhere)"),
        /** {@code train/ID.tif} with {@code train/ID_masks.tif}, and {@code test/}: what Cellpose trains from. */
        CELLPOSE("Cellpose (image and _masks side by side)");

        private final String label;

        Layout(String label) {
            this.label = label;
        }

        @Override
        public String toString() {
            return label;
        }
    }

    /** What an export wrote. */
    public record Report(Map<String, Integer> written, List<String> skipped, Path folder) {}

    private Exporter() {}

    /**
     * Export every annotated unit of a set.
     *
     * @param project  the open QuPath project, used to read images whose pixels are not
     *                 in the set's folder yet (may be null)
     * @param webUri   address of the OMERO web server, to open OMERO images not in the project (may be null)
     * @param progress called with (units done, units in total)
     */
    public static Report export(TrainingSet set, Path folder, Layout layout, Project<BufferedImage> project,
                                String webUri, BiConsumer<Integer, Integer> progress) throws Exception {
        List<Row> rows = new ArrayList<>();
        for (Row row : set.rows())
            if (row.isAnnotated())
                rows.add(row);
        if (rows.isEmpty())
            throw new IOException("This training set has no annotated images yet.");
        Map<String, Integer> written = new LinkedHashMap<>();
        List<String> skipped = new ArrayList<>();
        int done = 0;
        for (Row row : rows) {
            progress.accept(done++, rows.size());
            PixelArray mask;
            PixelArray input;
            try {
                mask = set.savedMask(row);
                if (mask == null)
                    throw new IOException("its mask was not found");
                Path inputFile = Bundle.inputPath(set.bundle(), row.annotationId);
                if (Files.exists(inputFile)) {
                    input = TiffIO.read(inputFile);
                } else {
                    ProjectImageEntry<BufferedImage> entry = Images.findOrAdd(requireProject(project), row, webUri);
                    try (ImageServer<BufferedImage> server = entry.getServerBuilder().build()) {
                        input = Units.readInput(server, row);
                    }
                    TiffIO.write(inputFile, input, false);
                }
                if (input.width() != mask.width() || input.height() != mask.height())
                    throw new IOException("its image (" + input.width() + " x " + input.height() + ") and mask ("
                            + mask.width() + " x " + mask.height() + ") differ in size");
            } catch (Exception e) {
                skipped.add(row.imageName + " (" + row.annotationId + "): " + e.getMessage());
                continue;
            }
            String id = row.annotationId;
            switch (layout) {
                case SPLIT_FOLDERS -> {
                    String prefix = switch (row.category) {
                        case Row.VALIDATION -> "val";
                        case Row.TEST -> "test";
                        default -> "train";
                    };
                    TiffIO.write(folder.resolve(prefix + "_input").resolve(id + ".tif"), input, false);
                    TiffIO.write(folder.resolve(prefix + "_label").resolve(id + ".tif"), mask, false);
                }
                case CELLPOSE -> {
                    // Cellpose trains on train/ and evaluates on test/; validation images go to test/.
                    String sub = row.category.equals(Row.TRAINING) ? "train" : "test";
                    TiffIO.write(folder.resolve(sub).resolve(id + ".tif"), input, false);
                    TiffIO.write(folder.resolve(sub).resolve(id + "_masks.tif"), mask, false);
                }
            }
            written.merge(row.category, 1, Integer::sum);
        }
        progress.accept(rows.size(), rows.size());
        return new Report(written, skipped, folder);
    }

    private static Project<BufferedImage> requireProject(Project<BufferedImage> project) throws IOException {
        if (project == null)
            throw new IOException("its image is not in this folder; open the set's QuPath project to read it");
        return project;
    }
}
