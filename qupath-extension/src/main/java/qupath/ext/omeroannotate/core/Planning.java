package qupath.ext.omeroannotate.core;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Which images go to which split, and which units (planes or patches) each image gives.
 * Everything is derived from the set's seed, so a plan can be reproduced.
 */
public final class Planning {

    private Planning() {}

    /** An image as planning sees it. */
    public record ImageRef(String uri, Long imageId, String name, int width, int height, int sizeZ, int sizeT) {}

    /** Raised when a set asks for more images than there are. */
    public static final class SelectionException extends RuntimeException {
        public SelectionException(String message) {
            super(message);
        }
    }

    /** A random stream for one purpose (and e.g. one image), reproducible from the seed. */
    public static Random rngFor(long seed, String purpose, String part) {
        try {
            byte[] hash = MessageDigest.getInstance("SHA-256")
                    .digest((seed + ":" + purpose + ":" + part).getBytes(StandardCharsets.UTF_8));
            return new Random(ByteBuffer.wrap(hash).getLong());
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }

    /**
     * Assign images to the training, validation and test splits at random.
     *
     * @return image URI to split, training images first; images not picked are left out
     * @throws SelectionException if more images are requested than available
     */
    public static Map<String, String> planSplit(List<ImageRef> images, SetConfig.NewSet spec) {
        List<String> uris = new ArrayList<>();
        for (ImageRef image : images)
            if (!uris.contains(image.uri()))
                uris.add(image.uri());
        Collections.shuffle(uris, rngFor(spec.seed, "split", ""));
        Map<String, String> picked = new LinkedHashMap<>();
        int n = uris.size();
        int nVal, nTest, nTrain;
        if (spec.useAllImages) {
            nVal = (int) Math.min(n, Math.floor(n * spec.validationFraction + 0.5));
            nTest = (int) Math.min(n - nVal, Math.floor(n * spec.testFraction + 0.5));
            nTrain = n - nVal - nTest;
        } else {
            nVal = spec.validateN;
            nTest = spec.testN;
            nTrain = spec.trainN;
            int requested = nTrain + nVal + nTest;
            if (requested > n)
                throw new SelectionException("Not enough images: you asked for " + nTrain + " training + "
                        + nVal + " validation + " + nTest + " test = " + requested + ", but only " + n
                        + (n == 1 ? " image is" : " images are") + " available.");
            if (requested == 0)
                throw new SelectionException("Choose at least one image to annotate.");
        }
        int i = 0;
        List<String> test = uris.subList(i, i += nTest);
        List<String> validation = uris.subList(i, i += nVal);
        List<String> training = uris.subList(i, i + nTrain);
        for (String uri : training)
            picked.put(uri, Row.TRAINING);
        for (String uri : validation)
            picked.put(uri, Row.VALIDATION);
        for (String uri : test)
            picked.put(uri, Row.TEST);
        return picked;
    }

    /** Every unit (plane or patch) one image gives; the same for the same seed. */
    public static List<Row> unitsForImage(ImageRef image, String category, SetConfig.NewSet spec) {
        List<Row> rows = new ArrayList<>();
        Random patchRng = rngFor(spec.seed, "patches", image.uri());
        for (int t : spec.timepoints) {
            if (t >= Math.max(1, image.sizeT()))
                continue;
            for (int z : spec.zSlices) {
                if (z >= Math.max(1, image.sizeZ()))
                    continue;
                if (!spec.usePatches) {
                    rows.add(row(image, category, spec.channel, t, z, -1, null));
                    continue;
                }
                int ph = Math.min(spec.patchHeight, image.height());
                int pw = Math.min(spec.patchWidth, image.width());
                boolean whole = image.width() < spec.patchWidth || image.height() < spec.patchHeight;
                if (whole) { // image smaller than a patch: the whole image is the patch
                    ph = image.height();
                    pw = image.width();
                }
                List<int[]> origins = whole ? List.of(new int[]{0, 0})
                        : patchOrigins(image.height(), image.width(), ph, pw, spec.patchesPerImage,
                        spec.randomPatches, patchRng);
                for (int i = 0; i < origins.size(); i++)
                    rows.add(row(image, category, spec.channel, t, z, i,
                            new int[]{origins.get(i)[0], origins.get(i)[1], pw, ph}));
            }
        }
        return rows;
    }

    private static Row row(ImageRef image, String category, int channel, int t, int z, int patchIndex, int[] patch) {
        Row row = new Row();
        row.imageUri = image.uri();
        row.imageId = image.imageId();
        row.imageName = image.name();
        row.annotationId = Ids.annotationId(image.uri(), t, z, patchIndex);
        row.category = category;
        row.timepoint = t;
        row.zSlice = z;
        row.zStart = z;
        row.zEnd = z;
        row.zLength = 1;
        row.channel = channel;
        if (patch != null) {
            row.isPatch = true;
            row.patchX = patch[0];
            row.patchY = patch[1];
            row.patchWidth = patch[2];
            row.patchHeight = patch[3];
        }
        return row;
    }

    /**
     * Top-left corners {@code (x, y)} of up to {@code n} patches that do not overlap:
     * drawn at random, or the first slots of a regular grid.
     */
    public static List<int[]> patchOrigins(int height, int width, int patchH, int patchW, int n,
                                           boolean random, Random rng) {
        List<int[]> grid = new ArrayList<>();
        for (int y = 0; y + patchH <= height; y += patchH)
            for (int x = 0; x + patchW <= width; x += patchW)
                grid.add(new int[]{x, y});
        if (!random)
            return new ArrayList<>(grid.subList(0, Math.min(n, grid.size())));
        int maxX = width - patchW, maxY = height - patchH;
        List<int[]> picked = new ArrayList<>();
        for (int attempt = 0; attempt < n * 50 && picked.size() < n; attempt++) {
            int x = rng.nextInt(maxX + 1), y = rng.nextInt(maxY + 1);
            boolean overlaps = false;
            for (int[] other : picked)
                if (x < other[0] + patchW && other[0] < x + patchW && y < other[1] + patchH && other[1] < y + patchH)
                    overlaps = true;
            if (!overlaps)
                picked.add(new int[]{x, y});
        }
        if (picked.size() < Math.min(n, grid.size())) {
            // A tight image: a random pick of grid slots always fits as many as the grid holds.
            Collections.shuffle(grid, rng);
            picked = new ArrayList<>(grid.subList(0, Math.min(n, grid.size())));
        }
        return picked;
    }
}
