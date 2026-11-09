package qupath.ext.omeroannotate.core;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;

/**
 * Pixels of one 2D image (optionally with several samples per pixel, e.g. RGB), held as
 * little-endian bytes together with their data type.
 *
 * @param dtype   numpy-style type code without byte order: u1, u2, u4, i1, i2, i4, f4 or f8
 * @param width   width in pixels
 * @param height  height in pixels
 * @param samples samples per pixel (1 for grey values and label images, 3 for RGB)
 * @param bytes   the values, row by row, samples interleaved, little-endian
 */
public record PixelArray(String dtype, int width, int height, int samples, byte[] bytes) {

    public PixelArray {
        int expected = width * height * samples * bytesPerSample(dtype);
        if (bytes.length != expected)
            throw new IllegalArgumentException("Expected " + expected + " bytes for " + width + "x" + height
                    + "x" + samples + " " + dtype + ", got " + bytes.length);
    }

    public static int bytesPerSample(String dtype) {
        return Integer.parseInt(dtype.substring(1));
    }

    public int bytesPerSample() {
        return bytesPerSample(dtype);
    }

    public boolean isFloat() {
        return dtype.startsWith("f");
    }

    public boolean isSigned() {
        return dtype.startsWith("i");
    }

    /**
     * SHA-256 of the pixels: data type, shape and values, not the file. It equals
     * {@code omero_annotate_ai.bundle.pixel_digest} of the same image in Python, so a mask
     * saved here and the same mask uploaded compressed to OMERO have the same digest.
     */
    public String digest() {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            String order = bytesPerSample() == 1 ? "|" : "<";
            String shape = samples == 1
                    ? "(" + height + ", " + width + ")"
                    : "(" + height + ", " + width + ", " + samples + ")";
            md.update((order + dtype + "|" + shape + "|").getBytes(StandardCharsets.UTF_8));
            md.update(bytes);
            return HexFormat.of().formatHex(md.digest());
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }

    /** The values as whole numbers (single-sample integer images, e.g. label masks). */
    public int[] toInts() {
        if (samples != 1)
            throw new IllegalStateException("Not a single-channel image");
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        int n = width * height;
        int[] values = new int[n];
        for (int i = 0; i < n; i++) {
            switch (dtype) {
                case "u1" -> values[i] = buffer.get() & 0xFF;
                case "i1" -> values[i] = buffer.get();
                case "u2" -> values[i] = buffer.getShort() & 0xFFFF;
                case "i2" -> values[i] = buffer.getShort();
                case "u4", "i4" -> values[i] = buffer.getInt();
                case "f4" -> values[i] = Math.round(buffer.getFloat());
                case "f8" -> values[i] = (int) Math.round(buffer.getDouble());
                case "u8", "i8" -> values[i] = (int) buffer.getLong();
                default -> throw new IllegalStateException("Unsupported data type " + dtype);
            }
        }
        return values;
    }

    /** A label image: 16-bit when the labels fit, else 32-bit. */
    public static PixelArray ofLabels(int[] labels, int width, int height) {
        int max = 0;
        for (int v : labels)
            max = Math.max(max, v);
        boolean wide = max > 0xFFFF;
        ByteBuffer buffer = ByteBuffer.allocate(labels.length * (wide ? 4 : 2)).order(ByteOrder.LITTLE_ENDIAN);
        for (int v : labels) {
            if (wide)
                buffer.putInt(v);
            else
                buffer.putShort((short) v);
        }
        return new PixelArray(wide ? "u4" : "u2", width, height, 1, buffer.array());
    }
}
