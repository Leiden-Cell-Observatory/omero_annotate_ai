package qupath.ext.omeroannotate.core;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.zip.DataFormatException;
import java.util.zip.Deflater;
import java.util.zip.Inflater;

/**
 * A small reader and writer for the plain TIFF files a training set holds: one 2D image,
 * grey or RGB, 8/16/32-bit integers or floats, stored in strips, uncompressed or
 * zlib-compressed. These are the files Python's {@code tifffile} writes and reads for
 * masks and input images; other TIFF flavours (tiles, LZW, BigTIFF) are refused with a
 * clear message.
 */
public final class TiffIO {

    private TiffIO() {}

    // --- Writing --------------------------------------------------------------------

    /** The bytes of a TIFF file holding {@code image}; {@code compress} uses zlib (lossless). */
    public static byte[] encode(PixelArray image, boolean compress) {
        byte[] data = compress ? deflate(image.bytes()) : image.bytes();
        int samples = image.samples();
        int bits = image.bytesPerSample() * 8;
        int sampleFormat = image.isFloat() ? 3 : image.isSigned() ? 2 : 1;
        int nTags = 11;
        int dataOffset = 8;
        int ifdOffset = pad(dataOffset + data.length);
        int extraOffset = ifdOffset + 2 + nTags * 12 + 4;
        int total = extraOffset + (samples > 1 ? samples * 4 : 0);

        ByteBuffer b = ByteBuffer.allocate(total).order(ByteOrder.LITTLE_ENDIAN);
        b.put((byte) 'I').put((byte) 'I').putShort((short) 42).putInt(ifdOffset);
        b.put(data);
        b.position(ifdOffset);
        b.putShort((short) nTags);
        tag(b, 256, 4, 1, image.width());
        tag(b, 257, 4, 1, image.height());
        if (samples == 1)
            tag(b, 258, 3, 1, bits);
        else
            tag(b, 258, 3, samples, extraOffset);
        tag(b, 259, 3, 1, compress ? 8 : 1);
        tag(b, 262, 3, 1, samples >= 3 ? 2 : 1);
        tag(b, 273, 4, 1, dataOffset);
        tag(b, 277, 3, 1, samples);
        tag(b, 278, 4, 1, image.height());
        tag(b, 279, 4, 1, data.length);
        tag(b, 284, 3, 1, 1);
        if (samples == 1)
            tag(b, 339, 3, 1, sampleFormat);
        else
            tag(b, 339, 3, samples, extraOffset + samples * 2);
        b.putInt(0);
        if (samples > 1) {
            for (int i = 0; i < samples; i++)
                b.putShort((short) bits);
            for (int i = 0; i < samples; i++)
                b.putShort((short) sampleFormat);
        }
        return b.array();
    }

    /** Write {@code image} to {@code path} atomically. */
    public static void write(Path path, PixelArray image, boolean compress) throws IOException {
        Bundle.atomicWrite(path, encode(image, compress));
    }

    private static int pad(int offset) {
        return (offset + 1) & ~1;
    }

    private static void tag(ByteBuffer b, int id, int type, int count, int value) {
        b.putShort((short) id).putShort((short) type).putInt(count);
        if (type == 3 && count == 1)
            b.putShort((short) value).putShort((short) 0);
        else
            b.putInt(value);
    }

    private static byte[] deflate(byte[] input) {
        Deflater deflater = new Deflater(6);
        deflater.setInput(input);
        deflater.finish();
        ByteArrayOutputStream out = new ByteArrayOutputStream(Math.max(64, input.length / 4));
        byte[] buffer = new byte[1 << 16];
        while (!deflater.finished())
            out.write(buffer, 0, deflater.deflate(buffer));
        deflater.end();
        return out.toByteArray();
    }

    // --- Reading --------------------------------------------------------------------

    public static PixelArray read(Path path) throws IOException {
        try {
            return decode(Files.readAllBytes(path));
        } catch (IOException e) {
            throw new IOException(path.getFileName() + ": " + e.getMessage(), e);
        }
    }

    /** The image in a TIFF file's first page. */
    public static PixelArray decode(byte[] file) throws IOException {
        if (file.length < 8)
            throw new IOException("not a TIFF file");
        ByteOrder order;
        if (file[0] == 'I' && file[1] == 'I')
            order = ByteOrder.LITTLE_ENDIAN;
        else if (file[0] == 'M' && file[1] == 'M')
            order = ByteOrder.BIG_ENDIAN;
        else
            throw new IOException("not a TIFF file");
        ByteBuffer b = ByteBuffer.wrap(file).order(order);
        int magic = b.getShort(2) & 0xFFFF;
        if (magic == 43)
            throw new IOException("BigTIFF files are not supported");
        if (magic != 42)
            throw new IOException("not a TIFF file");

        int ifd = b.getInt(4);
        int nTags = b.getShort(ifd) & 0xFFFF;
        int width = 0, height = 0, samples = 1, compression = 1, sampleFormat = 1, predictor = 1, planar = 1;
        int rowsPerStrip = Integer.MAX_VALUE;
        long[] bits = {1};
        long[] offsets = null, counts = null;
        for (int i = 0; i < nTags; i++) {
            int entry = ifd + 2 + i * 12;
            int id = b.getShort(entry) & 0xFFFF;
            switch (id) {
                case 256 -> width = (int) values(b, entry)[0];
                case 257 -> height = (int) values(b, entry)[0];
                case 258 -> bits = values(b, entry);
                case 259 -> compression = (int) values(b, entry)[0];
                case 273 -> offsets = values(b, entry);
                case 277 -> samples = (int) values(b, entry)[0];
                case 278 -> rowsPerStrip = (int) Math.min(Integer.MAX_VALUE, values(b, entry)[0]);
                case 279 -> counts = values(b, entry);
                case 284 -> planar = (int) values(b, entry)[0];
                case 317 -> predictor = (int) values(b, entry)[0];
                case 339 -> sampleFormat = (int) values(b, entry)[0];
                case 322, 323, 324, 325 -> throw new IOException("tiled TIFF files are not supported");
                default -> { }
            }
        }
        if (width <= 0 || height <= 0 || offsets == null || counts == null)
            throw new IOException("incomplete TIFF file");
        if (samples > 1 && planar != 1)
            throw new IOException("planar multi-channel TIFF files are not supported");
        int bytesPer = (int) bits[0] / 8;
        if (bits[0] % 8 != 0 || bytesPer == 0)
            throw new IOException(bits[0] + "-bit TIFF files are not supported");
        for (long bit : bits)
            if (bit != bits[0])
                throw new IOException("TIFF files with mixed bit depths are not supported");
        String kind = switch (sampleFormat) {
            case 2 -> "i";
            case 3 -> "f";
            default -> "u";
        };
        String dtype = kind + bytesPer;

        int rowBytes = width * samples * bytesPer;
        byte[] pixels = new byte[rowBytes * height];
        int position = 0;
        for (int s = 0; s < offsets.length; s++) {
            int rows = Math.min(rowsPerStrip, height - (int) Math.min(height, (long) s * rowsPerStrip));
            int expected = rows * rowBytes;
            if (expected <= 0)
                break;
            int offset = (int) offsets[s], count = (int) counts[s];
            if (offset < 0 || count < 0 || offset + count > file.length)
                throw new IOException("truncated TIFF file");
            switch (compression) {
                case 1 -> System.arraycopy(file, offset, pixels, position, Math.min(count, expected));
                case 8, 32946 -> inflate(file, offset, count, pixels, position, expected);
                default -> throw new IOException("TIFF compression " + compression
                        + " is not supported (use no compression or zlib/deflate)");
            }
            position += expected;
        }
        if (predictor == 2)
            undoHorizontalDifferencing(pixels, width, height, samples, bytesPer, order);
        else if (predictor != 1)
            throw new IOException("TIFF predictor " + predictor + " is not supported");
        if (order == ByteOrder.BIG_ENDIAN && bytesPer > 1)
            swapBytes(pixels, bytesPer);
        return new PixelArray(dtype, width, height, samples, pixels);
    }

    private static long[] values(ByteBuffer b, int entry) throws IOException {
        int type = b.getShort(entry + 2) & 0xFFFF;
        int count = b.getInt(entry + 4);
        int size = switch (type) {
            case 1, 2, 6, 7 -> 1;
            case 3, 8 -> 2;
            case 4, 9 -> 4;
            default -> throw new IOException("unsupported TIFF field type " + type);
        };
        int position = (long) size * count <= 4 ? entry + 8 : b.getInt(entry + 8);
        long[] result = new long[count];
        for (int i = 0; i < count; i++) {
            result[i] = switch (size) {
                case 1 -> b.get(position + i) & 0xFF;
                case 2 -> b.getShort(position + 2 * i) & 0xFFFF;
                default -> b.getInt(position + 4 * i) & 0xFFFFFFFFL;
            };
        }
        return result;
    }

    private static void inflate(byte[] file, int offset, int count, byte[] out, int position, int expected)
            throws IOException {
        Inflater inflater = new Inflater();
        try {
            inflater.setInput(file, offset, count);
            int done = 0;
            while (done < expected && !inflater.finished()) {
                int n = inflater.inflate(out, position + done, expected - done);
                if (n == 0 && (inflater.needsInput() || inflater.needsDictionary()))
                    break;
                done += n;
            }
        } catch (DataFormatException e) {
            throw new IOException("corrupt compressed TIFF data", e);
        } finally {
            inflater.end();
        }
    }

    private static void undoHorizontalDifferencing(byte[] pixels, int width, int height, int samples,
                                                   int bytesPer, ByteOrder order) {
        ByteBuffer b = ByteBuffer.wrap(pixels).order(order);
        int rowValues = width * samples;
        for (int y = 0; y < height; y++) {
            int row = y * rowValues * bytesPer;
            for (int i = samples; i < rowValues; i++) {
                int at = row + i * bytesPer, before = row + (i - samples) * bytesPer;
                switch (bytesPer) {
                    case 1 -> b.put(at, (byte) (b.get(at) + b.get(before)));
                    case 2 -> b.putShort(at, (short) (b.getShort(at) + b.getShort(before)));
                    case 4 -> b.putInt(at, b.getInt(at) + b.getInt(before));
                    default -> { }
                }
            }
        }
    }

    private static void swapBytes(byte[] pixels, int bytesPer) {
        for (int i = 0; i + bytesPer <= pixels.length; i += bytesPer) {
            for (int a = i, z = i + bytesPer - 1; a < z; a++, z--) {
                byte t = pixels[a];
                pixels[a] = pixels[z];
                pixels[z] = t;
            }
        }
    }
}
