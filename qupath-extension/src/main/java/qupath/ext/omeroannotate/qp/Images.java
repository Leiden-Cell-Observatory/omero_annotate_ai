package qupath.ext.omeroannotate.qp;

import qupath.ext.omeroannotate.core.Ids;
import qupath.ext.omeroannotate.core.Planning;
import qupath.ext.omeroannotate.core.Row;
import qupath.ext.omeroannotate.omero.OmeroAccess;
import qupath.lib.gui.commands.ProjectCommands;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.images.servers.ImageServerProvider;
import qupath.lib.projects.Project;
import qupath.lib.projects.ProjectImageEntry;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.net.URI;
import java.nio.file.Path;
import java.util.Collection;

/**
 * Which QuPath project image belongs to which image of a training set.
 * <p>
 * A set names its images by URI: {@code omero:image:<id>} for images on OMERO, the file
 * path for local images. The URI is remembered on the project entry, so an image is
 * found again even after the entry was renamed.
 */
public final class Images {

    /** Metadata key on a project entry: the training-set URI of its image. */
    public static final String ENTRY_URI = "omero-annotate-ai image";

    private Images() {}

    /** The training-set URI of an open image. */
    public static String uriOf(ImageServer<?> server) {
        Long omeroId = OmeroAccess.imageId(server);
        if (omeroId != null)
            return Ids.omeroUri(omeroId);
        Collection<URI> uris = server.getURIs();
        String base = uris.isEmpty() ? server.getPath() : normalize(uris.iterator().next());
        // A file holding several images (series): every image after the first carries its number.
        int series = seriesOf(server.getPath());
        return series > 0 ? base + SERIES + series : base;
    }

    private static final String SERIES = "#series=";
    private static final java.util.regex.Pattern SERIES_ARG = java.util.regex.Pattern.compile("--series,\\s*(\\d+)");

    /** The series number in a QuPath server path such as {@code ...file.lif[--series, 2]}; 0 if none. */
    static int seriesOf(String serverPath) {
        if (serverPath == null)
            return 0;
        java.util.regex.Matcher matcher = SERIES_ARG.matcher(serverPath);
        return matcher.find() ? Integer.parseInt(matcher.group(1)) : 0;
    }

    private static String normalize(URI uri) {
        if ("file".equalsIgnoreCase(uri.getScheme())) {
            try {
                return Path.of(uri).toAbsolutePath().normalize().toString().replace('\\', '/');
            } catch (RuntimeException e) {
                return uri.toString();
            }
        }
        Long omeroId = OmeroAccess.imageId(uri);
        return omeroId != null ? Ids.omeroUri(omeroId) : uri.toString();
    }

    /**
     * What planning needs to know about a project image (its size and planes). The
     * image is opened briefly; its URI is remembered on the entry.
     */
    public static Planning.ImageRef describe(ProjectImageEntry<BufferedImage> entry) throws IOException {
        try (ImageServer<BufferedImage> server = entry.getServerBuilder().build()) {
            String uri = uriOf(server);
            entry.putMetadataValue(ENTRY_URI, uri);
            return new Planning.ImageRef(uri, Ids.parseOmeroUri(uri), entry.getImageName(), server.getWidth(),
                    server.getHeight(), server.nZSlices(), server.nTimepoints());
        } catch (IOException e) {
            throw e;
        } catch (Exception e) {
            throw new IOException(entry.getImageName() + " could not be opened: " + e.getMessage(), e);
        }
    }

    /** The project entry of a unit's image, or null if the project does not have it. */
    public static ProjectImageEntry<BufferedImage> find(Project<BufferedImage> project, Row row) {
        if (project == null)
            return null;
        for (ProjectImageEntry<BufferedImage> entry : project.getImageList())
            if (row.imageUri.equals(entry.getMetadataValue(ENTRY_URI)))
                return entry;
        // Entries the plugin has not seen yet: recognise them by their link or file.
        ProjectImageEntry<BufferedImage> match = null;
        int matches = 0;
        for (ProjectImageEntry<BufferedImage> entry : project.getImageList()) {
            if (entry.getMetadataValue(ENTRY_URI) != null)
                continue;
            try {
                for (URI uri : entry.getURIs()) {
                    String base = row.imageUri.contains(SERIES)
                            ? row.imageUri.substring(0, row.imageUri.indexOf(SERIES)) : row.imageUri;
                    if (base.equals(normalize(uri))) {
                        match = entry;
                        matches++;
                    }
                }
            } catch (IOException | RuntimeException e) {
                // an entry whose location cannot be read is not the one
            }
        }
        if (matches == 1) { // several entries of one file (series) cannot be told apart this way
            match.putMetadataValue(ENTRY_URI, row.imageUri);
            return match;
        }
        return null;
    }

    /**
     * The project entry of a unit's image, adding the image to the project if needed.
     *
     * @param webUri address of the OMERO web server the OMERO extension is logged in to
     *               (needed to add an OMERO image), or null
     */
    public static ProjectImageEntry<BufferedImage> findOrAdd(Project<BufferedImage> project, Row row, String webUri)
            throws IOException {
        ProjectImageEntry<BufferedImage> entry = find(project, row);
        if (entry != null)
            return entry;
        String location;
        String[] args = new String[0];
        Long omeroId = Ids.parseOmeroUri(row.imageUri);
        if (omeroId != null) {
            if (webUri == null)
                throw new IOException("\"" + row.imageName + "\" is an image on OMERO. Log in to OMERO "
                        + "(Extensions ▸ OMERO ▸ Browse server) so QuPath can open it.");
            location = OmeroAccess.imageLink(webUri, omeroId);
        } else {
            int at = row.imageUri.indexOf(SERIES);
            location = at < 0 ? row.imageUri : row.imageUri.substring(0, at);
            if (at >= 0)
                args = new String[]{"--series", row.imageUri.substring(at + SERIES.length())};
            if (!location.contains("://") && !java.nio.file.Files.exists(Path.of(location)))
                throw new IOException("The image file is not on this computer: " + location);
        }
        ImageServer<BufferedImage> server;
        try {
            server = ImageServerProvider.buildServer(location, BufferedImage.class, args);
        } catch (IOException | RuntimeException e) {
            throw new IOException("\"" + row.imageName + "\" could not be opened (" + location + "): "
                    + e.getMessage(), e);
        }
        try {
            // Multi-channel images are fluorescence-like, so QuPath need not ask for each
            // one; for colour images QuPath decides (or asks) as the user has set it up.
            entry = ProjectCommands.addSingleImageToProject(project, server,
                    server.isRGB() ? null : ImageData.ImageType.FLUORESCENCE);
        } finally {
            try {
                server.close();
            } catch (Exception e) {
                // closing a server that was only read for its metadata
            }
        }
        if (entry == null)
            throw new IOException("\"" + row.imageName + "\" could not be added to the QuPath project");
        if (row.imageName != null && !row.imageName.isBlank())
            entry.setImageName(row.imageName);
        entry.putMetadataValue(ENTRY_URI, row.imageUri);
        project.syncChanges();
        return entry;
    }
}
