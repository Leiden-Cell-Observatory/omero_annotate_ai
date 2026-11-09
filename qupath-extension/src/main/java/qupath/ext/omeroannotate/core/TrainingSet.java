package qupath.ext.omeroannotate.core;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;

/**
 * A training set: a fixed selection of image units, split into training, validation and
 * test, plus everything annotated for it.
 * <p>
 * The set lives in one folder (config, tracking table, input images, masks). A set with
 * {@code storage: omero} also has a tracking table on OMERO, which is the shared record:
 * masks and ROIs are uploaded there and several computers can annotate the same set. The
 * formats are those of the Python package {@code omero-annotate-ai}.
 */
public final class TrainingSet {

    /** Prefix of the error of a unit whose local and uploaded masks differ. */
    public static final String NEEDS_REVIEW = "needs review: ";

    private final SetConfig config;
    private final List<Row> rows;
    private final Path bundle;
    private final RemoteStore remote;
    private final String softwareVersion;
    /** Units saved in this folder that the OMERO table does not show as done yet. */
    private final Set<String> tablePending = new LinkedHashSet<>();

    /** Turns a label mask into the outlines of its objects, in full-image coordinates. */
    public interface MaskTracer {
        List<RemoteStore.Polygon> trace(PixelArray mask, int offsetX, int offsetY);
    }

    private TrainingSet(SetConfig config, List<Row> rows, Path bundle, RemoteStore remote, String softwareVersion) {
        this.config = config;
        this.rows = rows;
        this.bundle = bundle;
        this.remote = remote;
        this.softwareVersion = softwareVersion;
    }

    // --- Creating and opening -------------------------------------------------------

    /**
     * A new training set from planned rows. For {@code storage: omero} the tracking table
     * is created on OMERO, linked to the containers, and the config published next to it.
     */
    public static TrainingSet create(SetConfig config, List<Row> rows, RemoteStore remote, Path home,
                                     String softwareVersion) throws Exception {
        if (rows.isEmpty())
            throw new IllegalArgumentException("A training set needs at least one image");
        if (config.tableId() != null || config.localUuid() != null)
            throw new IllegalArgumentException("This config already belongs to a training set");
        Path bundle;
        if (config.isLocal()) {
            String id = Bundle.newLocalId();
            config.setIdentity(null, null, id);
            bundle = Bundle.localBundle(home, id, config.name());
        } else {
            if (remote == null)
                throw new IllegalArgumentException("An OMERO training set needs an OMERO connection");
            if (config.containerIds().isEmpty())
                throw new IllegalArgumentException("An OMERO training set needs a dataset or project to attach to");
            for (Row row : rows)
                if (row.imageId == null)
                    throw new IllegalArgumentException(row.imageName + " is not an OMERO image; "
                            + "keep this set on this computer instead");
            long tableId = remote.createTable(config.name(), rows, config.containerType(), config.containerIds());
            config.setIdentity(remote.serverSlug(), tableId, null);
            bundle = Bundle.omeroBundle(home, remote.serverSlug(), tableId, config.name());
        }
        TrainingSet set = new TrainingSet(config, new ArrayList<>(rows), bundle, config.isLocal() ? null : remote,
                softwareVersion);
        set.saveLocalTable();
        set.saveConfig();
        set.publishConfig(null);
        return set;
    }

    /** Open the set kept in {@code folder}. OMERO sets need {@code remote}; local sets ignore it. */
    public static TrainingSet open(Path folder, RemoteStore remote, String softwareVersion) throws Exception {
        Path configFile = folder.resolve(Bundle.CONFIG);
        if (!Files.exists(configFile))
            throw new IOException("No training set in " + folder + " (no config.yaml)");
        SetConfig config = SetConfig.load(configFile);
        return open(config, folder, remote, softwareVersion);
    }

    /** Open the OMERO set tracked in table {@code tableId}; its folder is created under {@code home} if needed. */
    public static TrainingSet open(long tableId, RemoteStore remote, Path home, String softwareVersion)
            throws Exception {
        Path serverDir = home.resolve(remote.serverSlug());
        Path folder = null;
        if (Files.isDirectory(serverDir)) {
            try (Stream<Path> children = Files.list(serverDir)) {
                folder = children.filter(p -> p.getFileName().toString().startsWith(tableId + "_")
                        && Files.exists(p.resolve(Bundle.CONFIG))).sorted().findFirst().orElse(null);
            }
        }
        SetConfig config;
        if (folder != null) {
            config = SetConfig.load(folder.resolve(Bundle.CONFIG));
        } else {
            String yaml = remote.readConfig(tableId);
            if (yaml == null)
                throw new IOException("Training set " + tableId + " has no config on OMERO");
            config = SetConfig.parse(yaml);
            if (config.tableId() == null)
                config.setIdentity(remote.serverSlug(), tableId, null);
            folder = Bundle.omeroBundle(home, remote.serverSlug(), tableId, config.name());
        }
        return open(config, folder, remote, softwareVersion);
    }

    private static TrainingSet open(SetConfig config, Path folder, RemoteStore remote, String softwareVersion)
            throws Exception {
        if (config.isThreeD())
            throw new UnsupportedOperationException("This is a 3D training set; QuPath annotates 2D planes only. "
                    + "Use the napari plugin of omero-annotate-ai for 3D sets.");
        Map<String, Row> local = new LinkedHashMap<>();
        Path csv = folder.resolve(Bundle.TABLE);
        if (Files.exists(csv))
            for (Row row : RowCodec.fromCsv(Files.readString(csv, StandardCharsets.UTF_8)))
                local.put(row.annotationId, row);
        List<Row> rows;
        Set<String> pending = new LinkedHashSet<>();
        if (config.isLocal()) {
            rows = new ArrayList<>(local.values());
            remote = null;
        } else {
            Long tableId = config.tableId();
            if (tableId == null)
                throw new IOException("This OMERO training set's config has no table ID");
            if (remote == null)
                throw new IOException("This training set is tracked on OMERO; connect to OMERO to open it");
            checkSameSet(config, remote, tableId);
            rows = new ArrayList<>();
            for (Map<String, Object> cells : remote.readTable(tableId)) {
                Row row = RowCodec.fromCells(cells);
                Row mine = local.get(row.annotationId);
                if (mine != null) {
                    // Kept locally only: the last error, and digests the table may not hold yet.
                    boolean mineNewer = row.processed && mine.isAnnotated() && mine.labelId != null
                            && (row.labelId == null || mine.labelId > row.labelId);
                    if (!row.processed && (mine.processed || mine.error != null)) {
                        row.copyFrom(mergePending(row, mine));
                        if (mine.processed)
                            pending.add(row.annotationId);
                    } else if (mineNewer) {
                        // A correction (or an accept after an exclusion) made here whose table
                        // update did not get through: OMERO IDs only grow, so ours is the later one.
                        row.copyFrom(mergePending(row, mine));
                        pending.add(row.annotationId);
                    } else {
                        if (row.inputDigest == null)
                            row.inputDigest = mine.inputDigest;
                    }
                }
                rows.add(row);
            }
            if (rows.isEmpty())
                throw new IOException("Tracking table " + tableId + " has no rows");
        }
        TrainingSet set = new TrainingSet(config, rows, folder, remote, softwareVersion);
        set.tablePending.addAll(pending);
        set.releaseOwnClaims();
        set.closeOpenSessions();
        set.saveLocalTable();
        set.saveConfig();
        return set;
    }

    /**
     * Refuse to mix up servers: a folder made for a set on one OMERO server must not be
     * opened against another server that happens to have a table with the same ID (a
     * test copy of the production server, for example). A server reached under another
     * name is fine: the set on it must then carry this set's name and creation time.
     */
    private static void checkSameSet(SetConfig config, RemoteStore remote, long tableId) throws Exception {
        if (config.server() == null || config.server().equals(remote.serverSlug()))
            return;
        String yaml = remote.readConfig(tableId);
        SetConfig theirs = yaml == null ? null : SetConfig.parse(yaml);
        boolean same = theirs != null && config.name().equals(theirs.name())
                && String.valueOf(config.root().get("created")).equals(String.valueOf(theirs.root().get("created")));
        if (!same)
            throw new IOException("This training set belongs to the OMERO server \"" + config.server()
                    + "\", but QuPath is connected to \"" + remote.serverSlug() + "\", where table " + tableId
                    + " is " + (theirs == null ? "not a training set with a config" : "another set (\""
                    + theirs.name() + "\")") + ". Log in to the right server in the OMERO extension.");
    }

    /** A unit saved locally whose table row is still open: keep the local state, with the table's claim. */
    private static Row mergePending(Row table, Row mine) {
        Row merged = mine.copy();
        merged.claimedBy = table.claimedBy;
        merged.claimedAt = table.claimedAt;
        return merged;
    }

    // --- What the set is ------------------------------------------------------------

    public SetConfig config() {
        return config;
    }

    public String name() {
        return config.name();
    }

    public List<Row> rows() {
        return rows;
    }

    public Path bundle() {
        return bundle;
    }

    public boolean isLocal() {
        return remote == null;
    }

    public RemoteStore remote() {
        return remote;
    }

    public Long tableId() {
        return config.tableId();
    }

    public String softwareVersion() {
        return softwareVersion;
    }

    /** The OMERO user, or the login name of this computer for local sets. */
    public String user() {
        return remote != null ? remote.userName() : System.getProperty("user.name");
    }

    public Row row(String annotationId) {
        for (Row row : rows)
            if (row.annotationId.equals(annotationId))
                return row;
        return null;
    }

    /** Per split: units done (annotated or excluded), excluded, and in total. */
    public record Progress(int done, int excluded, int total) {}

    public Map<String, Progress> progress() {
        Map<String, Progress> result = new LinkedHashMap<>();
        for (String category : List.of(Row.TRAINING, Row.VALIDATION, Row.TEST)) {
            int done = 0, excluded = 0, total = 0;
            for (Row row : rows) {
                if (!row.category.equals(category))
                    continue;
                total++;
                if (row.processed)
                    done++;
                if (row.isExcluded())
                    excluded++;
            }
            result.put(category, new Progress(done, excluded, total));
        }
        return result;
    }

    // --- Files ----------------------------------------------------------------------

    public void saveLocalTable() throws IOException {
        Bundle.atomicWrite(bundle.resolve(Bundle.TABLE), RowCodec.toCsv(rows).getBytes(StandardCharsets.UTF_8));
    }

    public void saveConfig() throws IOException {
        config.save(bundle.resolve(Bundle.CONFIG));
    }

    /**
     * Replace the set's config on OMERO with the local one, after merging in what other
     * computers published (their sessions, larger image counts). Local sets: nothing.
     */
    public void publishConfig(Map<String, Object> startingSession) throws Exception {
        Long tableId = config.tableId();
        if (remote == null || tableId == null)
            return;
        mergeRemoteConfig(startingSession);
        saveConfig();
        remote.replaceConfig(tableId, config.toYaml(), config.containerType(), config.primaryContainerId());
    }

    private SetConfig mergeRemoteConfig(Map<String, Object> startingSession) throws Exception {
        String yaml = remote.readConfig(config.tableId());
        if (yaml == null)
            return null;
        SetConfig theirs = SetConfig.parse(yaml);
        List<Integer> used = new ArrayList<>();
        for (Row row : rows)
            used.add(row.session);
        config.mergeRemote(theirs, startingSession, used);
        return theirs;
    }

    // --- Sessions -------------------------------------------------------------------

    /** Record a new session; returns its number. */
    public int beginSession(String kind, String model) throws Exception {
        if (remote != null)
            mergeRemoteConfig(null);
        int max = 0;
        for (Map<String, Object> session : config.sessions())
            max = Math.max(max, SetConfig.sessionId(session));
        for (Row row : rows)
            max = Math.max(max, row.session);
        Map<String, Object> record = config.addSession(max + 1, kind, user(), softwareVersion, model);
        if (remote == null) {
            saveConfig();
            return SetConfig.sessionId(record);
        }
        String host = Ids.hostId();
        for (int attempt = 0; attempt < 5; attempt++) {
            publishConfig(record);
            String yaml = remote.readConfig(config.tableId());
            Object owner = host;
            if (yaml != null)
                for (Map<String, Object> s : SetConfig.parse(yaml).sessions())
                    if (SetConfig.sessionId(s) == SetConfig.sessionId(record))
                        owner = s.get("host");
            if (host.equals(owner))
                break;
        }
        return SetConfig.sessionId(record);
    }

    /** Close a session's record and publish the config. */
    public void endSession(int sessionId, int annotated, String note, String status) throws Exception {
        Map<String, Object> record = config.session(sessionId);
        if (record == null)
            return;
        record.put("ended", SetConfig.now());
        record.put("status", status);
        record.put("n_annotated", annotated);
        if (note != null && !note.isBlank())
            record.put("note", note.strip());
        saveConfig();
        publishConfig(null);
    }

    /** Sessions a crashed run on this computer left open become "interrupted". */
    private void closeOpenSessions() {
        String host = Ids.hostId();
        for (Map<String, Object> record : config.sessions()) {
            Object theirHost = record.get("host");
            if ("open".equals(record.get("status")) && (theirHost == null || host.equals(theirHost))) {
                int id = SetConfig.sessionId(record);
                int n = 0;
                for (Row row : rows)
                    if (row.isAnnotated() && row.session == id)
                        n++;
                record.put("status", "interrupted");
                record.put("n_annotated", n);
            }
        }
    }

    // --- Claims (several computers, one OMERO set) ----------------------------------

    /** {@code <user>/<host>}: whose claims a run on this computer may take over. */
    public String owner() {
        return (user() == null ? "unknown" : user()) + "/" + Ids.hostId();
    }

    /** The claim a session writes: {@code <session>@<user>/<host>}. */
    public String token(int session) {
        return session + "@" + owner();
    }

    static String claimNow() {
        return OffsetDateTime.now(ZoneOffset.UTC).truncatedTo(ChronoUnit.SECONDS)
                .format(DateTimeFormatter.ISO_OFFSET_DATE_TIME).replace("Z", "+00:00");
    }

    /** Whether a claim is held and younger than {@code hours} (an unreadable time counts as expired). */
    public static boolean isLive(String claimedBy, String claimedAt, double hours) {
        if (claimedBy == null || claimedBy.isEmpty() || claimedAt == null || claimedAt.isEmpty())
            return false;
        try {
            OffsetDateTime since = OffsetDateTime.parse(claimedAt.replace(' ', 'T'));
            return Duration.between(since, OffsetDateTime.now(ZoneOffset.UTC)).toMillis() < hours * 3_600_000;
        } catch (RuntimeException e) {
            return false;
        }
    }

    /** Whether a live claim belongs to another user or computer than {@code token}. */
    public static boolean heldByOther(String holder, String claimedAt, String token, double hours) {
        if (holder == null || holder.isEmpty())
            return false;
        String theirOwner = holder.substring(holder.indexOf('@') + 1);
        String myOwner = token.substring(token.indexOf('@') + 1);
        if (holder.indexOf('@') >= 0 && theirOwner.equals(myOwner))
            return false;
        return isLive(holder, claimedAt, hours);
    }

    /**
     * Release the claims an earlier run of this user on this computer left (a crash, a
     * dropped connection), so they do not keep other computers away until they expire.
     */
    private void releaseOwnClaims() {
        if (remote == null)
            return;
        String suffix = "@" + owner();
        List<String> held = new ArrayList<>();
        for (Row row : rows)
            if (row.claimedBy != null && row.claimedBy.endsWith(suffix))
                held.add(row.annotationId);
        if (held.isEmpty())
            return;
        try {
            remote.updateCells(config.tableId(), held, cells -> str(cells.get("claimed_by")).endsWith(suffix)
                    ? Map.of("claimed_by", "", "claimed_at", "") : null);
            for (Row row : rows) {
                if (held.contains(row.annotationId)) {
                    row.claimedBy = "";
                    row.claimedAt = "";
                }
            }
        } catch (Exception e) {
            // not released now: own claims never block this computer, and they expire
        }
    }

    /** Try to claim one unit for a session; false if another computer holds or saved it. */
    public boolean claim(Row row, String token, boolean annotated) throws Exception {
        if (remote == null)
            return true;
        double hours = config.claimHours();
        String stamp = claimNow();
        List<String> changed = remote.updateCells(config.tableId(), List.of(row.annotationId), cells -> {
            if (Boolean.TRUE.equals(cells.get("processed")) != annotated)
                return null;
            if (heldByOther(str(cells.get("claimed_by")), str(cells.get("claimed_at")), token, hours))
                return null;
            return Map.of("claimed_by", token, "claimed_at", stamp);
        });
        if (changed.isEmpty())
            return false;
        row.claimedBy = token;
        row.claimedAt = stamp;
        return true;
    }

    /** Clear the claim {@code token} holds on a unit. */
    public void release(Row row, String token) throws Exception {
        if (remote == null)
            return;
        // Decided on the table's cell, not the local row: a commit clears the local
        // claim before its upload, and a failed upload must still hand the unit back.
        remote.updateCells(config.tableId(), List.of(row.annotationId), cells ->
                token.equals(str(cells.get("claimed_by"))) ? Map.of("claimed_by", "", "claimed_at", "") : null);
        if (token.equals(row.claimedBy) || row.claimedBy.isEmpty()) {
            row.claimedBy = "";
            row.claimedAt = "";
        }
    }

    private static String str(Object value) {
        return value == null ? "" : value.toString();
    }

    /**
     * Take over what other computers saved meanwhile: units they annotated get the
     * table's row, rows they added are appended, claims are updated.
     */
    public void refresh() throws Exception {
        if (remote == null)
            return;
        Map<String, Row> theirs = new LinkedHashMap<>();
        for (Map<String, Object> cells : remote.readTable(config.tableId())) {
            Row row = RowCodec.fromCells(cells);
            theirs.put(row.annotationId, row);
        }
        for (Row row : rows) {
            Row their = theirs.remove(row.annotationId);
            if (their == null)
                continue;
            boolean newer = (their.labelId == null ? 0 : their.labelId) > (row.labelId == null ? 0 : row.labelId);
            if (their.processed && (!row.processed || newer)) {
                if (their.inputDigest == null)
                    their.inputDigest = row.inputDigest;
                row.copyFrom(their);
            } else {
                row.claimedBy = their.claimedBy;
                row.claimedAt = their.claimedAt;
            }
        }
        rows.addAll(theirs.values());
        saveLocalTable();
    }

    // --- Uploads --------------------------------------------------------------------

    /** Tag written on every ROI and label file the set uploads: {@code omero-annotate-ai:<table>:<unit>}. */
    public String tag(String annotationId) {
        return "omero-annotate-ai:" + config.tableId() + ":" + annotationId;
    }

    /** {@code <tag> | Training set: <name> | t=.. | z=.. | c=.. | Patch offset: (x, y)}. */
    public String uploadDescription(Row row) {
        StringBuilder sb = new StringBuilder(tag(row.annotationId));
        sb.append(" | Training set: ").append(config.name());
        sb.append(" | t=").append(row.timepoint);
        sb.append(" | z=").append(row.zSlice);
        sb.append(" | c=").append(row.channel);
        if (row.isPatch)
            sb.append(" | Patch offset: (").append(row.patchX).append(", ").append(row.patchY).append(")");
        return sb.toString();
    }

    /** {@code <set name>_<annotation id>_mask.tif}, with unsafe characters replaced by '_'. */
    public String labelFileName(Row row) {
        String raw = config.name() + "_" + row.annotationId + "_mask";
        StringBuilder sb = new StringBuilder();
        for (char c : raw.toCharArray())
            sb.append(Character.isLetterOrDigit(c) || c == '-' || c == '_' || c == '.' ? c : '_');
        return sb + ".tif";
    }

    /**
     * Upload what a saved unit still lacks on OMERO: the label file, then the ROI. Each
     * ID is written to the local table straight away. Uploads already carrying the unit's
     * tag (a run that stopped before its table update) are adopted instead of repeated.
     *
     * @return false, with the unit left open and the error recorded, if an upload fails
     *         or a tagged upload holds a different mask
     */
    boolean uploadUnit(Row row, PixelArray mask, MaskTracer tracer) throws IOException {
        try {
            if (row.imageId == null)
                throw new IllegalStateException("The image of this unit is not on OMERO");
            if (row.labelId == null || row.roiId == null) {
                RemoteStore.Tagged tagged = remote.findTagged(row.imageId, tag(row.annotationId));
                if (row.labelId == null && !tagged.labelIds().isEmpty()) {
                    long labelId = tagged.labelIds().get(tagged.labelIds().size() - 1);
                    byte[] file = remote.downloadFile(labelId);
                    if (file == null || !TiffIO.decode(file).digest().equals(mask.digest())) {
                        row.processed = false;
                        row.error = NEEDS_REVIEW + "another session uploaded a different mask for this unit (label "
                                + labelId + ")";
                        saveLocalTable();
                        return false;
                    }
                    row.labelId = labelId;
                    saveLocalTable();
                }
                if (row.roiId == null && !tagged.roiIds().isEmpty()) {
                    // ROIs of an attempt that was cut off (its answer never arrived). They are
                    // drawn again from the mask below, so the leftovers go: never two sets.
                    remote.delete("Roi", tagged.roiIds());
                }
            }
            String description = uploadDescription(row);
            if (row.labelId == null) {
                row.labelId = remote.uploadLabel(row.imageId, TiffIO.encode(mask, true), labelFileName(row), description);
                saveLocalTable();
            }
            if (row.roiId == null) {
                List<RemoteStore.Polygon> polygons = tracer.trace(mask, row.isPatch ? row.patchX : 0,
                        row.isPatch ? row.patchY : 0);
                row.roiId = remote.uploadRois(row.imageId, polygons, row.zSlice, row.timepoint, row.channel,
                        config.name() + "_ROIs", description);
                saveLocalTable();
            }
        } catch (Exception e) {
            row.processed = false;
            row.error = e.getClass().getSimpleName() + ": " + e.getMessage();
            saveLocalTable();
            return false;
        }
        row.error = null;
        return true;
    }

    /**
     * The saved mask of an annotated unit: the folder's copy if it matches the recorded
     * digest, else the label file on OMERO (kept in the folder if it has none).
     *
     * @return the mask, or null if there is none anywhere
     */
    public PixelArray savedMask(Row row) throws Exception {
        Path path = Bundle.maskPath(bundle, row.annotationId);
        PixelArray local = null;
        if (Files.exists(path)) {
            local = TiffIO.read(path);
            if (row.maskDigest == null || local.digest().equals(row.maskDigest))
                return local;
        }
        if (remote != null && row.labelId != null) {
            byte[] file = remote.downloadFile(row.labelId);
            if (file != null) {
                PixelArray theirs = TiffIO.decode(file);
                if (local == null)
                    TiffIO.write(path, theirs, false);
                return theirs;
            }
        }
        return local;
    }

    /** Write rows to the OMERO table; a failure is remembered so {@link #repair} can finish it. */
    boolean writeTable(List<Row> changed) {
        try {
            remote.writeRows(config.tableId(), changed);
            for (Row row : changed)
                tablePending.remove(row.annotationId);
            return true;
        } catch (Exception e) {
            for (Row row : changed)
                tablePending.add(row.annotationId);
            return false;
        }
    }

    /** Remove uploads that a newer save of {@code row} replaced (never the ones it uses now). */
    void deleteSuperseded(Long labelId, Long roiId, Row row) {
        for (Object[] old : new Object[][]{{"FileAnnotation", labelId, row.labelId}, {"Roi", roiId, row.roiId}}) {
            if (old[1] == null || old[1].equals(old[2]))
                continue;
            try {
                remote.delete((String) old[0], List.of((Long) old[1]));
            } catch (Exception e) {
                // an old upload that could not be deleted is superseded by the newer one
            }
        }
    }

    /** Remember that a unit's table row must still be written; always returns false. */
    boolean markTablePending(Row row) {
        tablePending.add(row.annotationId);
        return false;
    }

    /** Units saved in this folder that OMERO does not have (completely) yet. */
    public List<Row> unfinished() {
        List<Row> result = new ArrayList<>();
        if (remote == null)
            return result;
        for (Row row : rows) {
            boolean failedUpload = row.error != null && !row.error.startsWith(NEEDS_REVIEW) && !row.processed
                    && Files.exists(Bundle.maskPath(bundle, row.annotationId));
            if (failedUpload || tablePending.contains(row.annotationId))
                result.add(row);
        }
        return result;
    }

    /**
     * Finish what an interrupted run or a dropped connection left: upload the masks saved
     * in this folder that OMERO lacks and bring the tracking table up to date. A unit
     * another computer saved meanwhile is left as they saved it.
     *
     * @return how many units were completed
     */
    public int repair(MaskTracer tracer) throws Exception {
        if (remote == null)
            return 0;
        int repaired = 0;
        for (Row row : unfinished()) {
            List<Map<String, Object>> now = remote.readRows(config.tableId(), List.of(row.annotationId));
            if (!now.isEmpty()) {
                Row theirs = RowCodec.fromCells(now.get(0));
                boolean oursNewer = row.isAnnotated() && row.labelId != null
                        && (theirs.labelId == null || row.labelId > theirs.labelId);
                if (theirs.processed && oursNewer) {
                    // Our correction (or accept after an exclusion) is the later one; only its
                    // table update is missing. Write it, then remove what it replaced.
                    if (writeTable(List.of(row))) {
                        deleteSuperseded(theirs.labelId, theirs.roiId, row);
                        repaired++;
                    }
                    continue;
                }
                if (theirs.processed) { // saved elsewhere meanwhile: theirs stands
                    if (theirs.inputDigest == null)
                        theirs.inputDigest = row.inputDigest;
                    row.copyFrom(theirs);
                    tablePending.remove(row.annotationId);
                    continue;
                }
            }
            if (row.isExcluded()) {
                if (writeTable(List.of(row)))
                    repaired++;
                continue;
            }
            Path maskFile = Bundle.maskPath(bundle, row.annotationId);
            if (!Files.exists(maskFile)) {
                tablePending.remove(row.annotationId);
                row.processed = false;
                continue;
            }
            PixelArray mask = TiffIO.read(maskFile);
            if (!row.processed)
                row.markProcessed(config.annotationType());
            row.claimedBy = "";
            row.claimedAt = "";
            if (uploadUnit(row, mask, tracer) && writeTable(List.of(row)))
                repaired++;
        }
        saveLocalTable();
        return repaired;
    }
}
