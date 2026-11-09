package qupath.ext.omeroannotate.core;

import org.yaml.snakeyaml.DumperOptions;
import org.yaml.snakeyaml.LoaderOptions;
import org.yaml.snakeyaml.Yaml;
import org.yaml.snakeyaml.constructor.AbstractConstruct;
import org.yaml.snakeyaml.constructor.SafeConstructor;
import org.yaml.snakeyaml.nodes.Node;
import org.yaml.snakeyaml.nodes.ScalarNode;
import org.yaml.snakeyaml.nodes.Tag;
import org.yaml.snakeyaml.representer.Representer;
import org.yaml.snakeyaml.resolver.Resolver;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Pattern;

/**
 * A training set's definition and history: the {@code config.yaml} of the Python package
 * (schema version 3), kept as the YAML tree it is.
 * <p>
 * Keys this class does not know are left untouched, so a config written by Python keeps
 * everything when QuPath saves it again; dates stay the text they were. Only the parts a
 * running set changes are written here: {@code identity}, {@code sessions} and the seed.
 */
public final class SetConfig {

    public static final int SCHEMA_VERSION = 3;

    private final Map<String, Object> root;

    private SetConfig(Map<String, Object> root) {
        this.root = root;
    }

    // --- Reading and writing --------------------------------------------------------

    public static SetConfig parse(String yaml) {
        Object loaded = yaml().load(yaml);
        if (!(loaded instanceof Map<?, ?> map))
            throw new IllegalArgumentException("Not a training set config (no YAML mapping)");
        @SuppressWarnings("unchecked")
        SetConfig config = new SetConfig((Map<String, Object>) map);
        Object version = config.root.getOrDefault("schema_version", SCHEMA_VERSION);
        if (!(version instanceof Number n) || n.intValue() != SCHEMA_VERSION)
            throw new IllegalArgumentException("Config schema_version " + version
                    + " is not supported; this version reads schema_version " + SCHEMA_VERSION);
        return config;
    }

    public static SetConfig load(Path file) throws IOException {
        return parse(Files.readString(file, StandardCharsets.UTF_8));
    }

    public String toYaml() {
        return yaml().dump(root);
    }

    public void save(Path file) throws IOException {
        Bundle.atomicWrite(file, toYaml().getBytes(StandardCharsets.UTF_8));
    }

    public SetConfig copy() {
        return parse(toYaml());
    }

    /** The YAML tree itself (live). */
    public Map<String, Object> root() {
        return root;
    }

    private static Yaml yaml() {
        DumperOptions dump = new DumperOptions();
        dump.setDefaultFlowStyle(DumperOptions.FlowStyle.BLOCK);
        dump.setIndent(2);
        dump.setAllowUnicode(true);
        dump.setWidth(4096);
        dump.setSplitLines(false);
        LoaderOptions load = new LoaderOptions();
        return new Yaml(new PyConstructor(load), new PyRepresenter(dump), dump, load, new PyResolver());
    }

    /**
     * A date or time exactly as it is written in the file. Python reads it as a datetime;
     * here it is only carried along, so it comes out the way it went in.
     */
    public record Stamp(String text) {
        @Override
        public String toString() {
            return text;
        }
    }

    private static final Tag VALUE_TAG = new Tag(Tag.PREFIX + "value");

    /**
     * Decides which plain scalars are numbers, dates and so on, the way PyYAML does. The
     * two libraries must agree, or a text one of them writes without quotes (an ID such
     * as {@code 1234e567}, a note such as {@code 2026-11-09}) is read by the other as a
     * number or a date.
     */
    private static final class PyResolver extends Resolver {
        private static final Pattern PY_FLOAT = Pattern.compile(
                "^(?:[-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+][0-9]+)?"
                        + "|\\.[0-9][0-9_]*(?:[eE][-+][0-9]+)?"
                        + "|[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*"
                        + "|[-+]?\\.(?:inf|Inf|INF)"
                        + "|\\.(?:nan|NaN|NAN))$");

        @Override
        protected void addImplicitResolvers() {
            addImplicitResolver(Tag.BOOL, BOOL, "yYnNtTfFoO", 10);
            addImplicitResolver(Tag.INT, INT, "-+0123456789");
            addImplicitResolver(Tag.FLOAT, PY_FLOAT, "-+0123456789.");
            addImplicitResolver(Tag.MERGE, MERGE, "<", 10);
            addImplicitResolver(Tag.NULL, NULL, "~nN\0", 10);
            addImplicitResolver(Tag.NULL, EMPTY, null, 10);
            addImplicitResolver(Tag.TIMESTAMP, TIMESTAMP, "0123456789", 50);
            addImplicitResolver(VALUE_TAG, VALUE, "=", 10);
        }
    }

    /** Dates stay text ({@link Stamp}); a lone "=" is ordinary text. */
    private static final class PyConstructor extends SafeConstructor {
        PyConstructor(LoaderOptions options) {
            super(options);
            this.yamlConstructors.put(Tag.TIMESTAMP, new AbstractConstruct() {
                @Override
                public Object construct(Node node) {
                    return new Stamp(((ScalarNode) node).getValue());
                }
            });
            this.yamlConstructors.put(VALUE_TAG, new AbstractConstruct() {
                @Override
                public Object construct(Node node) {
                    return ((ScalarNode) node).getValue();
                }
            });
        }
    }

    /** Writes dates back unquoted, and numbers in a form PyYAML reads as numbers. */
    private static final class PyRepresenter extends Representer {
        PyRepresenter(DumperOptions options) {
            super(options);
            this.representers.put(Stamp.class, data ->
                    representScalar(Tag.TIMESTAMP, ((Stamp) data).text(), DumperOptions.ScalarStyle.PLAIN));
            this.representers.put(Double.class, data -> representScalar(Tag.FLOAT, pythonFloat((Double) data)));
            this.representers.put(Float.class, data ->
                    representScalar(Tag.FLOAT, pythonFloat(((Float) data).doubleValue())));
        }
    }

    /** A number as PyYAML reads a float: always with a dot, never Java's {@code 1.0E10}. */
    static String pythonFloat(double value) {
        if (Double.isNaN(value))
            return ".nan";
        if (Double.isInfinite(value))
            return value > 0 ? ".inf" : "-.inf";
        String text = new java.math.BigDecimal(Double.toString(value)).toPlainString();
        return text.contains(".") ? text : text + ".0";
    }

    // --- Tree access ----------------------------------------------------------------

    @SuppressWarnings("unchecked")
    private Map<String, Object> section(String key) {
        Object value = root.get(key);
        if (!(value instanceof Map)) {
            value = new LinkedHashMap<String, Object>();
            root.put(key, value);
        }
        return (Map<String, Object>) value;
    }

    private static int intOf(Object value, int fallback) {
        return value instanceof Number n ? n.intValue() : fallback;
    }

    private static double doubleOf(Object value, double fallback) {
        return value instanceof Number n ? n.doubleValue() : fallback;
    }

    private static boolean boolOf(Object value, boolean fallback) {
        return value instanceof Boolean b ? b : fallback;
    }

    private static List<Integer> intList(Object value, List<Integer> fallback) {
        if (!(value instanceof List<?> list))
            return fallback;
        List<Integer> result = new ArrayList<>();
        for (Object item : list)
            if (item instanceof Number n)
                result.add(n.intValue());
        return result;
    }

    // --- What the set is ------------------------------------------------------------

    public String name() {
        return Objects.toString(root.get("name"), "");
    }

    /** {@code "omero"} (tracking table and uploads) or {@code "local"} (the folder only). */
    public String storage() {
        return Objects.toString(root.getOrDefault("storage", "omero"), "omero");
    }

    public boolean isLocal() {
        return "local".equals(storage());
    }

    public String source() {
        return Objects.toString(root.getOrDefault("source", "omero"), "omero");
    }

    public Long tableId() {
        return section("identity").get("table_id") instanceof Number n ? n.longValue() : null;
    }

    public String localUuid() {
        Object value = section("identity").get("local_uuid");
        return value == null ? null : value.toString();
    }

    public String server() {
        Object value = section("identity").get("server");
        return value == null ? null : value.toString();
    }

    public void setIdentity(String server, Long tableId, String localUuid) {
        Map<String, Object> identity = section("identity");
        identity.put("server", server);
        identity.put("table_id", tableId);
        identity.put("local_uuid", localUuid);
    }

    public String containerType() {
        return Objects.toString(section("omero").getOrDefault("container_type", "dataset"), "dataset");
    }

    /** The container the set's config is attached to (the first of several). */
    public long primaryContainerId() {
        List<Long> all = containerIds();
        return all.isEmpty() ? 0 : all.get(0);
    }

    public List<Long> containerIds() {
        Map<String, Object> omero = section("omero");
        List<Long> ids = new ArrayList<>();
        if (omero.get("container_ids") instanceof List<?> list && !list.isEmpty()) {
            for (Object item : list)
                if (item instanceof Number n)
                    ids.add(n.longValue());
            return ids;
        }
        if (omero.get("container_id") instanceof Number n && n.longValue() != 0)
            ids.add(n.longValue());
        return ids;
    }

    /** The channel that is annotated. */
    public int labelChannel() {
        Map<String, Object> sc = section("spatial_coverage");
        if (sc.get("label_channel") instanceof Number n)
            return n.intValue();
        List<Integer> channels = intList(sc.get("channels"), List.of(0));
        return channels.isEmpty() ? 0 : channels.get(0);
    }

    public boolean isThreeD() {
        return boolOf(section("spatial_coverage").get("three_d"), false);
    }

    public boolean usesPatches() {
        return boolOf(section("spatial_coverage").get("use_patches"), false);
    }

    public String annotationType() {
        return Objects.toString(section("annotation_methodology").getOrDefault("annotation_type",
                "segmentation_mask"), "segmentation_mask");
    }

    /** Hours a claim on a unit lasts before another computer may take the unit over. */
    public double claimHours() {
        return doubleOf(section("workflow").get("claim_hours"), 6.0);
    }

    public Long seed() {
        return section("training").get("seed") instanceof Number n ? n.longValue() : null;
    }

    // --- Sessions -------------------------------------------------------------------

    /** The session records, oldest first (live maps). */
    @SuppressWarnings("unchecked")
    public List<Map<String, Object>> sessions() {
        Object value = root.get("sessions");
        if (!(value instanceof List)) {
            value = new ArrayList<Map<String, Object>>();
            root.put("sessions", value);
        }
        return (List<Map<String, Object>>) value;
    }

    public static int sessionId(Map<String, Object> session) {
        return intOf(session.get("id"), -1);
    }

    public Map<String, Object> session(int id) {
        for (Map<String, Object> session : sessions())
            if (sessionId(session) == id)
                return session;
        return null;
    }

    /** Append a new, open session record. */
    public Map<String, Object> addSession(int id, String kind, String user, String softwareVersion, String model) {
        Map<String, Object> s = new LinkedHashMap<>();
        s.put("id", id);
        s.put("kind", kind);
        s.put("started", now());
        s.put("ended", null);
        s.put("status", "open");
        s.put("user", user);
        s.put("host", Ids.hostId());
        s.put("software_version", softwareVersion);
        s.put("model", model);
        s.put("batch_size", 1);
        s.put("n_annotated", 0);
        s.put("note", null);
        sessions().add(s);
        return s;
    }

    /**
     * Take over what other computers published in the config on OMERO: their session
     * records (this computer's are kept) and the larger image counts.
     *
     * @param starting a session being started here; if another computer took its number
     *                 meanwhile it gets the next free one (pass null otherwise)
     * @param usedSessionIds session numbers the rows carry
     */
    public void mergeRemote(SetConfig remote, Map<String, Object> starting, List<Integer> usedSessionIds) {
        String host = Ids.hostId();
        Map<Integer, Map<String, Object>> merged = new LinkedHashMap<>();
        for (Map<String, Object> mine : sessions())
            merged.put(sessionId(mine), mine);
        for (Map<String, Object> theirs : remote.sessions()) {
            int id = sessionId(theirs);
            Map<String, Object> mine = merged.get(id);
            Object theirHost = theirs.get("host");
            if (mine != null && mine == starting && !host.equals(theirHost)) {
                merged.put(id, theirs);
                int max = 0;
                for (int taken : merged.keySet())
                    max = Math.max(max, taken);
                for (Map<String, Object> other : remote.sessions())
                    max = Math.max(max, sessionId(other));
                for (int used : usedSessionIds)
                    max = Math.max(max, used);
                mine.put("id", max + 1);
                merged.put(max + 1, mine);
            } else if (mine == null || (mine.get("host") != null && !host.equals(mine.get("host")))) {
                merged.put(id, theirs);
            }
        }
        List<Map<String, Object>> sorted = new ArrayList<>(merged.values());
        sorted.sort(Comparator.comparingInt(SetConfig::sessionId));
        root.put("sessions", sorted);
        Map<String, Object> training = section("training");
        Map<String, Object> theirTraining = remote.section("training");
        for (String key : List.of("train_n", "validate_n", "test_n")) {
            if (training.get(key) instanceof Number a && theirTraining.get(key) instanceof Number b)
                training.put(key, Math.max(a.intValue(), b.intValue()));
        }
    }

    /** The current local time as PyYAML writes a datetime. */
    public static Stamp now() {
        return new Stamp(LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSSSSS")));
    }

    // --- A new set ------------------------------------------------------------------

    /** What the user chooses when making a new training set in QuPath. */
    public static final class NewSet {
        public String name = "training_set";
        public boolean local = true;
        public String containerType = "dataset";
        public List<Long> containerIds = List.of();
        public int channel = 0;
        public List<Integer> timepoints = List.of(0);
        public List<Integer> zSlices = List.of(0);
        public boolean usePatches = false;
        public int patchWidth = 512;
        public int patchHeight = 512;
        public int patchesPerImage = 1;
        public boolean randomPatches = true;
        public int trainN = 3;
        public int validateN = 1;
        public int testN = 0;
        public boolean useAllImages = false;
        public double validationFraction = 0.2;
        public double testFraction = 0.0;
        public long seed = new java.security.SecureRandom().nextInt(Integer.MAX_VALUE);
        public String criteria = "";
        public String toolVersion = null;
        public Map<String, Object> qupathInfo = new LinkedHashMap<>();
    }

    /** The config of a new set, valid for the Python package (schema version 3). */
    public static SetConfig create(NewSet spec) {
        Map<String, Object> root = new LinkedHashMap<>();
        root.put("schema_version", SCHEMA_VERSION);
        root.put("name", spec.name);
        root.put("authors", new ArrayList<>());
        root.put("created", now());

        Map<String, Object> tool = new LinkedHashMap<>();
        tool.put("name", "QuPath");
        tool.put("version", spec.toolVersion);
        tool.put("mode", "interactive");
        tool.put("parameters", null);
        Map<String, Object> methodology = new LinkedHashMap<>();
        methodology.put("annotation_type", "segmentation_mask");
        methodology.put("annotation_method", "semi_automatic");
        methodology.put("annotation_criteria", spec.criteria == null ? "" : spec.criteria);
        methodology.put("annotation_coverage", "representative");
        methodology.put("annotation_tool", tool);
        root.put("annotation_methodology", methodology);

        Map<String, Object> coverage = new LinkedHashMap<>();
        coverage.put("channels", new ArrayList<>(List.of(spec.channel)));
        coverage.put("timepoints", new ArrayList<>(spec.timepoints));
        coverage.put("timepoint_mode", "specific");
        coverage.put("z_slices", new ArrayList<>(spec.zSlices));
        coverage.put("z_slice_mode", "specific");
        coverage.put("three_d", false);
        coverage.put("use_patches", spec.usePatches);
        if (spec.usePatches) {
            // The Python package reads patch_size as (height, width).
            coverage.put("patch_size", new ArrayList<>(List.of(spec.patchHeight, spec.patchWidth)));
            coverage.put("patches_per_image", spec.patchesPerImage);
            coverage.put("random_patches", spec.randomPatches);
        }
        root.put("spatial_coverage", coverage);

        Map<String, Object> training = new LinkedHashMap<>();
        training.put("split_unit", "image");
        if (!spec.useAllImages) { // with "all images" the counts mean nothing (Python warns about them)
            training.put("train_n", spec.trainN);
            training.put("validate_n", spec.validateN);
            training.put("test_n", spec.testN);
        }
        training.put("segment_all", spec.useAllImages);
        if (spec.useAllImages) {
            training.put("validation_fraction", spec.validationFraction);
            training.put("test_fraction", spec.testFraction);
            // Python requires the three fractions to add up to at most 1, in floating point.
            double train = Math.max(0.0, Math.floor((1.0 - spec.validationFraction - spec.testFraction) * 1000) / 1000.0);
            while (train > 0 && train + spec.validationFraction + spec.testFraction > 1.0)
                train = Math.max(0.0, Math.round((train - 0.001) * 1000) / 1000.0);
            training.put("train_fraction", train);
        }
        training.put("seed", spec.seed);
        root.put("training", training);

        Map<String, Object> omero = new LinkedHashMap<>();
        omero.put("container_type", spec.containerType);
        omero.put("container_id", spec.containerIds.isEmpty() ? 0 : spec.containerIds.get(0));
        if (spec.containerIds.size() > 1)
            omero.put("container_ids", new ArrayList<>(spec.containerIds));
        root.put("omero", omero);

        root.put("source", "omero");
        root.put("storage", spec.local ? "local" : "omero");
        Map<String, Object> identity = new LinkedHashMap<>();
        identity.put("server", null);
        identity.put("table_id", null);
        identity.put("local_uuid", null);
        root.put("identity", identity);
        root.put("sessions", new ArrayList<>());
        root.put("tags", new ArrayList<>(List.of("qupath")));
        Map<String, Object> custom = new LinkedHashMap<>();
        custom.put("qupath", spec.qupathInfo);
        root.put("custom_metadata", custom);
        return new SetConfig(root);
    }
}
