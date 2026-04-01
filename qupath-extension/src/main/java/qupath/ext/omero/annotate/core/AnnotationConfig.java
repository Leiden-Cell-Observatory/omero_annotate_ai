package qupath.ext.omero.annotate.core;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;
import com.fasterxml.jackson.dataformat.yaml.YAMLGenerator;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Workflow configuration compatible with the Python AnnotationConfig (schema v2.0.0).
 * <p>
 * Supports YAML round-trip serialization so that configs created in QuPath
 * can be read by the Python omero_annotate_ai package and vice versa.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class AnnotationConfig {

    private static final Logger logger = LoggerFactory.getLogger(AnnotationConfig.class);
    private static final ObjectMapper YAML_MAPPER = createYamlMapper();

    @JsonProperty("schema_version")
    private String schemaVersion = "2.0.0";

    @JsonProperty("name")
    private String name = "QuPath Annotation Workflow";

    @JsonProperty("version")
    private String version = "1.0.0";

    @JsonProperty("created")
    private String created = Instant.now().toString();

    @JsonProperty("authors")
    private List<AuthorInfo> authors = new ArrayList<>();

    @JsonProperty("study")
    private StudyContext study = new StudyContext();

    @JsonProperty("spatial_coverage")
    private SpatialCoverage spatialCoverage = new SpatialCoverage();

    @JsonProperty("training")
    private TrainingConfig training = new TrainingConfig();

    @JsonProperty("ai_model")
    private AIModelConfig aiModel = new AIModelConfig();

    @JsonProperty("workflow")
    private WorkflowConfig workflow = new WorkflowConfig();

    @JsonProperty("output")
    private OutputConfig output = new OutputConfig();

    @JsonProperty("omero")
    private OMEROConfig omero = new OMEROConfig();

    @JsonProperty("annotations")
    private List<ImageAnnotationRecord> annotations = new ArrayList<>();

    @JsonProperty("tags")
    private List<String> tags = new ArrayList<>();

    // --- Nested configuration classes ---

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class AuthorInfo {
        @JsonProperty("name")
        public String name = "";
        @JsonProperty("email")
        public String email;
        @JsonProperty("affiliation")
        public String affiliation;
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class StudyContext {
        @JsonProperty("title")
        public String title = "";
        @JsonProperty("description")
        public String description = "";
        @JsonProperty("organism")
        public String organism;
        @JsonProperty("imaging_method")
        public String imagingMethod;
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class SpatialCoverage {
        @JsonProperty("channels")
        public List<Integer> channels = List.of(0);
        @JsonProperty("timepoints")
        public List<Integer> timepoints = List.of(0);
        @JsonProperty("z_slices")
        public List<Integer> zSlices = List.of(0);
        @JsonProperty("three_d")
        public boolean threeD = false;
        @JsonProperty("use_patches")
        public boolean usePatches = true;
        @JsonProperty("patch_size")
        public List<Integer> patchSize = List.of(256, 256);
        @JsonProperty("patches_per_image")
        public int patchesPerImage = 1;
        @JsonProperty("label_channel")
        public Integer labelChannel;
        @JsonProperty("training_channels")
        public List<Integer> trainingChannels;
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class TrainingConfig {
        @JsonProperty("segment_all")
        public boolean segmentAll = false;
        @JsonProperty("train_fraction")
        public double trainFraction = 0.8;
        @JsonProperty("validation_fraction")
        public double validationFraction = 0.2;
        @JsonProperty("train_n")
        public Integer trainN;
        @JsonProperty("validate_n")
        public Integer validateN;
        @JsonProperty("test_n")
        public Integer testN;
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class AIModelConfig {
        @JsonProperty("framework")
        public String framework = "qupath";
        @JsonProperty("model_name")
        public String modelName;
        @JsonProperty("model_version")
        public String modelVersion;
        @JsonProperty("training_mode")
        public String trainingMode;
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class WorkflowConfig {
        @JsonProperty("read_only_mode")
        public boolean readOnlyMode = false;
        @JsonProperty("batch_size")
        public int batchSize = 0;
        @JsonProperty("resume_from_table")
        public boolean resumeFromTable = false;
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class OutputConfig {
        @JsonProperty("output_directory")
        public String outputDirectory;
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class OMEROConfig {
        @JsonProperty("container_type")
        public String containerType = "dataset";
        @JsonProperty("container_id")
        public Long containerId;
        @JsonProperty("container_ids")
        public List<Long> containerIds;
        @JsonProperty("table_id")
        public Long tableId;
    }

    // --- Annotation management ---

    public void addAnnotation(ImageAnnotationRecord record) {
        annotations.add(record);
    }

    public List<ImageAnnotationRecord> getUnprocessed() {
        return annotations.stream().filter(a -> !a.isProcessed()).toList();
    }

    public List<ImageAnnotationRecord> getProcessed() {
        return annotations.stream().filter(ImageAnnotationRecord::isProcessed).toList();
    }

    public void markCompleted(long imageId, Long roiId, Long labelId) {
        for (var annotation : annotations) {
            if (annotation.getImageId() == imageId) {
                annotation.markProcessed(roiId, labelId);
            }
        }
    }

    public Map<String, Object> getProgressSummary() {
        int total = annotations.size();
        int completed = getProcessed().size();
        var summary = new HashMap<String, Object>();
        summary.put("total_units", total);
        summary.put("completed_units", completed);
        summary.put("pending_units", total - completed);
        summary.put("progress_percent", total > 0 ? Math.round(1000.0 * completed / total) / 10.0 : 0);
        return summary;
    }

    // --- YAML serialization ---

    public static AnnotationConfig loadFromYaml(Path path) throws IOException {
        logger.info("Loading config from {}", path);
        return YAML_MAPPER.readValue(path.toFile(), AnnotationConfig.class);
    }

    public void saveToYaml(Path path) throws IOException {
        logger.info("Saving config to {}", path);
        YAML_MAPPER.writeValue(path.toFile(), this);
    }

    private static ObjectMapper createYamlMapper() {
        var factory = new YAMLFactory()
                .disable(YAMLGenerator.Feature.WRITE_DOC_START_MARKER)
                .enable(YAMLGenerator.Feature.MINIMIZE_QUOTES);
        var mapper = new ObjectMapper(factory);
        mapper.registerModule(new JavaTimeModule());
        mapper.disable(SerializationFeature.WRITE_DATES_AS_TIMESTAMPS);
        mapper.enable(SerializationFeature.INDENT_OUTPUT);
        return mapper;
    }

    // --- Getters and setters ---

    public String getSchemaVersion() { return schemaVersion; }
    public void setSchemaVersion(String schemaVersion) { this.schemaVersion = schemaVersion; }

    public String getName() { return name; }
    public void setName(String name) { this.name = name; }

    public String getVersion() { return version; }
    public void setVersion(String version) { this.version = version; }

    public String getCreated() { return created; }
    public void setCreated(String created) { this.created = created; }

    public List<AuthorInfo> getAuthors() { return authors; }
    public void setAuthors(List<AuthorInfo> authors) { this.authors = authors; }

    public StudyContext getStudy() { return study; }
    public void setStudy(StudyContext study) { this.study = study; }

    public SpatialCoverage getSpatialCoverage() { return spatialCoverage; }
    public void setSpatialCoverage(SpatialCoverage spatialCoverage) { this.spatialCoverage = spatialCoverage; }

    public TrainingConfig getTraining() { return training; }
    public void setTraining(TrainingConfig training) { this.training = training; }

    public AIModelConfig getAiModel() { return aiModel; }
    public void setAiModel(AIModelConfig aiModel) { this.aiModel = aiModel; }

    public WorkflowConfig getWorkflow() { return workflow; }
    public void setWorkflow(WorkflowConfig workflow) { this.workflow = workflow; }

    public OutputConfig getOutput() { return output; }
    public void setOutput(OutputConfig output) { this.output = output; }

    public OMEROConfig getOmero() { return omero; }
    public void setOmero(OMEROConfig omero) { this.omero = omero; }

    public List<ImageAnnotationRecord> getAnnotations() { return annotations; }
    public void setAnnotations(List<ImageAnnotationRecord> annotations) { this.annotations = annotations; }

    public List<String> getTags() { return tags; }
    public void setTags(List<String> tags) { this.tags = tags; }

    public boolean isLocalOnlyMode() {
        return workflow.readOnlyMode || !isOmeroConfigured();
    }

    public boolean isOmeroConfigured() {
        return omero.containerId != null && omero.containerId > 0;
    }
}
