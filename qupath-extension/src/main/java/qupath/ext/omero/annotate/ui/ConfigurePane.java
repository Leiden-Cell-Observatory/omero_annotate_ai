package qupath.ext.omero.annotate.ui;

import javafx.geometry.Insets;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.omero.annotate.core.AnnotationConfig;
import qupath.lib.gui.QuPathGUI;

import java.io.File;
import java.nio.file.Path;
import java.util.List;

/**
 * Tab 2: Workflow configuration settings.
 * <p>
 * Provides controls for patch size, train/val split, channel selection,
 * OMERO container settings, and output directory.
 */
public class ConfigurePane extends ScrollPane {

    private static final Logger logger = LoggerFactory.getLogger(ConfigurePane.class);

    private final QuPathGUI qupath;

    // Annotation settings
    private final Spinner<Integer> patchWidthSpinner;
    private final Spinner<Integer> patchHeightSpinner;
    private final Slider trainSplitSlider;
    private final Label trainSplitLabel;
    private final CheckBox usePatchesCheckBox;
    private final TextField channelsField;

    // OMERO settings
    private final VBox omeroSettingsBox;
    private final ComboBox<String> containerTypeCombo;
    private final TextField containerIdField;

    // Output settings
    private final TextField outputDirField;
    private final TextField workflowNameField;

    // Config load
    private final TextField configPathField;

    public ConfigurePane(QuPathGUI qupath, ConnectionPane connectionPane) {
        this.qupath = qupath;
        setFitToWidth(true);

        var content = new VBox(12);
        content.setPadding(new Insets(15));

        // Workflow name
        workflowNameField = new TextField("QuPath Annotation Workflow");
        workflowNameField.setPromptText("Workflow name");
        var nameSection = createSection("Workflow", new HBox(8, new Label("Name:"), workflowNameField));
        HBox.setHgrow(workflowNameField, Priority.ALWAYS);

        // Patch settings
        usePatchesCheckBox = new CheckBox("Use patches (centroid-centered)");
        usePatchesCheckBox.setSelected(true);

        patchWidthSpinner = new Spinner<>(32, 2048, 256, 32);
        patchWidthSpinner.setEditable(true);
        patchWidthSpinner.setPrefWidth(100);
        patchHeightSpinner = new Spinner<>(32, 2048, 256, 32);
        patchHeightSpinner.setEditable(true);
        patchHeightSpinner.setPrefWidth(100);

        var patchSizeBox = new HBox(8,
                new Label("Width:"), patchWidthSpinner,
                new Label("Height:"), patchHeightSpinner
        );
        patchSizeBox.disableProperty().bind(usePatchesCheckBox.selectedProperty().not());

        // Train/val split
        trainSplitSlider = new Slider(0, 100, 80);
        trainSplitSlider.setShowTickLabels(true);
        trainSplitSlider.setShowTickMarks(true);
        trainSplitSlider.setMajorTickUnit(20);
        trainSplitSlider.setBlockIncrement(5);
        trainSplitLabel = new Label("Train: 80% / Val: 20%");
        trainSplitSlider.valueProperty().addListener((obs, oldVal, newVal) -> {
            int train = newVal.intValue();
            trainSplitLabel.setText(String.format("Train: %d%% / Val: %d%%", train, 100 - train));
        });

        // Channels
        channelsField = new TextField("0");
        channelsField.setPromptText("Channel indices (comma-separated, e.g. 0,1,2)");

        var patchSection = createSection("Annotation Settings",
                usePatchesCheckBox,
                new Label("Patch size (pixels):"),
                patchSizeBox,
                new Separator(),
                new Label("Train/Validation split:"),
                trainSplitSlider,
                trainSplitLabel,
                new Separator(),
                new HBox(8, new Label("Channels:"), channelsField)
        );

        // OMERO settings (hidden in local mode)
        containerTypeCombo = new ComboBox<>();
        containerTypeCombo.getItems().addAll("dataset", "project", "plate");
        containerTypeCombo.setValue("dataset");
        containerIdField = new TextField();
        containerIdField.setPromptText("OMERO container ID");

        omeroSettingsBox = createSection("OMERO Container",
                new HBox(8, new Label("Type:"), containerTypeCombo,
                        new Label("ID:"), containerIdField)
        );

        // Bind OMERO settings visibility to local mode
        connectionPane.getLocalModeCheckBox().selectedProperty().addListener((obs, oldVal, newVal) -> {
            omeroSettingsBox.setVisible(!newVal);
            omeroSettingsBox.setManaged(!newVal);
        });
        omeroSettingsBox.setVisible(!connectionPane.isLocalOnlyMode());
        omeroSettingsBox.setManaged(!connectionPane.isLocalOnlyMode());

        // Output directory
        outputDirField = new TextField();
        outputDirField.setPromptText("Output directory path");
        var browseButton = new Button("Browse...");
        browseButton.setOnAction(e -> browseOutputDir());
        var outputBox = new HBox(8, outputDirField, browseButton);
        HBox.setHgrow(outputDirField, Priority.ALWAYS);

        var outputSection = createSection("Output", outputBox);

        // Load config
        configPathField = new TextField();
        configPathField.setPromptText("Path to existing annotation_config.yaml");
        var loadConfigButton = new Button("Load Config");
        loadConfigButton.setOnAction(e -> loadConfig());
        var browseConfigButton = new Button("Browse...");
        browseConfigButton.setOnAction(e -> browseConfig());
        var configBox = new HBox(8, configPathField, browseConfigButton, loadConfigButton);
        HBox.setHgrow(configPathField, Priority.ALWAYS);

        var configSection = createSection("Load Existing Config", configBox);

        content.getChildren().addAll(nameSection, patchSection, omeroSettingsBox, outputSection, configSection);
        setContent(content);
    }

    /**
     * Build an AnnotationConfig from the current UI settings.
     */
    public AnnotationConfig buildConfig() {
        var config = new AnnotationConfig();
        config.setName(workflowNameField.getText());

        // Spatial coverage
        var spatial = config.getSpatialCoverage();
        spatial.usePatches = usePatchesCheckBox.isSelected();
        spatial.patchSize = List.of(patchHeightSpinner.getValue(), patchWidthSpinner.getValue());
        spatial.channels = parseChannels(channelsField.getText());

        // Training split
        var training = config.getTraining();
        training.trainFraction = trainSplitSlider.getValue() / 100.0;
        training.validationFraction = 1.0 - training.trainFraction;

        // AI model
        config.getAiModel().framework = "qupath";

        // Output
        String outputPath = outputDirField.getText().trim();
        if (!outputPath.isEmpty()) {
            config.getOutput().outputDirectory = outputPath;
        }

        // OMERO
        if (!containerIdField.getText().trim().isEmpty()) {
            try {
                config.getOmero().containerType = containerTypeCombo.getValue();
                config.getOmero().containerId = Long.parseLong(containerIdField.getText().trim());
            } catch (NumberFormatException e) {
                logger.warn("Invalid container ID: {}", containerIdField.getText());
            }
        }

        return config;
    }

    /**
     * Populate UI fields from an existing config.
     */
    public void loadFromConfig(AnnotationConfig config) {
        workflowNameField.setText(config.getName());

        var spatial = config.getSpatialCoverage();
        usePatchesCheckBox.setSelected(spatial.usePatches);
        if (spatial.patchSize.size() >= 2) {
            patchHeightSpinner.getValueFactory().setValue(spatial.patchSize.get(0));
            patchWidthSpinner.getValueFactory().setValue(spatial.patchSize.get(1));
        }
        channelsField.setText(spatial.channels.stream()
                .map(String::valueOf).reduce((a, b) -> a + "," + b).orElse("0"));

        var training = config.getTraining();
        trainSplitSlider.setValue(training.trainFraction * 100);

        if (config.getOutput().outputDirectory != null) {
            outputDirField.setText(config.getOutput().outputDirectory);
        }

        if (config.getOmero().containerId != null) {
            containerTypeCombo.setValue(config.getOmero().containerType);
            containerIdField.setText(String.valueOf(config.getOmero().containerId));
        }
    }

    public String getOutputDirectory() {
        return outputDirField.getText().trim();
    }

    private VBox createSection(String title, javafx.scene.Node... children) {
        var titleLabel = new Label(title);
        titleLabel.setStyle("-fx-font-weight: bold; -fx-font-size: 13px;");
        var box = new VBox(8);
        box.setPadding(new Insets(5, 0, 10, 0));
        box.getChildren().add(titleLabel);
        box.getChildren().addAll(children);
        return box;
    }

    private void browseOutputDir() {
        var chooser = new javafx.stage.DirectoryChooser();
        chooser.setTitle("Select Output Directory");
        File dir = chooser.showDialog(getScene().getWindow());
        if (dir != null) {
            outputDirField.setText(dir.getAbsolutePath());
        }
    }

    private void browseConfig() {
        var chooser = new javafx.stage.FileChooser();
        chooser.setTitle("Select Annotation Config");
        chooser.getExtensionFilters().add(
                new javafx.stage.FileChooser.ExtensionFilter("YAML files", "*.yaml", "*.yml")
        );
        File file = chooser.showOpenDialog(getScene().getWindow());
        if (file != null) {
            configPathField.setText(file.getAbsolutePath());
        }
    }

    private void loadConfig() {
        String path = configPathField.getText().trim();
        if (path.isEmpty()) return;

        try {
            var config = AnnotationConfig.loadFromYaml(Path.of(path));
            loadFromConfig(config);
            logger.info("Loaded config from {}", path);
        } catch (Exception e) {
            logger.error("Failed to load config: {}", e.getMessage());
            var alert = new Alert(Alert.AlertType.ERROR,
                    "Failed to load config: " + e.getMessage());
            alert.showAndWait();
        }
    }

    private List<Integer> parseChannels(String text) {
        try {
            return java.util.Arrays.stream(text.split(","))
                    .map(String::trim)
                    .filter(s -> !s.isEmpty())
                    .map(Integer::parseInt)
                    .toList();
        } catch (NumberFormatException e) {
            return List.of(0);
        }
    }
}
