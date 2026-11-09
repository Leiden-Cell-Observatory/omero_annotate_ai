package qupath.ext.omeroannotate.gui;

import javafx.geometry.Insets;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.RadioButton;
import javafx.scene.control.Spinner;
import javafx.scene.control.TextField;
import javafx.scene.control.ToggleGroup;
import javafx.scene.control.Tooltip;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import qupath.ext.omeroannotate.core.SetConfig;
import qupath.fx.dialogs.Dialogs;

import java.util.List;

/**
 * The "New training set" dialog: which images, what to annotate in each, how many for
 * training, validation and test, and where the set is kept.
 */
final class NewSetDialog {

    /** What the dialog needs to know about the situation it is opened in. */
    static final class Context {
        /** Number of images in the open QuPath project (0: none or no project). */
        int projectImages = 0;
        /** Null if sets can be kept on OMERO, else why not (shown to the user). */
        String omeroUnavailable = "Log in to OMERO first.";
        /** A dataset or project to suggest: {@code type, id, name}; may be null. */
        String[] suggestedContainer = null;
        List<String> channelNames = List.of("Channel 1");
        boolean rgb = false;
        int sizeZ = 1;
        int sizeT = 1;
        int currentZ = 0;
        int currentT = 0;
        String suggestedName = "training_set";
    }

    /** What the user asked for. */
    record Request(SetConfig.NewSet spec, boolean fromOmeroContainer, String containerType, long containerId) {}

    private NewSetDialog() {}

    private static Label heading(String text) {
        Label label = new Label(text);
        label.setStyle("-fx-font-weight: bold; -fx-padding: 8 0 0 0;");
        return label;
    }

    /** A label that is never shortened to "…". */
    private static Label fixed(String text) {
        Label label = new Label(text);
        label.setMinWidth(javafx.scene.layout.Region.USE_PREF_SIZE);
        return label;
    }

    private static Spinner<Integer> spinner(int min, int max, int value, int width) {
        Spinner<Integer> spinner = new Spinner<>(min, max, value);
        spinner.setEditable(true);
        spinner.setPrefWidth(width);
        // Take over a typed number when the field loses focus, not only on Enter.
        spinner.focusedProperty().addListener((v, o, focused) -> {
            if (!focused)
                commit(spinner);
        });
        return spinner;
    }

    private static void commit(Spinner<Integer> spinner) {
        try {
            spinner.getValueFactory().setValue(Integer.parseInt(spinner.getEditor().getText().strip()));
        } catch (RuntimeException e) {
            spinner.getEditor().setText(String.valueOf(spinner.getValue()));
        }
    }

    /** Show the dialog; null if cancelled. */
    static Request show(Context context) {
        TextField name = new TextField(context.suggestedName);
        name.setPromptText("e.g. nuclei_dapi");

        // --- Images ---
        ToggleGroup source = new ToggleGroup();
        RadioButton fromProject = new RadioButton("The images of this QuPath project (" + context.projectImages + ")");
        fromProject.setToggleGroup(source);
        fromProject.setMinWidth(javafx.scene.layout.Region.USE_PREF_SIZE);
        fromProject.setDisable(context.projectImages == 0);
        RadioButton fromOmero = new RadioButton("All images of an OMERO");
        fromOmero.setToggleGroup(source);
        fromOmero.setMinWidth(javafx.scene.layout.Region.USE_PREF_SIZE);
        ComboBox<String> containerType = new ComboBox<>();
        containerType.getItems().addAll("dataset", "project");
        containerType.setValue(context.suggestedContainer == null ? "dataset" : context.suggestedContainer[0]);
        TextField containerId = new TextField(context.suggestedContainer == null ? "" : context.suggestedContainer[1]);
        containerId.setPromptText("ID");
        containerId.setPrefWidth(70);
        Label containerName = new Label(context.suggestedContainer == null ? "" : context.suggestedContainer[2]);
        containerName.setStyle("-fx-opacity: 0.7;");
        HBox omeroRow = new HBox(6, fromOmero, containerType, containerId, containerName);
        omeroRow.setAlignment(javafx.geometry.Pos.CENTER_LEFT);
        boolean omeroOk = context.omeroUnavailable == null;
        fromOmero.setDisable(!omeroOk);
        if (!omeroOk)
            Tooltip.install(omeroRow, new Tooltip(context.omeroUnavailable));
        containerType.disableProperty().bind(fromOmero.selectedProperty().not());
        containerId.disableProperty().bind(fromOmero.selectedProperty().not());
        if (context.projectImages > 0)
            fromProject.setSelected(true);
        else if (omeroOk)
            fromOmero.setSelected(true);

        // --- Where it is kept ---
        ToggleGroup storage = new ToggleGroup();
        RadioButton local = new RadioButton("On this computer");
        local.setToggleGroup(storage);
        local.setSelected(true);
        RadioButton onOmero = new RadioButton("On OMERO: masks and ROIs are uploaded, colleagues can join in");
        onOmero.setToggleGroup(storage);
        onOmero.setDisable(!omeroOk);
        Label omeroHint = new Label(omeroOk ? "" : context.omeroUnavailable);
        omeroHint.setWrapText(true);
        omeroHint.setMaxWidth(460);
        omeroHint.setStyle("-fx-opacity: 0.7; -fx-font-size: 0.9em;");
        omeroHint.setManaged(!omeroOk);
        omeroHint.setVisible(!omeroOk);

        // --- What to annotate ---
        ComboBox<String> channel = new ComboBox<>();
        for (int i = 0; i < context.channelNames.size(); i++)
            channel.getItems().add((i + 1) + ": " + context.channelNames.get(i));
        channel.getSelectionModel().select(0);
        channel.setDisable(context.rgb);
        Label channelNote = new Label(context.rgb ? "Colour image: all three colours are kept" : "");
        channelNote.setStyle("-fx-opacity: 0.7;");
        Spinner<Integer> z = spinner(1, Math.max(1, context.sizeZ), Math.min(context.sizeZ, context.currentZ + 1), 80);
        z.setDisable(context.sizeZ <= 1);
        Spinner<Integer> t = spinner(1, Math.max(1, context.sizeT), Math.min(context.sizeT, context.currentT + 1), 80);
        t.setDisable(context.sizeT <= 1);
        TextField criteria = new TextField();
        criteria.setPromptText("e.g. every nucleus, also dividing ones; skip debris");

        ToggleGroup region = new ToggleGroup();
        RadioButton whole = new RadioButton("The whole image");
        whole.setToggleGroup(region);
        whole.setSelected(true);
        RadioButton patches = new RadioButton("Patches of");
        patches.setToggleGroup(region);
        patches.setMinWidth(javafx.scene.layout.Region.USE_PREF_SIZE);
        Spinner<Integer> patchSize = spinner(32, 8192, 512, 90);
        Spinner<Integer> patchCount = spinner(1, 100, 2, 70);
        CheckBox random = new CheckBox("at random positions");
        random.setSelected(true);
        HBox patchRow = new HBox(6, patches, patchSize, fixed("pixels square,"), patchCount, fixed("per image"));
        patchRow.setAlignment(javafx.geometry.Pos.CENTER_LEFT);
        random.setPadding(new Insets(0, 0, 0, 26));
        patchSize.disableProperty().bind(patches.selectedProperty().not());
        patchCount.disableProperty().bind(patches.selectedProperty().not());
        random.disableProperty().bind(patches.selectedProperty().not());

        // --- How many ---
        // Suggested counts fit the images there are (unknown for OMERO until it is asked).
        int available = context.projectImages > 0 ? context.projectImages : 4;
        Spinner<Integer> train = spinner(0, 100000, Math.max(1, Math.min(3, available - 1)), 80);
        Spinner<Integer> validate = spinner(0, 100000, available >= 2 ? 1 : 0, 80);
        Spinner<Integer> test = spinner(0, 100000, 0, 80);
        CheckBox all = new CheckBox("Use all images:");
        all.setMinWidth(javafx.scene.layout.Region.USE_PREF_SIZE);
        Spinner<Integer> valPercent = spinner(0, 90, 20, 70);
        Spinner<Integer> testPercent = spinner(0, 90, 0, 70);
        HBox countRow = new HBox(6, fixed("Training"), train, fixed("Validation"), validate,
                fixed("Test"), test);
        countRow.setAlignment(javafx.geometry.Pos.CENTER_LEFT);
        HBox allRow = new HBox(6, all, valPercent, fixed("% for validation,"), testPercent, fixed("% for test"));
        allRow.setAlignment(javafx.geometry.Pos.CENTER_LEFT);
        countRow.disableProperty().bind(all.selectedProperty());
        valPercent.disableProperty().bind(all.selectedProperty().not());
        testPercent.disableProperty().bind(all.selectedProperty().not());
        Label countHint = new Label("Images are picked at random and each stays in one group. "
                + "Validation images are used to check a model while it trains; test images only afterwards.");
        countHint.setWrapText(true);
        countHint.setMaxWidth(460);
        countHint.setStyle("-fx-opacity: 0.7; -fx-font-size: 0.9em;");

        GridPane what = new GridPane();
        what.setHgap(8);
        what.setVgap(6);
        what.addRow(0, new Label("Channel"), channel, channelNote);
        what.addRow(1, new Label("Z-slice"), z);
        what.addRow(2, new Label("Timepoint"), t);

        Label error = new Label();
        error.setWrapText(true);
        error.setMaxWidth(460);
        error.setStyle("-fx-text-fill: #d64545; -fx-font-weight: bold;");
        error.setManaged(false);
        error.setVisible(false);

        VBox content = new VBox(6,
                new Label("Name"), name,
                heading("Images"), fromProject, omeroRow,
                heading("What to annotate in each image"), what, whole, patchRow, random,
                new Label("What counts as an object? (optional)"), criteria,
                heading("How many images"), countRow, allRow, countHint,
                heading("Where to keep the training set"), local, onOmero, omeroHint,
                error);
        content.setPadding(new Insets(8));
        content.setPrefWidth(500);

        while (true) {
            if (!Dialogs.showConfirmDialog("New training set", content))
                return null;
            for (Spinner<Integer> spinner : List.of(z, t, patchSize, patchCount, train, validate, test, valPercent, testPercent))
                commit(spinner);
            String problem = null;
            long id = 0;
            if (name.getText() == null || name.getText().isBlank())
                problem = "Give the training set a name.";
            else if (source.getSelectedToggle() == null)
                problem = "There are no images to choose from: open a QuPath project with images, or log in to OMERO.";
            else if (fromOmero.isSelected()) {
                try {
                    id = Long.parseLong(containerId.getText().strip());
                } catch (RuntimeException e) {
                    problem = "Enter the ID of the OMERO " + containerType.getValue() + " (the number shown in OMERO).";
                }
            }
            if (problem == null && !all.isSelected() && train.getValue() + validate.getValue() + test.getValue() == 0)
                problem = "Choose at least one image to annotate.";
            if (problem == null && all.isSelected() && valPercent.getValue() + testPercent.getValue() >= 100)
                problem = "Validation and test together must leave images for training.";
            if (problem != null) {
                error.setText(problem);
                error.setManaged(true);
                error.setVisible(true);
                continue;
            }
            SetConfig.NewSet spec = new SetConfig.NewSet();
            spec.name = name.getText().strip();
            spec.local = local.isSelected();
            spec.channel = context.rgb ? 0 : Math.max(0, channel.getSelectionModel().getSelectedIndex());
            spec.zSlices = List.of(z.getValue() - 1);
            spec.timepoints = List.of(t.getValue() - 1);
            spec.usePatches = patches.isSelected();
            spec.patchWidth = patchSize.getValue();
            spec.patchHeight = patchSize.getValue();
            spec.patchesPerImage = patchCount.getValue();
            spec.randomPatches = random.isSelected();
            spec.trainN = train.getValue();
            spec.validateN = validate.getValue();
            spec.testN = test.getValue();
            spec.useAllImages = all.isSelected();
            spec.validationFraction = valPercent.getValue() / 100.0;
            spec.testFraction = testPercent.getValue() / 100.0;
            spec.criteria = criteria.getText() == null ? "" : criteria.getText().strip();
            if (fromOmero.isSelected()) {
                spec.containerType = containerType.getValue();
                spec.containerIds = List.of(id);
            }
            return new Request(spec, fromOmero.isSelected(), containerType.getValue(), id);
        }
    }
}
