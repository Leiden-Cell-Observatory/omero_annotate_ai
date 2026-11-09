package qupath.ext.omeroannotate.gui;

import javafx.geometry.Insets;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.Spinner;
import javafx.scene.layout.GridPane;
import qupath.fx.dialogs.Dialogs;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.scripting.QPEx;
import qupath.lib.gui.scripting.languages.GroovyLanguage;
import qupath.lib.images.ImageData;
import qupath.lib.plugins.PathPlugin;
import qupath.lib.scripting.ScriptParameters;

import java.awt.image.BufferedImage;
import java.util.List;
import java.util.Locale;

/**
 * Optional helpers that fill the frame with objects. They only start tools that exist
 * anyway: QuPath's own cell detection, and Cellpose through the BIOP Cellpose extension.
 * Any other tool (InstanSeg, StarDist, SAM, the brush) works too: run it on the frame.
 */
final class Detectors {

    private static final String CELLPOSE_CLASS = "qupath.ext.biop.cellpose.Cellpose2D";
    private static final String CELL_DETECTION_CLASS = "qupath.imagej.detect.cells.WatershedCellDetection";

    private Detectors() {}

    static boolean hasCellpose() {
        return exists(CELLPOSE_CLASS);
    }

    static boolean hasCellDetection() {
        return exists(CELL_DETECTION_CLASS);
    }

    private static boolean exists(String name) {
        try {
            Class.forName(name, false, Detectors.class.getClassLoader());
            return true;
        } catch (ClassNotFoundException | LinkageError e) {
            return false;
        }
    }

    /** Open QuPath's own cell detection for the selected frame (the user sets its parameters). */
    @SuppressWarnings("unchecked")
    static void runCellDetection(QuPathGUI qupath) {
        try {
            Class<?> cls = Class.forName(CELL_DETECTION_CLASS, true, Detectors.class.getClassLoader());
            qupath.runPlugin((PathPlugin<BufferedImage>) cls.getDeclaredConstructor().newInstance(), null, true);
        } catch (Exception e) {
            Dialogs.showErrorMessage("Cell detection", "QuPath's cell detection could not be started: " + e.getMessage());
        }
    }

    /** What the user chooses for a Cellpose run. */
    record CellposeSettings(String model, double diameter, int channel) {}

    /** Ask for the Cellpose model and object size; null if cancelled. */
    static CellposeSettings askCellpose(int channel, boolean rgb, String lastModel, double lastDiameter) {
        ComboBox<String> model = new ComboBox<>();
        model.getItems().addAll("cyto3", "nuclei", "cyto2", "cpsam");
        model.setEditable(true);
        model.setValue(lastModel);
        Spinner<Double> diameter = new Spinner<>(0.0, 2000.0, lastDiameter, 5.0);
        diameter.setEditable(true);
        GridPane grid = new GridPane();
        grid.setHgap(8);
        grid.setVgap(8);
        grid.setPadding(new Insets(6));
        grid.addRow(0, new Label("Model"), model);
        grid.addRow(1, new Label("Object diameter (pixels)"), diameter);
        Label hint = new Label("Diameter 0 lets Cellpose estimate the size. You can also give the path of your "
                + "own trained model." + (rgb ? "" : "\nCellpose runs on channel " + (channel + 1)
                + ", the channel this training set annotates."));
        hint.setWrapText(true);
        hint.setMaxWidth(380);
        grid.add(hint, 0, 2, 2, 1);
        if (!Dialogs.showConfirmDialog("Detect with Cellpose", grid))
            return null;
        String name = model.getEditor().getText() == null || model.getEditor().getText().isBlank()
                ? model.getValue() : model.getEditor().getText().strip();
        diameter.increment(0); // commit typed text
        return new CellposeSettings(name, diameter.getValue(), channel);
    }

    /** The Groovy script of a Cellpose run on the selected frame (shown to users who want to adapt it). */
    static String cellposeScript(CellposeSettings settings, boolean rgb) {
        String model = settings.model().replace("\\", "\\\\").replace("'", "\\'");
        StringBuilder sb = new StringBuilder();
        sb.append("import qupath.ext.biop.cellpose.Cellpose2D\n\n");
        sb.append("// Runs Cellpose inside the selected frame; every object found becomes an annotation.\n");
        sb.append("def frames = getSelectedObjects().findAll { it.isAnnotation() }\n");
        sb.append("if (frames.isEmpty()) { println 'Select the frame first'; return }\n");
        sb.append("def builder = Cellpose2D.builder('").append(model).append("')\n");
        if (!rgb)
            sb.append("        .channels(").append(settings.channel()).append(")\n");
        if (settings.diameter() > 0)
            sb.append(String.format(Locale.ROOT, "        .diameter(%.1f)\n", settings.diameter()));
        sb.append("        .createAnnotations()\n");
        sb.append("builder.build().detectObjects(getCurrentImageData(), frames)\n");
        return sb.toString();
    }

    /** Run a Groovy script on the open image. Call off the JavaFX thread. */
    static void runScript(QuPathGUI qupath, ImageData<BufferedImage> imageData, String script) throws Exception {
        ScriptParameters parameters = ScriptParameters.builder()
                .setScript(script)
                .setProject(qupath.getProject())
                .setImageData(imageData)
                .setDefaultImports(QPEx.getCoreClasses())
                .setDefaultStaticImports(List.of(QPEx.class))
                .setSystemWriters()
                .build();
        GroovyLanguage.getInstance().execute(parameters);
    }
}
