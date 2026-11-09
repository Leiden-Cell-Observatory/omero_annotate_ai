package qupath.ext.omeroannotate;

import javafx.application.Platform;
import javafx.scene.control.Menu;
import javafx.scene.control.MenuItem;
import javafx.scene.control.Tab;
import javafx.scene.control.TabPane;
import javafx.stage.WindowEvent;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.omeroannotate.gui.TrainingSetPane;
import qupath.lib.common.Version;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.extensions.GitHubProject;
import qupath.lib.gui.extensions.QuPathExtension;
import qupath.lib.gui.tools.GuiTools;

import java.net.URI;

/**
 * QuPath extension of omero-annotate-ai: annotate training sets image by image in QuPath,
 * kept on OMERO (shared with the Python package and its napari plugin) or on this
 * computer. Adds a "Training set" tab next to Project, Image and Annotations.
 */
public class OmeroAnnotateExtension implements QuPathExtension, GitHubProject {

    private static final Logger logger = LoggerFactory.getLogger(OmeroAnnotateExtension.class);

    public static final String VERSION = "0.1.0";
    private static final String NAME = "OMERO Annotate AI";
    private static final String TAB_TITLE = "Training set";
    private static final String DOCS = "https://leiden-cell-observatory.github.io/omero_annotate_ai/qupath-plugin/";

    private boolean installed = false;
    private TrainingSetPane pane;
    private Tab tab;

    @Override
    public void installExtension(QuPathGUI qupath) {
        if (installed)
            return;
        installed = true;
        if (Platform.isFxApplicationThread())
            install(qupath);
        else
            Platform.runLater(() -> install(qupath));
    }

    private void install(QuPathGUI qupath) {
        try {
            pane = new TrainingSetPane(qupath);
            tab = new Tab(TAB_TITLE, pane);
            tab.setClosable(false);
            TabPane tabs = qupath.getAnalysisTabPane();
            tabs.getTabs().add(tab);

            MenuItem show = new MenuItem("Show the training set panel");
            show.setOnAction(e -> {
                if (!tabs.getTabs().contains(tab))
                    tabs.getTabs().add(tab);
                tabs.getSelectionModel().select(tab);
            });
            MenuItem help = new MenuItem("Help (web page)");
            help.setOnAction(e -> GuiTools.browseURI(URI.create(DOCS)));
            Menu menu = qupath.getMenu("Extensions>" + NAME, true);
            menu.getItems().addAll(show, help);

            qupath.getStage().addEventHandler(WindowEvent.WINDOW_HIDDEN, e -> pane.shutdown());
        } catch (RuntimeException e) {
            logger.error("The OMERO Annotate AI extension could not be installed", e);
        }
    }

    @Override
    public String getName() {
        return NAME;
    }

    @Override
    public String getDescription() {
        return "Annotate omero-annotate-ai training sets in QuPath, image by image, with any tool "
                + "(Cellpose, InstanSeg, SAM, brush). Sets are kept on OMERO or on this computer.";
    }

    @Override
    public Version getQuPathVersion() {
        return Version.parse("v0.6.0");
    }

    @Override
    public Version getVersion() {
        return Version.parse("v" + VERSION);
    }

    @Override
    public GitHubRepo getRepository() {
        return GitHubRepo.create(NAME, "Leiden-Cell-Observatory", "omero_annotate_ai");
    }
}
