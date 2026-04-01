package qupath.ext.omero.annotate;

import javafx.scene.control.MenuItem;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.omero.annotate.ui.AnnotateDialog;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.extensions.QuPathExtension;
import qupath.lib.gui.tools.MenuTools;

/**
 * QuPath extension for creating AI training data from annotations.
 * <p>
 * Supports two modes:
 * <ul>
 *     <li><b>Local-only</b>: Works with any image, exports patches + masks locally</li>
 *     <li><b>OMERO mode</b>: Full round-trip with OMERO server (tracking table, config sync)</li>
 * </ul>
 * <p>
 * Compatible with the Python omero_annotate_ai package for cross-tool workflows.
 */
public class OmeroAnnotateExtension implements QuPathExtension {

    private static final Logger logger = LoggerFactory.getLogger(OmeroAnnotateExtension.class);

    private static final String EXTENSION_NAME = "OMERO Annotate AI";
    private static final String EXTENSION_DESCRIPTION =
            "Create AI training data from QuPath annotations with optional OMERO integration";

    private AnnotateDialog dialog;

    @Override
    public void installExtension(QuPathGUI qupath) {
        logger.info("Installing {} extension", EXTENSION_NAME);

        var menuItem = new MenuItem("Open Annotate Dialog");
        menuItem.setOnAction(e -> showDialog(qupath));

        var quickExport = new MenuItem("Quick Export from Current Image");
        quickExport.setOnAction(e -> quickExport(qupath));

        MenuTools.addMenuItems(
                qupath.getMenu("Extensions>" + EXTENSION_NAME, true),
                menuItem,
                quickExport
        );
    }

    @Override
    public String getName() {
        return EXTENSION_NAME;
    }

    @Override
    public String getDescription() {
        return EXTENSION_DESCRIPTION;
    }

    /**
     * Returns the QuPath version this extension was built for.
     */
    @Override
    public String getQuPathVersion() {
        return "0.6.0";
    }

    private void showDialog(QuPathGUI qupath) {
        if (dialog == null) {
            dialog = new AnnotateDialog(qupath);
        }
        dialog.show();
    }

    private void quickExport(QuPathGUI qupath) {
        var imageData = qupath.getImageData();
        if (imageData == null) {
            logger.warn("No image is currently open");
            return;
        }
        // Open dialog directly on the export tab
        if (dialog == null) {
            dialog = new AnnotateDialog(qupath);
        }
        dialog.showAndSelectTab(2); // Run/Export tab
    }
}
