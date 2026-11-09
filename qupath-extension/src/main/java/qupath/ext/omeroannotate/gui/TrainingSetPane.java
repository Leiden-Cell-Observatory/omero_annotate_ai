package qupath.ext.omeroannotate.gui;

import javafx.application.Platform;
import javafx.beans.property.StringProperty;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Node;
import javafx.scene.control.Button;
import javafx.scene.control.ContentDisplay;
import javafx.scene.control.Label;
import javafx.scene.control.ListCell;
import javafx.scene.control.ListView;
import javafx.scene.control.MenuButton;
import javafx.scene.control.MenuItem;
import javafx.scene.control.ProgressBar;
import javafx.scene.control.ProgressIndicator;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.Separator;
import javafx.scene.control.SeparatorMenuItem;
import javafx.scene.control.TextField;
import javafx.scene.control.TitledPane;
import javafx.scene.control.Tooltip;
import javafx.scene.input.KeyCode;
import javafx.scene.input.KeyCodeCombination;
import javafx.scene.input.KeyCombination;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.Region;
import javafx.scene.layout.VBox;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.omeroannotate.OmeroAnnotateExtension;
import qupath.ext.omeroannotate.core.AnnotationSession;
import qupath.ext.omeroannotate.core.Bundle;
import qupath.ext.omeroannotate.core.Ids;
import qupath.ext.omeroannotate.core.PixelArray;
import qupath.ext.omeroannotate.core.Planning;
import qupath.ext.omeroannotate.core.RemoteStore;
import qupath.ext.omeroannotate.core.Row;
import qupath.ext.omeroannotate.core.SetConfig;
import qupath.ext.omeroannotate.core.TrainingSet;
import qupath.ext.omeroannotate.omero.OmeroAccess;
import qupath.ext.omeroannotate.qp.Exporter;
import qupath.ext.omeroannotate.qp.Images;
import qupath.ext.omeroannotate.qp.Units;
import qupath.fx.dialogs.Dialogs;
import qupath.fx.dialogs.FileChoosers;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.prefs.PathPrefs;
import qupath.lib.gui.tools.GuiTools;
import qupath.lib.gui.viewer.QuPathViewer;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ImageChannel;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.hierarchy.PathObjectHierarchy;
import qupath.lib.objects.hierarchy.events.PathObjectHierarchyListener;
import qupath.lib.projects.Project;
import qupath.lib.projects.ProjectIO;
import qupath.lib.projects.ProjectImageEntry;
import qupath.lib.projects.Projects;

import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

/**
 * The "Training set" panel: choose or make a training set, then annotate its images one
 * by one and finish with a note.
 * <p>
 * The panel has three faces: the chooser (no set open), the overview of an open set
 * (progress per split, sessions) and the annotation view (one image at a time). Anything
 * that talks to OMERO or reads pixels runs on a background thread, so QuPath stays
 * responsive; QuPath objects are only touched on the JavaFX thread.
 */
public final class TrainingSetPane extends BorderPane {

    private static final Logger logger = LoggerFactory.getLogger(TrainingSetPane.class);
    private static final KeyCombination ACCEPT_KEY = new KeyCodeCombination(KeyCode.ENTER, KeyCombination.SHORTCUT_DOWN);

    private final QuPathGUI qupath;
    private final ExecutorService worker = Executors.newSingleThreadExecutor(r -> {
        Thread thread = new Thread(r, "omero-annotate-ai");
        thread.setDaemon(true);
        return thread;
    });
    private final StringProperty omeroAddress =
            PathPrefs.createPersistentPreference("omeroAnnotateAi.omeroAddress", "");
    private final StringProperty lastCellposeModel =
            PathPrefs.createPersistentPreference("omeroAnnotateAi.cellposeModel", "cyto3");

    // Connection and set
    private RemoteStore remote;
    private OmeroAccess.Login login;
    private TrainingSet set;

    // Session
    private AnnotationSession session;
    private final List<Row> history = new ArrayList<>();
    private int position = -1;
    private Row current;
    private boolean confirmEmpty = false;
    private boolean noMoreUnits = false;
    /** Units accepted, skipped or excluded in this session (a review counts images looked at). */
    private final Set<String> handled = new LinkedHashSet<>();
    private PathObjectHierarchy watched;
    private final PathObjectHierarchyListener objectListener = event -> {
        if (!event.isChanging())
            Platform.runLater(this::updateObjectCount);
    };

    // Shared widgets
    private final ScrollPane scroll = new ScrollPane();
    private final Label status = new Label();
    private final Label busyText = new Label();
    private final ProgressIndicator busy = new ProgressIndicator();
    private boolean isBusy = false;

    // Chooser widgets
    private final ListView<SetChoice> choices = new ListView<>();
    private final Label omeroStatus = new Label();

    // Session widgets
    private final Label unitTitle = new Label();
    private final Label unitDetail = new Label();
    private final Label unitState = new Label();
    private final Label objectCount = new Label();
    private final Button accept = new Button("Accept and next");
    private final Button skip = new Button("Skip");
    private final Button exclude = new Button("Exclude…");
    private final Button previous = new Button("◀ Previous");
    private final TextField note = new TextField();
    private final ProgressBar sessionProgress = new ProgressBar(0);
    private final Label sessionProgressText = new Label();

    public TrainingSetPane(QuPathGUI qupath) {
        this.qupath = qupath;
        scroll.setFitToWidth(true);
        scroll.setStyle("-fx-background-color: transparent;");
        setCenter(scroll);

        busy.setPrefSize(16, 16);
        busy.setMaxSize(16, 16);
        status.setWrapText(true);
        status.setMaxWidth(Double.MAX_VALUE);
        status.setMinHeight(Region.USE_PREF_SIZE);
        busyText.setStyle("-fx-opacity: 0.75;");
        HBox busyRow = new HBox(6, busy, busyText);
        busyRow.setAlignment(Pos.CENTER_LEFT);
        busyRow.visibleProperty().bind(busy.visibleProperty());
        busyRow.managedProperty().bind(busy.visibleProperty());
        busy.setVisible(false);
        status.managedProperty().bind(status.textProperty().isNotEmpty());
        VBox statusBar = new VBox(4, status, busyRow);
        statusBar.setPadding(new Insets(6, 10, 6, 10));
        statusBar.setMinHeight(30);
        statusBar.setStyle("-fx-border-color: -fx-box-border transparent transparent transparent; -fx-border-width: 1 0 0 0;");
        setBottom(statusBar);

        showChooser();
    }

    // =================================================================================
    // Small building blocks
    // =================================================================================

    private static Label title(String text) {
        Label label = new Label(text);
        label.setStyle("-fx-font-size: 1.25em; -fx-font-weight: bold;");
        label.setWrapText(true);
        return label;
    }

    private static Label heading(String text) {
        Label label = new Label(text);
        label.setStyle("-fx-font-weight: bold; -fx-padding: 6 0 0 0;");
        return label;
    }

    private static Label hint(String text) {
        Label label = new Label(text);
        label.setWrapText(true);
        label.setStyle("-fx-opacity: 0.75;");
        return label;
    }

    private static Node banner(String text, String color, Node... actions) {
        Label label = new Label(text);
        label.setWrapText(true);
        VBox box = new VBox(6, label);
        if (actions.length > 0)
            box.getChildren().add(new HBox(6, actions));
        box.setPadding(new Insets(8));
        box.setStyle("-fx-background-color: " + color + "; -fx-background-radius: 6;");
        return box;
    }

    private static VBox page(Node... nodes) {
        VBox box = new VBox(8, nodes);
        box.setPadding(new Insets(10));
        box.setFillWidth(true);
        return box;
    }

    private static void fullWidth(Button... buttons) {
        for (Button button : buttons) {
            button.setMaxWidth(Double.MAX_VALUE);
            HBox.setHgrow(button, Priority.ALWAYS);
        }
    }

    private void say(String message) {
        status.setStyle("");
        status.setText(message == null ? "" : message);
    }

    private void warn(String message) {
        status.setStyle("-fx-text-fill: #c77700; -fx-font-weight: bold;");
        status.setText(message);
    }

    private static String message(Throwable error) {
        Throwable cause = error;
        while (cause.getCause() != null && (cause.getMessage() == null
                || cause instanceof java.util.concurrent.ExecutionException))
            cause = cause.getCause();
        String text = cause.getMessage();
        return text == null || text.isBlank() ? cause.getClass().getSimpleName() : text;
    }

    /** An error text without the Java class name in front. */
    private static String plain(String error) {
        return error == null ? "" : error.replaceFirst("^[A-Za-z.]*(Exception|Error): ", "");
    }

    private void fail(String what, Throwable error) {
        logger.warn("{}: {}", what, error.toString(), error);
        status.setStyle("-fx-text-fill: #d64545; -fx-font-weight: bold;");
        status.setText(what + ": " + message(error));
    }

    /**
     * Run {@code task} on the background thread while the panel is disabled and shows
     * {@code message}; then hand its result to {@code done} on the JavaFX thread.
     */
    private <T> void background(String message, Callable<T> task, Consumer<T> done) {
        background(message, task, done, error -> fail("That did not work", error));
    }

    private <T> void background(String message, Callable<T> task, Consumer<T> done, Consumer<Throwable> failed) {
        setBusy(true, message);
        worker.submit(() -> {
            try {
                T result = task.call();
                Platform.runLater(() -> {
                    setBusy(false, null);
                    try {
                        done.accept(result);
                    } catch (RuntimeException e) {
                        fail("That did not work", e);
                    }
                });
            } catch (Throwable error) {
                Platform.runLater(() -> {
                    setBusy(false, null);
                    failed.accept(error);
                });
            }
        });
    }

    private void setBusy(boolean value, String message) {
        isBusy = value;
        busy.setVisible(value);
        if (scroll.getContent() != null)
            scroll.getContent().setDisable(value);
        busyText.setText(value && message != null ? message : "");
    }

    /** "1 image", "3 images", "2 patches". */
    private static String count(long n, String noun) {
        return n + " " + (n == 1 ? noun : noun.endsWith("ch") ? noun + "es" : noun + "s");
    }

    /** What one unit of the open set is called. */
    private String unitNoun() {
        return set != null && set.config().usesPatches() ? "patch" : "image";
    }

    private Path home() {
        return Bundle.defaultHome();
    }

    private String softwareVersion() {
        return "qupath-extension-omero-annotate-ai " + OmeroAnnotateExtension.VERSION + " (QuPath "
                + QuPathGUI.getVersion() + ")";
    }

    // =================================================================================
    // OMERO
    // =================================================================================

    /** Join the OMERO extension's login if not connected yet. Background thread. */
    private RemoteStore connect() throws Exception {
        return connect(null);
    }

    /**
     * Join the OMERO extension's login. Background thread.
     * <p>
     * A connection made earlier is used again as long as the OMERO extension is still
     * logged in with it and the server answers; otherwise a new one is made, so logging
     * in again (or as someone else) in the OMERO extension is picked up here.
     *
     * @param server folder name of the server a set belongs to, to pick the right login
     *               when QuPath is logged in to several servers; may be null
     */
    private RemoteStore connect(String server) throws Exception {
        String missing = OmeroAccess.missing();
        if (missing != null)
            throw new IllegalStateException(missing);
        List<OmeroAccess.Login> logins = OmeroAccess.logins();
        if (logins.isEmpty()) {
            dropConnection();
            throw new IllegalStateException("You are not logged in to OMERO. Log in with "
                    + "Extensions ▸ OMERO ▸ Browse server, then try again.");
        }
        if (remote != null) {
            boolean sameLogin = login != null
                    && logins.stream().anyMatch(l -> l.sessionId().equals(login.sessionId()));
            boolean rightServer = server == null || server.equals(remote.serverSlug()) || logins.size() == 1;
            if (sameLogin && rightServer) {
                try {
                    remote.containersOfImage(-1); // a cheap question: is the server still there?
                    return remote;
                } catch (Exception e) {
                    logger.debug("The OMERO connection is gone; connecting again", e);
                }
            }
            dropConnection();
        }
        Exception last = null;
        RemoteStore first = null;
        OmeroAccess.Login firstLogin = null;
        for (OmeroAccess.Login candidate : logins) {
            RemoteStore store;
            try {
                store = OmeroAccess.connect(candidate, omeroAddress.get());
            } catch (Exception e) {
                last = e;
                continue;
            }
            if (server == null || server.equals(store.serverSlug())) {
                if (first != null)
                    first.close();
                login = candidate;
                remote = store;
                return remote;
            }
            if (first == null) {
                first = store;
                firstLogin = candidate;
            } else {
                store.close();
            }
        }
        if (first == null)
            throw last;
        login = firstLogin; // no login matches the set's server name: the first one that works
        remote = first;
        return remote;
    }

    private void dropConnection() {
        RemoteStore old = remote;
        remote = null;
        if (old != null) {
            try {
                old.close();
            } catch (RuntimeException e) {
                logger.debug("Closing a dead OMERO connection", e);
            }
        }
    }

    private void disconnect() {
        RemoteStore old = remote;
        remote = null;
        if (old != null)
            worker.submit(old::close);
    }

    private String webUri() {
        if (login != null)
            return login.webUri();
        List<OmeroAccess.Login> logins = OmeroAccess.logins();
        return logins.isEmpty() ? null : logins.get(0).webUri();
    }

    /** Null if sets can be kept on OMERO right now, else why not. */
    private String omeroUnavailable() {
        String missing = OmeroAccess.missing();
        if (missing != null)
            return missing;
        if (OmeroAccess.logins().isEmpty())
            return "Log in to OMERO first (Extensions ▸ OMERO ▸ Browse server).";
        return null;
    }

    private void openOmeroBrowser() {
        try {
            var menu = qupath.getMenu("Extensions>OMERO", false);
            if (menu != null) {
                for (MenuItem item : menu.getItems()) {
                    if (item.getText() != null && item.getText().toLowerCase().contains("browse")) {
                        if (item instanceof javafx.scene.control.Menu sub && !sub.getItems().isEmpty())
                            sub.getItems().get(sub.getItems().size() - 1).fire();
                        else
                            item.fire();
                        return;
                    }
                }
            }
        } catch (RuntimeException e) {
            logger.debug("OMERO menu not found", e);
        }
        Dialogs.showPlainMessage("Log in to OMERO", OmeroAccess.hasOmeroExtension()
                ? "Use Extensions ▸ OMERO ▸ Browse server to log in, then press Refresh here."
                : OmeroAccess.missing());
    }

    // =================================================================================
    // Face 1: the chooser
    // =================================================================================

    private void showChooser() {
        choices.setPlaceholder(hint("No training sets yet.\nMake one with \"New training set…\"."));
        choices.setPrefHeight(230);
        choices.setCellFactory(list -> new ListCell<>() {
            @Override
            protected void updateItem(SetChoice item, boolean empty) {
                super.updateItem(item, empty);
                if (empty || item == null) {
                    setGraphic(null);
                    setText(null);
                    return;
                }
                Label name = new Label(item.name());
                name.setStyle("-fx-font-weight: bold;");
                Label detail = new Label(item.detail());
                detail.setStyle("-fx-opacity: 0.75; -fx-font-size: 0.9em;");
                setGraphic(new VBox(1, name, detail));
                setContentDisplay(ContentDisplay.GRAPHIC_ONLY);
            }
        });
        choices.setOnMouseClicked(event -> {
            if (event.getClickCount() == 2 && choices.getSelectionModel().getSelectedItem() != null)
                openChoice(choices.getSelectionModel().getSelectedItem());
        });

        Button open = new Button("Open");
        open.setDefaultButton(true);
        open.disableProperty().bind(choices.getSelectionModel().selectedItemProperty().isNull());
        open.setOnAction(e -> openChoice(choices.getSelectionModel().getSelectedItem()));
        Button create = new Button("New training set…");
        create.setOnAction(e -> newSet());
        Button refresh = new Button("Refresh");
        refresh.setTooltip(new Tooltip("Look again for training sets on this computer and on OMERO"));
        refresh.setOnAction(e -> refreshChoices());
        fullWidth(open, create);
        HBox buttons = new HBox(6, open, create, refresh);

        MenuItem openFolder = new MenuItem("Open a training set folder…");
        openFolder.setOnAction(e -> {
            File folder = FileChoosers.promptForDirectory("Training set folder", home().toFile());
            if (folder != null)
                openChoice(new SetChoice(folder.getName(), folder.toPath(), null, null, ""));
        });
        MenuItem byTable = new MenuItem("Open a set on OMERO by its table ID…");
        byTable.setOnAction(e -> {
            String text = Dialogs.showInputDialog("Training set on OMERO",
                    "ID of the set's tracking table (shown in OMERO and in the Python package)", "");
            if (text == null || text.isBlank())
                return;
            try {
                long id = Long.parseLong(text.strip());
                openChoice(new SetChoice("table " + id, null, id, null, ""));
            } catch (NumberFormatException ex) {
                warn("A table ID is a number.");
            }
        });
        MenuItem showHome = new MenuItem("Show the training sets folder");
        showHome.setOnAction(e -> {
            try {
                Files.createDirectories(home());
                GuiTools.browseDirectory(home().toFile());
            } catch (Exception ex) {
                fail("The folder could not be opened", ex);
            }
        });
        MenuButton more = new MenuButton("More", null, openFolder, byTable, new SeparatorMenuItem(), showHome);

        omeroStatus.setWrapText(true);
        Button loginButton = new Button("Log in to OMERO…");
        loginButton.setOnAction(e -> openOmeroBrowser());
        TextField address = new TextField();
        address.setPromptText("host:port (leave empty: found automatically)");
        address.textProperty().bindBidirectional(omeroAddress);
        address.textProperty().addListener((v, o, n) -> disconnect());
        VBox advancedBox = new VBox(4, hint("Only needed if your OMERO server is not found by itself, for "
                + "example when OMERO.web runs on another machine than the server."), address);
        advancedBox.setPadding(new Insets(6));
        TitledPane advanced = new TitledPane("OMERO server address", advancedBox);
        advanced.setExpanded(false);
        advanced.setAnimated(false);

        VBox page = page(
                title("Training sets"),
                hint("A training set is a selection of images that you annotate one by one. "
                        + "The result is ready to train Cellpose, micro-SAM and other models."),
                choices, buttons, more,
                new Separator(),
                heading("OMERO"), omeroStatus, loginButton, advanced);
        scroll.setContent(page);
        refreshChoices();
    }

    private void refreshChoices() {
        String unavailable = omeroUnavailable();
        Long openImage = qupath.getImageData() == null ? null : OmeroAccess.imageId(qupath.getImageData().getServer());
        background("Looking for training sets…", () -> {
            List<SetChoice> found = new ArrayList<>(SetChoice.local(home()));
            String omero;
            if (unavailable != null) {
                omero = unavailable;
            } else {
                try {
                    RemoteStore store = connect();
                    omero = "Connected as " + store.userName() + " (" + login.label() + ").";
                    if (openImage != null) {
                        Set<Long> known = new LinkedHashSet<>();
                        for (SetChoice choice : found)
                            if (choice.isOnOmero() && store.serverSlug().equals(choice.server()))
                                known.add(choice.tableId());
                        for (String[] container : store.containersOfImage(openImage)) {
                            for (RemoteStore.SetInfo info : store.listSets(container[0], Long.parseLong(container[1]))) {
                                if (known.add(info.tableId()))
                                    found.add(new SetChoice(info.name(), null, info.tableId(), store.serverSlug(),
                                            "On OMERO (" + container[0] + " \"" + container[2] + "\", table "
                                                    + info.tableId() + ")"));
                            }
                        }
                    } else {
                        omero += " Open an OMERO image to see the training sets of its dataset.";
                    }
                } catch (Exception e) {
                    remote = null;
                    omero = message(e);
                }
            }
            return new Object[]{found, omero};
        }, result -> {
            @SuppressWarnings("unchecked")
            List<SetChoice> found = (List<SetChoice>) result[0];
            choices.getItems().setAll(found);
            omeroStatus.setText((String) result[1]);
        });
    }

    private void openChoice(SetChoice choice) {
        if (choice == null)
            return;
        background("Opening " + choice.name() + "…", () -> {
            TrainingSet opened;
            if (choice.folder() != null) {
                SetConfig config = SetConfig.load(choice.folder().resolve(Bundle.CONFIG));
                opened = TrainingSet.open(choice.folder(), config.isLocal() ? null : connect(config.server()),
                        softwareVersion());
            } else {
                opened = TrainingSet.open(choice.tableId(), connect(choice.server()), home(), softwareVersion());
            }
            return opened;
        }, opened -> {
            set = opened;
            showOverview();
        }, error -> fail("The training set could not be opened", error));
    }

    // =================================================================================
    // New training set
    // =================================================================================

    private void newSet() {
        NewSetDialog.Context context = new NewSetDialog.Context();
        Project<BufferedImage> project = qupath.getProject();
        context.projectImages = project == null ? 0 : project.getImageList().size();
        context.omeroUnavailable = omeroUnavailable();
        if (project != null && project.getName() != null)
            context.suggestedName = project.getName().replaceAll("[/\\\\]?project\\.qpproj$", "")
                    .replaceAll("-project$", "");
        ImageData<BufferedImage> imageData = qupath.getImageData();
        Long openImage = null;
        if (imageData != null) {
            ImageServer<BufferedImage> server = imageData.getServer();
            List<String> names = new ArrayList<>();
            for (ImageChannel channel : server.getMetadata().getChannels())
                names.add(channel.getName());
            if (!names.isEmpty())
                context.channelNames = names;
            context.rgb = server.isRGB();
            context.sizeZ = server.nZSlices();
            context.sizeT = server.nTimepoints();
            context.currentZ = qupath.getViewer().getZPosition();
            context.currentT = qupath.getViewer().getTPosition();
            openImage = OmeroAccess.imageId(server);
        } else {
            List<String> names = new ArrayList<>();
            for (int i = 1; i <= 6; i++)
                names.add("Channel " + i);
            context.channelNames = names;
            context.sizeZ = 999;
            context.sizeT = 999;
        }
        Long imageId = openImage;
        if (context.omeroUnavailable == null && imageId != null) {
            // Suggest the dataset of the open image; looked up before the dialog opens.
            background("Looking up the image on OMERO…", () -> {
                List<String[]> containers = connect().containersOfImage(imageId);
                return containers.isEmpty() ? null : containers.get(0);
            }, container -> {
                context.suggestedContainer = container;
                askNewSet(context);
            }, error -> askNewSet(context));
        } else {
            askNewSet(context);
        }
    }

    private void askNewSet(NewSetDialog.Context context) {
        NewSetDialog.Request request = NewSetDialog.show(context);
        if (request == null)
            return;
        Project<BufferedImage> project = qupath.getProject();
        List<ProjectImageEntry<BufferedImage>> entries = request.fromOmeroContainer() || project == null
                ? List.of() : new ArrayList<>(project.getImageList());
        SetConfig.NewSet spec = request.spec();
        spec.toolVersion = QuPathGUI.getVersion() == null ? null : QuPathGUI.getVersion().toString();
        spec.qupathInfo = new LinkedHashMap<>();
        spec.qupathInfo.put("extension_version", OmeroAnnotateExtension.VERSION);
        background("Making the training set…", () -> {
            List<Planning.ImageRef> images = new ArrayList<>();
            RemoteStore store = spec.local && !request.fromOmeroContainer() ? null : connect();
            if (request.fromOmeroContainer()) {
                for (RemoteStore.RemoteImage image : store.listImages(request.containerType(), request.containerId()))
                    images.add(new Planning.ImageRef(Ids.omeroUri(image.id()), image.id(), image.name(),
                            image.width(), image.height(), image.sizeZ(), image.sizeT()));
                if (images.isEmpty())
                    throw new IllegalStateException("The OMERO " + request.containerType() + " "
                            + request.containerId() + " has no images (or you may not see it).");
            } else {
                int n = 0;
                for (ProjectImageEntry<BufferedImage> entry : entries) {
                    int count = ++n;
                    Platform.runLater(() -> busyText.setText("Reading image " + count + " of " + entries.size() + "…"));
                    images.add(Images.describe(entry));
                }
                project.syncChanges();
            }
            Map<String, String> split = Planning.planSplit(images, spec);
            List<Row> rows = new ArrayList<>();
            List<String> tooLarge = new ArrayList<>();
            for (Map.Entry<String, String> picked : split.entrySet()) {
                for (Planning.ImageRef image : images) {
                    if (!image.uri().equals(picked.getKey()))
                        continue;
                    if (!spec.usePatches && (long) image.width() * image.height() > Units.MAX_UNIT_PIXELS)
                        tooLarge.add(image.name());
                    rows.addAll(Planning.unitsForImage(image, picked.getValue(), spec));
                    break;
                }
            }
            if (!tooLarge.isEmpty())
                throw new IllegalStateException(tooLarge.get(0) + (tooLarge.size() > 1 ? " and " + (tooLarge.size() - 1)
                        + " more are" : " is") + " too large to annotate as a whole image. Choose \"Patches\" instead.");
            if (rows.isEmpty())
                throw new IllegalStateException("None of the images has the chosen z-slice and timepoint.");
            if (!spec.local && !request.fromOmeroContainer()) {
                // Images of the project that live on OMERO: the set is attached to their datasets.
                Set<Long> datasets = new LinkedHashSet<>();
                for (Row row : rows) {
                    if (row.imageId == null)
                        throw new IllegalStateException("\"" + row.imageName + "\" is not an image on OMERO, so "
                                + "this set cannot be kept on OMERO. Choose \"On this computer\".");
                    List<String[]> containers = store.containersOfImage(row.imageId);
                    if (containers.isEmpty())
                        throw new IllegalStateException("\"" + row.imageName + "\" is not in a dataset on OMERO.");
                    datasets.add(Long.parseLong(containers.get(0)[1]));
                }
                spec.containerType = "dataset";
                spec.containerIds = new ArrayList<>(datasets);
            }
            return TrainingSet.create(SetConfig.create(spec), rows, spec.local ? null : store, home(),
                    softwareVersion());
        }, created -> {
            set = created;
            showOverview();
            say("Training set \"" + created.name() + "\" is ready: " + count(created.rows().size(),
                    created.config().usesPatches() ? "patch" : "image") + " to annotate.");
        }, error -> {
            if (error instanceof Planning.SelectionException)
                Dialogs.showErrorMessage("New training set", error.getMessage());
            fail("The training set was not made", error);
        });
    }

    // =================================================================================
    // Face 2: the overview of an open set
    // =================================================================================

    private void showOverview() {
        if (set == null) {
            showChooser();
            return;
        }
        long toDo = set.rows().stream().filter(r -> !r.processed).count();
        long annotated = set.rows().stream().filter(Row::isAnnotated).count();
        long waiting = set.unfinished().size();

        Label where = hint(set.isLocal() ? "Kept on this computer"
                : "Kept on OMERO (" + set.config().server() + ", table " + set.tableId() + ")");
        GridPane progress = new GridPane();
        progress.setHgap(8);
        progress.setVgap(6);
        int line = 0;
        for (Map.Entry<String, TrainingSet.Progress> entry : set.progress().entrySet()) {
            TrainingSet.Progress p = entry.getValue();
            if (p.total() == 0)
                continue;
            String name = entry.getKey().substring(0, 1).toUpperCase() + entry.getKey().substring(1);
            ProgressBar bar = new ProgressBar(p.total() == 0 ? 0 : (double) p.done() / p.total());
            bar.setMaxWidth(Double.MAX_VALUE);
            GridPane.setHgrow(bar, Priority.ALWAYS);
            Label count = new Label(p.done() + " of " + p.total()
                    + (p.excluded() > 0 ? " (" + p.excluded() + " excluded)" : ""));
            progress.addRow(line++, new Label(name), bar, count);
        }

        Button annotate = new Button(toDo == 0 ? "Everything is annotated" : "Annotate (" + toDo + " to do)");
        annotate.setDefaultButton(toDo > 0);
        annotate.setDisable(toDo == 0);
        annotate.setOnAction(e -> startSession(false));
        Button review = new Button("Review (" + annotated + " annotated)");
        review.setDisable(annotated == 0);
        review.setTooltip(new Tooltip("Go through the annotated images again and correct them"));
        review.setOnAction(e -> startSession(true));
        fullWidth(annotate, review);
        Button export = new Button("Export training data…");
        export.setDisable(annotated == 0);
        export.setOnAction(e -> exportSet());
        Button folder = new Button("Show folder");
        folder.setOnAction(e -> GuiTools.browseDirectory(set.bundle().toFile()));
        Button close = new Button("Close");
        close.setOnAction(e -> {
            set = null;
            showChooser();
        });
        fullWidth(export);

        VBox page = page(title(set.name()), where, progress);
        if (waiting > 0) {
            Button upload = new Button("Upload now");
            upload.setOnAction(e -> background("Uploading to OMERO…", () -> set.repair(Units.TRACER), n -> {
                showOverview();
                if (set.unfinished().isEmpty())
                    say(n + (n == 1 ? " image is" : " images are") + " now on OMERO.");
                else
                    warn("Some images could not be uploaded yet. They are safe in the set's folder.");
            }));
            page.getChildren().add(banner(waiting + (waiting == 1 ? " image is" : " images are")
                    + " saved on this computer but not on OMERO yet (an upload was interrupted).",
                    "rgba(230, 150, 0, 0.22)", upload));
        }
        long needsReview = set.rows().stream()
                .filter(r -> r.error != null && r.error.startsWith(TrainingSet.NEEDS_REVIEW)).count();
        if (needsReview > 0)
            page.getChildren().add(banner(needsReview + (needsReview == 1 ? " image has" : " images have")
                    + " two different masks (one here, one on OMERO). They are skipped; resolve them with "
                    + "the Python package (see \"Resume and repeat\" in its documentation).", "rgba(214, 69, 69, 0.18)"));
        page.getChildren().addAll(new HBox(6, annotate, review), new HBox(6, export, folder, close));

        List<Map<String, Object>> sessions = set.config().sessions();
        if (!sessions.isEmpty()) {
            page.getChildren().addAll(new Separator(), heading("Sessions"));
            for (int i = sessions.size() - 1; i >= Math.max(0, sessions.size() - 4); i--) {
                Map<String, Object> s = sessions.get(i);
                String when = String.valueOf(s.get("started"));
                when = when.length() >= 16 ? when.substring(0, 16).replace('T', ' ') : when;
                String text = when + " · " + s.get("kind") + " · " + s.get("n_annotated") + " saved"
                        + (s.get("user") == null ? "" : " · " + s.get("user"))
                        + ("finished".equals(s.get("status")) ? "" : " · " + s.get("status"));
                Label lineLabel = new Label(text);
                lineLabel.setWrapText(true);
                page.getChildren().add(lineLabel);
                if (s.get("note") != null && !s.get("note").toString().isBlank()) {
                    Label noteLabel = hint("“" + s.get("note") + "”");
                    noteLabel.setPadding(new Insets(0, 0, 0, 12));
                    page.getChildren().add(noteLabel);
                }
            }
        }
        scroll.setContent(page);
    }

    private void exportSet() {
        File folder = FileChoosers.promptForDirectory("Folder for the training data", set.bundle().getParent().toFile());
        if (folder == null)
            return;
        Exporter.Layout layout = Dialogs.showChoiceDialog("Export training data", "Folder layout",
                Exporter.Layout.values(), Exporter.Layout.SPLIT_FOLDERS);
        if (layout == null)
            return;
        Project<BufferedImage> project = qupath.getProject();
        String web = webUri();
        TrainingSet exporting = set;
        background("Exporting…", () -> Exporter.export(exporting, folder.toPath(), layout, project, web,
                (done, total) -> Platform.runLater(() -> busyText.setText("Exporting image "
                        + Math.min(done + 1, total) + " of " + total + "…"))), report -> {
            StringBuilder text = new StringBuilder("Written to " + report.folder() + "\n\n");
            report.written().forEach((category, n) -> text.append(n).append(" ").append(category).append('\n'));
            if (!report.skipped().isEmpty()) {
                text.append("\nNot exported:\n");
                report.skipped().forEach(s -> text.append("• ").append(s).append('\n'));
            }
            Dialogs.showPlainMessage("Export training data", text.toString());
            say("Training data written to " + report.folder());
        });
    }

    // =================================================================================
    // Face 3: annotating
    // =================================================================================

    private String setKey() {
        return set.isLocal() ? "local:" + set.config().localUuid()
                : "omero:" + set.config().server() + ":" + set.tableId();
    }

    /** The QuPath project the set's images are opened in: the open one, else the set's own. */
    private Project<BufferedImage> ensureProject() throws Exception {
        Project<BufferedImage> project = qupath.getProject();
        Path own = Bundle.projectDir(set.bundle());
        if (project != null) {
            File base = Projects.getBaseDirectory(project);
            boolean otherSet = base != null && base.toPath().startsWith(home()) && !base.toPath().equals(own);
            if (!otherSet)
                return project;
        }
        File file = own.resolve(ProjectIO.DEFAULT_PROJECT_NAME + "." + ProjectIO.getProjectExtension()).toFile();
        if (file.exists()) {
            project = ProjectIO.loadProject(file, BufferedImage.class);
        } else {
            Files.createDirectories(own);
            project = Projects.createProject(own.toFile(), BufferedImage.class);
            project.syncChanges();
        }
        qupath.setProject(project);
        return project;
    }

    private void startSession(boolean review) {
        try {
            ensureProject();
        } catch (Exception e) {
            fail("The QuPath project of this set could not be opened", e);
            return;
        }
        TrainingSet target = set;
        background("Starting…", () -> {
            AnnotationSession started = new AnnotationSession(target, review, "QuPath", Units.TRACER);
            Row first = started.next();
            return new Object[]{started, first};
        }, result -> {
            session = (AnnotationSession) result[0];
            handled.clear();
            history.clear();
            position = -1;
            current = null;
            noMoreUnits = false;
            note.clear();
            buildSessionView();
            qupath.getStage().getScene().getAccelerators().put(ACCEPT_KEY, () -> {
                if (!isBusy && session != null && !accept.isDisabled())
                    acceptCurrent();
            });
            say("");
            Row first = (Row) result[1];
            if (first == null) {
                noMoreUnits = true;
                updateSessionView();
                say(review ? "There is nothing to review." : "There is nothing left to annotate: other "
                        + "computers hold the remaining images.");
            } else {
                history.add(first);
                position = 0;
                showUnit(first);
            }
        }, error -> fail("The session could not be started", error));
    }

    private void buildSessionView() {
        unitTitle.setStyle("-fx-font-size: 1.15em; -fx-font-weight: bold;");
        unitTitle.setWrapText(true);
        unitDetail.setWrapText(true);
        unitDetail.setStyle("-fx-opacity: 0.75;");
        unitState.setWrapText(true);
        objectCount.setStyle("-fx-font-weight: bold;");
        sessionProgress.setMaxWidth(Double.MAX_VALUE);

        accept.setDefaultButton(true);
        accept.setTooltip(new Tooltip("Save every object inside the frame as this image's mask and go on ("
                + ACCEPT_KEY.getDisplayText() + ")"));
        accept.setOnAction(e -> acceptCurrent());
        skip.setTooltip(new Tooltip("Go on without saving: an open image is left for a later session, "
                + "a saved one stays as it was"));
        skip.setOnAction(e -> skipCurrent());
        exclude.setTooltip(new Tooltip("Leave this image out of the training set for good (out of focus, artefact)"));
        exclude.setOnAction(e -> excludeCurrent());
        previous.setTooltip(new Tooltip("Go back to the image before this one"));
        previous.setOnAction(e -> goTo(position - 1));
        fullWidth(accept, skip, exclude, previous);

        MenuItem cellDetection = new MenuItem("QuPath cell detection…");
        cellDetection.setDisable(!Detectors.hasCellDetection());
        cellDetection.setOnAction(e -> {
            if (selectFrame())
                Detectors.runCellDetection(qupath);
        });
        MenuItem cellpose = new MenuItem(Detectors.hasCellpose() ? "Cellpose…" : "Cellpose (extension not installed)");
        cellpose.setDisable(!Detectors.hasCellpose());
        cellpose.setOnAction(e -> runCellpose());
        MenuItem others = new MenuItem("InstanSeg, StarDist, SAM, brush…");
        others.setOnAction(e -> Dialogs.showPlainMessage("Other tools",
                "Any tool works: the frame is selected when an image opens, so run the tool on it "
                        + "(InstanSeg, StarDist, your own script), prompt SAM inside it, or draw with the brush "
                        + "and wand.\n\nEvery object whose centre lies inside the frame becomes one object of "
                        + "the mask, whether it is an annotation, a detection or a cell. Objects with a class "
                        + "ending in * (such as Ignore*) are left out."));
        MenuButton detect = new MenuButton("Detect objects", null, cellDetection, cellpose, new SeparatorMenuItem(), others);
        detect.setMaxWidth(Double.MAX_VALUE);
        Button selectFrame = new Button("Show frame");
        selectFrame.setTooltip(new Tooltip("Go to the frame and select it"));
        selectFrame.setOnAction(e -> {
            if (selectFrame())
                zoomToFrame();
        });
        Button clear = new Button("Clear frame");
        clear.setTooltip(new Tooltip("Delete every object inside the frame"));
        clear.setOnAction(e -> clearFrame());
        HBox.setHgrow(detect, Priority.ALWAYS);

        note.setPromptText("e.g. nuclei only, skipped debris");
        Button finish = new Button("Finish session");
        finish.setMaxWidth(Double.MAX_VALUE);
        finish.setOnAction(e -> finishSession(false));

        Region gap = new Region();
        gap.setMinHeight(4);
        VBox page = page(
                title(set.name() + (session.isReview() ? " · review" : "")),
                sessionProgress, sessionProgressText,
                new Separator(),
                unitTitle, unitDetail, unitState,
                hint("Mark every object inside the frame. Draw by hand or let a tool detect them, "
                        + "then correct what is wrong."),
                new HBox(6, detect, selectFrame, clear),
                objectCount,
                new HBox(6, accept),
                new HBox(6, previous, skip, exclude),
                gap, new Separator(),
                heading("When you stop"),
                new Label("A note about this session (optional)"), note, finish);
        scroll.setContent(page);
    }

    private void updateSessionView() {
        if (session == null)
            return;
        int saved = session.isReview() ? handled.size() : session.savedCount();
        int left = session.remaining() + (current != null && !isDone(current) ? 1 : 0);
        int pendingInHistory = 0;
        for (int i = position + 1; i < history.size(); i++)
            if (!isDone(history.get(i)))
                pendingInHistory++;
        left += pendingInHistory;
        int total = Math.max(1, saved + left);
        sessionProgress.setProgress((double) saved / total);
        int elsewhere = session.heldElsewhere();
        sessionProgressText.setText(saved + (session.isReview() ? " reviewed · " : " saved in this session · ")
                + left + " to go"
                + (elsewhere > 0 ? " · " + elsewhere + " being annotated on another computer" : ""));
        boolean has = current != null;
        accept.setDisable(!has);
        skip.setDisable(!has);
        exclude.setDisable(!has || session.isReview() || current.isAnnotated() || current.isExcluded());
        previous.setDisable(position <= 0);
        if (!has) {
            unitTitle.setText(noMoreUnits ? "All done" : "");
            unitDetail.setText(noMoreUnits ? "There are no more images in this session. Add a note if you like "
                    + "and finish the session." : "");
            unitState.setText("");
            unitState.setManaged(false);
            objectCount.setText("");
            return;
        }
        unitTitle.setText(current.imageName);
        StringBuilder detail = new StringBuilder(current.category);
        if (current.isPatch)
            detail.append(" · patch ").append(current.patchWidth).append(" × ").append(current.patchHeight)
                    .append(" at (").append(current.patchX).append(", ").append(current.patchY).append(")");
        if (current.zSlice > 0)
            detail.append(" · z ").append(current.zSlice + 1);
        if (current.timepoint > 0)
            detail.append(" · t ").append(current.timepoint + 1);
        unitDetail.setText(detail.toString());
        boolean waiting = !current.processed && current.error != null
                && Files.exists(Bundle.maskPath(set.bundle(), current.annotationId));
        unitState.setManaged(current.processed || waiting);
        unitState.setVisible(current.processed || waiting);
        if (waiting) {
            unitState.setText("Saved on this computer, not on OMERO yet. Accept again to retry the upload.");
            unitState.setStyle("-fx-text-fill: #c77700;");
        } else if (current.isExcluded()) {
            unitState.setText("Excluded. Accept it to include it after all.");
            unitState.setStyle("-fx-text-fill: #c77700;");
        } else if (current.isAnnotated()) {
            unitState.setText("Saved. Change the objects and accept again to correct it.");
            unitState.setStyle("-fx-text-fill: #2e9e5b;");
        } else {
            unitState.setText("");
            unitState.setStyle("");
        }
        accept.setText(current.isAnnotated() ? "Save changes and next" : "Accept and next");
        updateObjectCount();
    }

    /** Whether a unit needs nothing more in this session. */
    private boolean isDone(Row row) {
        return session != null && session.isReview() ? handled.contains(row.annotationId) : row.processed;
    }

    private ImageData<BufferedImage> imageDataOfCurrent() {
        ImageData<BufferedImage> imageData = qupath.getViewer().getImageData();
        Project<BufferedImage> project = qupath.getProject();
        if (imageData == null || project == null || current == null)
            return null;
        ProjectImageEntry<BufferedImage> entry = project.getEntry(imageData);
        if (entry == null || !current.imageUri.equals(entry.getMetadataValue(Images.ENTRY_URI)))
            return null;
        return imageData;
    }

    private PathObject currentFrame(ImageData<BufferedImage> imageData) {
        return Units.ensureFrame(imageData.getHierarchy(), setKey(), current, imageData.getServer());
    }

    private void updateObjectCount() {
        ImageData<BufferedImage> imageData = imageDataOfCurrent();
        if (session == null || imageData == null) {
            objectCount.setText("");
            return;
        }
        PathObject frame = Units.findFrame(imageData.getHierarchy(), setKey(), current);
        if (frame == null) {
            objectCount.setText("");
            return;
        }
        int n = Units.objectsInFrame(imageData.getHierarchy(), frame).size();
        objectCount.setText(n == 0 ? "No objects in the frame yet" : n + (n == 1 ? " object" : " objects") + " in the frame");
    }

    private void watch(PathObjectHierarchy hierarchy) {
        if (watched == hierarchy)
            return;
        if (watched != null)
            watched.removeListener(objectListener);
        watched = hierarchy;
        if (watched != null)
            watched.addListener(objectListener);
    }

    /** Open the unit's image, put its frame in place and bring it into view. */
    private void showUnit(Row row) {
        current = row;
        confirmEmpty = false;
        updateSessionView();
        Project<BufferedImage> project = qupath.getProject();
        String web = webUri();
        background("Opening " + row.imageName + "…", () -> Images.findOrAdd(project, row, web), entry -> {
            qupath.refreshProject();
            qupath.openImageEntry(entry);
            ImageData<BufferedImage> imageData = qupath.getViewer().getImageData();
            if (imageData == null || project.getEntry(imageData) != entry) {
                warn("The image was not opened. Press \"Show frame\" to try again.");
                return;
            }
            ImageServer<BufferedImage> server = imageData.getServer();
            if (row.isPatch && (row.patchX + row.patchWidth > server.getWidth()
                    || row.patchY + row.patchHeight > server.getHeight()))
                warn("This image is smaller than the training set expects; the frame was cut to fit.");
            PathObject frame = currentFrame(imageData);
            frame.setName(frameName(row));
            watch(imageData.getHierarchy());
            QuPathViewer viewer = qupath.getViewer();
            viewer.setZPosition(Math.max(0, Math.min(row.zSlice, server.nZSlices() - 1)));
            viewer.setTPosition(Math.max(0, Math.min(row.timepoint, server.nTimepoints() - 1)));
            imageData.getHierarchy().getSelectionModel().setSelectedObject(frame);
            zoomToFrame();
            if (row.isAnnotated() && !Objects.equals(frame.getMetadata().get(Units.META_DIGEST), row.maskDigest))
                loadSavedObjects(row, imageData, frame);
            else
                updateSessionView();
        }, error -> {
            fail("The image could not be opened", error);
            updateSessionView();
        });
    }

    private static String frameName(Row row) {
        if (row.isExcluded())
            return "Excluded (" + row.category + ")";
        return (row.isAnnotated() ? "Annotated (" : "Annotate here (") + row.category + ")";
    }

    /** Replace the objects in the frame by those of the unit's saved mask. */
    private void loadSavedObjects(Row row, ImageData<BufferedImage> imageData, PathObject frame) {
        TrainingSet target = set;
        background("Loading the saved objects…", () -> target.savedMask(row), mask -> {
            if (mask == null) {
                warn("The saved mask of this image was not found.");
                return;
            }
            PathObjectHierarchy hierarchy = imageData.getHierarchy();
            int[] r = Units.region(row, imageData.getServer());
            List<PathObject> old = Units.objectsInFrame(hierarchy, frame);
            if (!old.isEmpty())
                hierarchy.removeObjects(old, true);
            List<PathObject> objects = Units.objectsFromMask(mask, r[0], r[1], Units.plane(row));
            hierarchy.addObjects(objects);
            frame.getMetadata().put(Units.META_DIGEST, row.maskDigest);
            hierarchy.getSelectionModel().setSelectedObject(frame);
            updateSessionView();
        });
    }

    private boolean selectFrame() {
        ImageData<BufferedImage> imageData = imageDataOfCurrent();
        if (imageData == null) {
            if (current != null)
                showUnit(current);
            return false;
        }
        imageData.getHierarchy().getSelectionModel().setSelectedObject(currentFrame(imageData));
        return true;
    }

    private void zoomToFrame() {
        ImageData<BufferedImage> imageData = imageDataOfCurrent();
        if (imageData == null)
            return;
        int[] r = Units.region(current, imageData.getServer());
        QuPathViewer viewer = qupath.getViewer();
        double viewWidth = Math.max(100, viewer.getView().getWidth());
        double viewHeight = Math.max(100, viewer.getView().getHeight());
        double downsample = Math.max(r[2] / viewWidth, r[3] / viewHeight) * 1.15;
        viewer.setDownsampleFactor(Math.max(downsample, 0.05));
        viewer.setCenterPixelLocation(r[0] + r[2] / 2.0, r[1] + r[3] / 2.0);
    }

    private void clearFrame() {
        ImageData<BufferedImage> imageData = imageDataOfCurrent();
        if (imageData == null)
            return;
        PathObject frame = currentFrame(imageData);
        List<PathObject> objects = Units.objectsInFrame(imageData.getHierarchy(), frame);
        if (objects.isEmpty())
            return;
        if (!Dialogs.showConfirmDialog("Clear frame", "Delete the " + objects.size()
                + (objects.size() == 1 ? " object" : " objects") + " inside the frame?"))
            return;
        imageData.getHierarchy().removeObjects(objects, true);
        imageData.getHierarchy().getSelectionModel().setSelectedObject(frame);
    }

    private void runCellpose() {
        ImageData<BufferedImage> imageData = imageDataOfCurrent();
        if (imageData == null || !selectFrame())
            return;
        boolean rgb = imageData.getServer().isRGB();
        Detectors.CellposeSettings settings = Detectors.askCellpose(Math.max(0, current.channel), rgb,
                lastCellposeModel.get(), 0);
        if (settings == null)
            return;
        lastCellposeModel.set(settings.model());
        String script = Detectors.cellposeScript(settings, rgb);
        background("Cellpose is running… (the first run can take a while)", () -> {
            Detectors.runScript(qupath, imageData, script);
            return null;
        }, done -> {
            selectFrame();
            updateObjectCount();
            say("Cellpose is done. Check the objects and correct what is wrong.");
        }, error -> {
            fail("Cellpose did not run", error);
            Dialogs.showErrorMessage("Cellpose", "Cellpose did not run: " + message(error)
                    + "\n\nCheck the Cellpose settings under Edit ▸ Preferences (the Python environment), "
                    + "or run your own Cellpose script on the frame.");
        });
    }

    private void acceptCurrent() {
        if (session == null || current == null || isBusy)
            return;
        ImageData<BufferedImage> imageData = imageDataOfCurrent();
        if (imageData == null) {
            warn("Another image is open. Going back to the training image…");
            showUnit(current);
            return;
        }
        Row row = current;
        PathObject frame = currentFrame(imageData);
        List<PathObject> objects = Units.objectsInFrame(imageData.getHierarchy(), frame);
        if (objects.isEmpty() && !confirmEmpty) {
            confirmEmpty = true;
            warn("There are no objects in the frame. Press Accept again to save this image as empty "
                    + "(nothing to segment), or Skip to leave it for later.");
            return;
        }
        ImageServer<BufferedImage> server = imageData.getServer();
        int[] r = Units.region(row, server);
        boolean needInput = !Files.exists(Bundle.inputPath(set.bundle(), row.annotationId)) || row.inputDigest == null;
        AnnotationSession running = session;
        background("Saving " + row.imageName + "…", () -> {
            PixelArray mask = Units.rasterize(objects, r[0], r[1], r[2], r[3]);
            PixelArray input = needInput ? Units.readInput(server, row) : null;
            return new Object[]{running.commit(row, mask, input), mask};
        }, result -> {
            AnnotationSession.Result outcome = (AnnotationSession.Result) result[0];
            PixelArray mask = (PixelArray) result[1];
            confirmEmpty = false;
            handled.add(row.annotationId);
            switch (outcome.outcome()) {
                case SAVED, REVISED, UNCHANGED -> {
                    frame.getMetadata().put(Units.META_DIGEST, mask.digest());
                    frame.setName(frameName(row));
                    saveImage(imageData);
                    int n = Units.countLabels(mask);
                    say(switch (outcome.outcome()) {
                        case UNCHANGED -> row.imageName + ": nothing changed.";
                        case REVISED -> row.imageName + ": correction saved (" + n + (n == 1 ? " object)." : " objects).");
                        default -> row.imageName + ": saved with " + n + (n == 1 ? " object" : " objects")
                                + (set.isLocal() ? "." : " and uploaded to OMERO.");
                    });
                    advance();
                }
                case SAVED_ELSEWHERE -> {
                    saveImage(imageData);
                    Dialogs.showPlainMessage("Saved on another computer", outcome.message());
                    advance();
                }
                case UPLOAD_FAILED -> {
                    saveImage(imageData);
                    Dialogs.showWarningNotification("Not on OMERO yet", "Your work on " + row.imageName
                            + " is safe on this computer, but OMERO could not be updated. Upload it later from "
                            + "the set's overview.");
                    frame.setName("Saved here, not on OMERO yet (" + row.category + ")");
                    warn("Your work on " + row.imageName + " is saved on this computer, but OMERO could not be "
                            + "updated (" + plain(outcome.message()) + "). Press Skip to go on, or finish the session; "
                            + "the upload can be repeated from the set's overview.");
                    updateSessionView();
                }
                default -> advance();
            }
        }, error -> fail("The image could not be saved", error));
    }

    private void saveImage(ImageData<BufferedImage> imageData) {
        try {
            Project<BufferedImage> project = qupath.getProject();
            ProjectImageEntry<BufferedImage> entry = project == null ? null : project.getEntry(imageData);
            if (entry != null) {
                entry.saveImageData(imageData);
                imageData.setChanged(false);
                project.syncChanges();
            }
        } catch (Exception e) {
            logger.warn("The QuPath objects of this image could not be saved in the project", e);
        }
    }

    private void skipCurrent() {
        if (session == null || current == null)
            return;
        Row row = current;
        AnnotationSession running = session;
        ImageData<BufferedImage> imageData = imageDataOfCurrent();
        if (imageData != null)
            saveImage(imageData);
        if (row.processed) { // a saved or excluded image: nothing to hand back, just go on
            handled.add(row.annotationId);
            say(row.imageName + " is left as it was.");
            advance();
            return;
        }
        background("Skipping…", () -> {
            running.skip(row);
            return null;
        }, done -> {
            handled.add(row.annotationId);
            say(running.isReview() ? row.imageName + " is left as it was."
                    : row.imageName + " is left for a later session.");
            advance();
        });
    }

    private void excludeCurrent() {
        if (session == null || current == null)
            return;
        if (!Dialogs.showConfirmDialog("Exclude image", "Leave \"" + current.imageName + "\" out of the training "
                + "set for good?\n\nUse this for images that cannot be annotated (out of focus, artefacts). "
                + "It counts as done and is not exported."))
            return;
        Row row = current;
        AnnotationSession running = session;
        ImageData<BufferedImage> imageData = imageDataOfCurrent();
        background("Excluding…", () -> running.exclude(row), outcome -> {
            if (imageData != null) {
                currentFrame(imageData).setName(frameName(row));
                saveImage(imageData);
            }
            if (outcome.outcome() == AnnotationSession.Outcome.EXCLUDED)
                say(row.imageName + " is excluded.");
            else
                warn(outcome.message());
            advance();
        });
    }

    /** Go on: to the next image already seen in this session, else to a new one. */
    private void advance() {
        if (position < history.size() - 1) {
            goTo(position + 1);
            return;
        }
        AnnotationSession running = session;
        background("Looking for the next image…", running::next, next -> {
            if (next == null) {
                current = null;
                noMoreUnits = true;
                position = history.size();
                updateSessionView();
                return;
            }
            history.add(next);
            position = history.size() - 1;
            showUnit(next);
        });
    }

    private void goTo(int index) {
        if (index < 0 || index >= history.size())
            return;
        ImageData<BufferedImage> imageData = imageDataOfCurrent();
        if (imageData != null)
            saveImage(imageData);
        position = index;
        noMoreUnits = false;
        showUnit(history.get(index));
    }

    /** End the session and go back to the overview. */
    private void finishSession(boolean interrupted) {
        if (session == null)
            return;
        AnnotationSession running = session;
        String text = note.getText();
        ImageData<BufferedImage> imageData = imageDataOfCurrent();
        if (imageData != null)
            saveImage(imageData);
        background("Finishing the session…", () -> {
            running.finish(text, interrupted);
            return null;
        }, done -> {
            int saved = running.savedCount();
            endSessionUi();
            showOverview();
            say("Session finished: " + saved + (saved == 1 ? " image" : " images") + " saved.");
        }, error -> {
            endSessionUi();
            showOverview();
            fail("The session was closed, but its record could not be written", error);
        });
    }

    private void endSessionUi() {
        qupath.getStage().getScene().getAccelerators().remove(ACCEPT_KEY);
        watch(null);
        session = null;
        current = null;
        history.clear();
        position = -1;
    }

    /**
     * Called when QuPath closes: a running session is recorded as interrupted and its
     * images are handed back, so they stay available for the next session.
     */
    public void shutdown() {
        AnnotationSession running = session;
        if (running != null && !running.isFinished()) {
            try {
                worker.submit(() -> {
                    running.finish(note.getText(), true);
                    return null;
                }).get(15, TimeUnit.SECONDS);
            } catch (Exception e) {
                logger.warn("The running session could not be closed cleanly: {}", e.toString());
            }
        }
        RemoteStore store = remote;
        if (store != null) {
            try {
                worker.submit(store::close).get(5, TimeUnit.SECONDS);
            } catch (Exception e) {
                logger.debug("OMERO connection not closed", e);
            }
        }
        worker.shutdownNow();
    }
}
