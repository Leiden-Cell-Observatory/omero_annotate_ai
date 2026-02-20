"""Main dock widget for the OMERO Annotate AI napari plugin."""

from pathlib import Path
from typing import Optional

import napari
from qtpy.QtCore import Qt, QThreadPool
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from omero_annotate_ai import create_default_config, load_config
from omero_annotate_ai.omero.omero_functions import list_annotation_tables
from omero_annotate_ai.omero.simple_connection import SimpleOMEROConnection

from ._worker import AnnotationWorker

try:
    from omero.gateway import BlitzGateway
except ImportError:
    BlitzGateway = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Helper: get the BlitzGateway from the omero-napari plugin (if installed)
# ---------------------------------------------------------------------------

def _get_omero_napari_conn():
    """Return a live BlitzGateway from the omero-napari plugin, or None."""
    try:
        from omero_napari.gateway import get_gateway  # type: ignore[import]
        conn = get_gateway()
        if conn is not None and conn.isConnected():
            return conn
    except (ImportError, AttributeError):
        pass
    try:
        viewer = napari.current_viewer()
        if viewer is not None:
            conn = getattr(viewer, "_omero_conn", None)
            if conn is not None and conn.isConnected():
                return conn
    except Exception:  # noqa: BLE001
        pass
    return None


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class OMEROAnnotateWidget(QWidget):
    """Single dock widget with three tabs: Connection, Configure, Run."""

    def __init__(self, napari_viewer: napari.Viewer) -> None:
        super().__init__()
        self._viewer = napari_viewer
        self._conn: Optional["BlitzGateway"] = None
        self._config = create_default_config()
        self._thread_pool = QThreadPool.globalInstance()

        self._build_ui()
        # Try to pick up an existing connection straight away
        self._refresh_connection()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        # Title
        title = QLabel("<b>OMERO Annotate AI</b>")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(title)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_connection_tab(), "Connection")
        self._tabs.addTab(self._build_configure_tab(), "Configure")
        self._tabs.addTab(self._build_run_tab(), "Run")

        root.addWidget(self._tabs)

        # Status bar
        self._status = QLabel("")
        self._status.setWordWrap(True)
        root.addWidget(self._status)

    # ---- Tab 0: Connection -------------------------------------------

    def _build_connection_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Status indicator
        self._conn_status = QLabel("<i>Not connected</i>")
        self._conn_status.setWordWrap(True)
        layout.addWidget(self._conn_status)

        # Built-in login form
        login_box = QGroupBox("Connect to OMERO")
        login_form = QFormLayout(login_box)

        self._login_host = QLineEdit()
        self._login_host.setPlaceholderText("e.g. omero.example.org")
        login_form.addRow("Host:", self._login_host)

        self._login_port = QLineEdit("4064")
        login_form.addRow("Port:", self._login_port)

        self._login_user = QLineEdit()
        self._login_user.setPlaceholderText("username")
        login_form.addRow("Username:", self._login_user)

        self._login_password = QLineEdit()
        self._login_password.setEchoMode(QLineEdit.EchoMode.Password)
        self._login_password.setPlaceholderText("password")
        login_form.addRow("Password:", self._login_password)

        connect_btn = QPushButton("Connect")
        connect_btn.clicked.connect(self._connect_direct)
        disconnect_btn = QPushButton("Disconnect")
        disconnect_btn.clicked.connect(self._disconnect)
        btn_row = QHBoxLayout()
        btn_row.addWidget(connect_btn)
        btn_row.addWidget(disconnect_btn)
        login_form.addRow("", btn_row)

        layout.addWidget(login_box)

        # omero-napari fallback
        ext_box = QGroupBox("Or use omero-napari connection")
        ext_layout = QVBoxLayout(ext_box)
        ext_info = QLabel("If the omero-napari plugin is installed and connected, click Refresh to use that session.")
        ext_info.setWordWrap(True)
        ext_layout.addWidget(ext_info)
        refresh_btn = QPushButton("Refresh from omero-napari")
        refresh_btn.clicked.connect(self._refresh_connection)
        ext_layout.addWidget(refresh_btn)
        layout.addWidget(ext_box)

        layout.addStretch()
        return tab

    # ---- Tab 1: Configure --------------------------------------------

    def _build_configure_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # --- Load from YAML ---
        yaml_box = QGroupBox("Load from YAML config")
        yaml_layout = QHBoxLayout(yaml_box)
        self._yaml_path = QLineEdit()
        self._yaml_path.setPlaceholderText("path/to/annotation_config.yaml")
        self._yaml_path.setReadOnly(True)
        load_btn = QPushButton("Browse…")
        load_btn.clicked.connect(self._load_yaml)
        yaml_layout.addWidget(self._yaml_path)
        yaml_layout.addWidget(load_btn)
        layout.addWidget(yaml_box)

        # --- OMERO container ---
        omero_box = QGroupBox("OMERO Container")
        omero_form = QFormLayout(omero_box)

        self._name_edit = QLineEdit("my_annotation")
        omero_form.addRow("Annotation name:", self._name_edit)

        self._container_type = QComboBox()
        self._container_type.addItems(["dataset", "plate", "project", "screen"])
        omero_form.addRow("Container type:", self._container_type)

        self._container_id = QLineEdit()
        self._container_id.setPlaceholderText("e.g. 123")
        omero_form.addRow("Container ID:", self._container_id)

        out_row = QWidget()
        out_layout = QHBoxLayout(out_row)
        out_layout.setContentsMargins(0, 0, 0, 0)
        self._output_dir = QLineEdit()
        self._output_dir.setPlaceholderText("output directory")
        browse_out_btn = QPushButton("Browse…")
        browse_out_btn.clicked.connect(self._browse_output_dir)
        out_layout.addWidget(self._output_dir)
        out_layout.addWidget(browse_out_btn)
        omero_form.addRow("Output dir:", out_row)

        layout.addWidget(omero_box)

        # --- Inner tabs: Annotation / Technical / Workflow ---
        inner_tabs = QTabWidget()

        # Annotation settings tab
        ann_scroll = QScrollArea()
        ann_scroll.setWidgetResizable(True)
        ann_inner = QWidget()
        ann_form = QFormLayout(ann_inner)

        self._segment_all = QCheckBox("Annotate all images (no train/val split)")
        self._segment_all.stateChanged.connect(self._on_segment_all_changed)
        ann_form.addRow("", self._segment_all)

        self._train_n = QSpinBox()
        self._train_n.setRange(0, 200)
        self._train_n.setValue(3)
        ann_form.addRow("Training images:", self._train_n)

        self._validate_n = QSpinBox()
        self._validate_n.setRange(0, 200)
        self._validate_n.setValue(2)
        ann_form.addRow("Validation images:", self._validate_n)

        self._test_n = QSpinBox()
        self._test_n.setRange(0, 200)
        self._test_n.setValue(0)
        ann_form.addRow("Test images:", self._test_n)

        self._channel = QSpinBox()
        self._channel.setRange(0, 15)
        ann_form.addRow("Channel:", self._channel)

        self._timepoint_mode = QComboBox()
        self._timepoint_mode.addItems(["specific", "all", "random"])
        self._timepoint_mode.currentTextChanged.connect(self._on_timepoint_mode_changed)
        ann_form.addRow("Timepoint mode:", self._timepoint_mode)

        self._timepoints_edit = QLineEdit("0")
        self._timepoints_edit.setPlaceholderText("e.g. 0,1,2")
        ann_form.addRow("Timepoints:", self._timepoints_edit)

        self._z_slice_mode = QComboBox()
        self._z_slice_mode.addItems(["specific", "all", "random"])
        self._z_slice_mode.currentTextChanged.connect(self._on_z_slice_mode_changed)
        ann_form.addRow("Z-slice mode:", self._z_slice_mode)

        self._z_slices_edit = QLineEdit("0")
        self._z_slices_edit.setPlaceholderText("e.g. 0,1,2")
        ann_form.addRow("Z-slices:", self._z_slices_edit)

        self._three_d = QCheckBox("3D volumetric processing")
        self._three_d.stateChanged.connect(self._on_three_d_changed)
        ann_form.addRow("", self._three_d)

        self._z_range_start = QSpinBox()
        self._z_range_start.setRange(0, 9999)
        self._z_range_start.setEnabled(False)
        ann_form.addRow("Z-range start:", self._z_range_start)

        self._z_range_end = QSpinBox()
        self._z_range_end.setRange(0, 9999)
        self._z_range_end.setValue(10)
        self._z_range_end.setEnabled(False)
        ann_form.addRow("Z-range end:", self._z_range_end)

        self._use_patches = QCheckBox("Use patches")
        self._use_patches.stateChanged.connect(self._on_use_patches_changed)
        ann_form.addRow("", self._use_patches)

        self._patches_per_image = QSpinBox()
        self._patches_per_image.setRange(1, 50)
        self._patches_per_image.setValue(1)
        self._patches_per_image.setEnabled(False)
        ann_form.addRow("Patches per image:", self._patches_per_image)

        self._patch_size_edit = QLineEdit("512,512")
        self._patch_size_edit.setEnabled(False)
        ann_form.addRow("Patch size (h,w):", self._patch_size_edit)

        self._random_patches = QCheckBox("Random patch placement")
        self._random_patches.setChecked(True)
        self._random_patches.setEnabled(False)
        ann_form.addRow("", self._random_patches)

        ann_scroll.setWidget(ann_inner)
        inner_tabs.addTab(ann_scroll, "Annotation")

        # Technical settings tab
        tech_widget = QWidget()
        tech_form = QFormLayout(tech_widget)

        self._model_type = QComboBox()
        self._model_type.addItems(["vit_b_lm", "vit_b", "vit_l", "vit_h"])
        tech_form.addRow("SAM model:", self._model_type)

        self._batch_size = QSpinBox()
        self._batch_size.setRange(0, 20)
        self._batch_size.setValue(0)
        self._batch_size.setToolTip("0 = process all images in one batch")
        tech_form.addRow("Batch size:", self._batch_size)

        inner_tabs.addTab(tech_widget, "Technical")

        # Workflow settings tab
        wf_widget = QWidget()
        wf_form = QFormLayout(wf_widget)

        self._read_only = QCheckBox("Read-only mode (save locally, skip OMERO upload)")
        wf_form.addRow("", self._read_only)

        inner_tabs.addTab(wf_widget, "Workflow")

        layout.addWidget(inner_tabs)

        # --- Existing tables / resume ---
        tables_box = QGroupBox("Existing annotation tables")
        tables_layout = QVBoxLayout(tables_box)
        list_tables_btn = QPushButton("List tables for this container")
        list_tables_btn.clicked.connect(self._list_tables)
        tables_layout.addWidget(list_tables_btn)

        self._tables_display = QPlainTextEdit()
        self._tables_display.setReadOnly(True)
        self._tables_display.setMaximumHeight(70)
        self._tables_display.setPlaceholderText("Click 'List tables'…")
        tables_layout.addWidget(self._tables_display)

        tables_layout.addWidget(QLabel("Resume from table ID:"))
        self._resume_id = QLineEdit()
        self._resume_id.setPlaceholderText("Table ID (optional)")
        tables_layout.addWidget(self._resume_id)

        layout.addWidget(tables_box)
        layout.addStretch()

        scroll.setWidget(inner)
        return scroll

    # ---- Configure tab reactive helpers ---------------------------------

    def _on_segment_all_changed(self, state: int) -> None:
        enabled = state == 0
        self._train_n.setEnabled(enabled)
        self._validate_n.setEnabled(enabled)
        self._test_n.setEnabled(enabled)

    def _on_timepoint_mode_changed(self, mode: str) -> None:
        self._timepoints_edit.setEnabled(mode == "specific")

    def _on_z_slice_mode_changed(self, mode: str) -> None:
        self._z_slices_edit.setEnabled(mode == "specific")

    def _on_three_d_changed(self, state: int) -> None:
        enabled = state != 0
        self._z_range_start.setEnabled(enabled)
        self._z_range_end.setEnabled(enabled)

    def _on_use_patches_changed(self, state: int) -> None:
        enabled = state != 0
        self._patches_per_image.setEnabled(enabled)
        self._patch_size_edit.setEnabled(enabled)
        self._random_patches.setEnabled(enabled)

    # ---- Tab 2: Run --------------------------------------------------

    def _build_run_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Config summary
        summary_box = QGroupBox("Config summary")
        summary_layout = QVBoxLayout(summary_box)
        self._config_summary = QTextEdit()
        self._config_summary.setReadOnly(True)
        self._config_summary.setMaximumHeight(120)
        summary_layout.addWidget(self._config_summary)

        refresh_summary_btn = QPushButton("Refresh summary")
        refresh_summary_btn.clicked.connect(self._refresh_summary)
        summary_layout.addWidget(refresh_summary_btn)
        layout.addWidget(summary_box)

        # Action buttons
        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("Initialize & Run Annotation")
        self._run_btn.setStyleSheet("font-weight: bold;")
        self._run_btn.clicked.connect(self._run_pipeline)
        btn_row.addWidget(self._run_btn)

        save_btn = QPushButton("Save config YAML")
        save_btn.clicked.connect(self._save_yaml)
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)

        # Progress
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)  # indeterminate by default
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        # Log
        log_label = QLabel("Log:")
        layout.addWidget(log_label)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        layout.addWidget(self._log)

        return tab

    # ------------------------------------------------------------------
    # Slot implementations
    # ------------------------------------------------------------------

    def _connect_direct(self) -> None:
        """Connect using the built-in login form via SimpleOMEROConnection."""
        host = self._login_host.text().strip()
        user = self._login_user.text().strip()
        password = self._login_password.text()
        port_txt = self._login_port.text().strip()

        if not host or not user or not password:
            self._set_status("Please fill in host, username, and password.")
            return

        port = int(port_txt) if port_txt.isdigit() else 4064
        self._conn_status.setText("<i>Connecting…</i>")
        self._set_status("Connecting…")

        try:
            mgr = SimpleOMEROConnection()
            conn = mgr.connect(host=host, username=user, password=password, port=port)
            if conn and conn.isConnected():
                self._conn = conn
                self._update_conn_status()
            else:
                self._conn_status.setText(
                    "<span style='color:red'>Connection failed. Check credentials.</span>"
                )
                self._set_status("Connection failed.")
        except Exception as exc:  # noqa: BLE001
            self._conn_status.setText(
                f"<span style='color:red'>Error: {exc}</span>"
            )
            self._set_status(f"Connection error: {exc}")

    def _disconnect(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass
            self._conn = None
        self._conn_status.setText("<i>Not connected</i>")
        self._set_status("Disconnected.")

    def _refresh_connection(self) -> None:
        """Pick up an existing connection from the omero-napari plugin."""
        conn = _get_omero_napari_conn()
        if conn is not None:
            self._conn = conn
            self._update_conn_status()
        else:
            self._conn_status.setText(
                "<span style='color:orange'>omero-napari not connected.</span>"
            )
            self._set_status("omero-napari not connected.")

    def _update_conn_status(self) -> None:
        """Refresh the status label from the current connection."""
        if self._conn is None:
            self._conn_status.setText("<i>Not connected</i>")
            self._set_status("No OMERO connection.")
            return
        try:
            host = self._conn.host
            user = self._conn.getUser().getName()
        except Exception:  # noqa: BLE001
            host = user = "unknown"
        self._conn_status.setText(
            f"<span style='color:green'>Connected to <b>{host}</b> as <b>{user}</b></span>"
        )
        self._set_status(f"Connected to {host} as {user}")

    def _load_yaml(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open annotation config", "", "YAML files (*.yaml *.yml)"
        )
        if not path:
            return
        try:
            self._config = load_config(path)
            self._yaml_path.setText(path)
            self._sync_ui_from_config()
            self._set_status(f"Loaded config: {Path(path).name}")
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Error loading YAML: {exc}")

    def _browse_output_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select output directory")
        if d:
            self._output_dir.setText(d)

    def _list_tables(self) -> None:
        if self._conn is None:
            self._tables_display.setPlainText("Not connected.")
            return
        container_type = self._container_type.currentText()
        container_id_txt = self._container_id.text().strip()
        if not container_id_txt.isdigit():
            self._tables_display.setPlainText("Enter a valid container ID first.")
            return
        try:
            tables = list_annotation_tables(
                self._conn, container_type, int(container_id_txt)
            )
            if tables:
                lines = [f"ID {t['table_id']}: {t['name']}" for t in tables]
                self._tables_display.setPlainText("\n".join(lines))
            else:
                self._tables_display.setPlainText("No annotation tables found.")
        except Exception as exc:  # noqa: BLE001
            self._tables_display.setPlainText(f"Error: {exc}")

    def _refresh_summary(self) -> None:
        self._apply_ui_to_config()
        try:
            import yaml
            summary = yaml.dump(
                self._config.model_dump(exclude={"annotations"}),
                default_flow_style=False,
                allow_unicode=True,
            )
        except Exception:  # noqa: BLE001
            summary = str(self._config)
        self._config_summary.setPlainText(summary)

    def _run_pipeline(self) -> None:
        if self._conn is None:
            self._set_status("No OMERO connection. Connect first.")
            return

        self._apply_ui_to_config()

        # Resume from table if specified
        resume_txt = self._resume_id.text().strip()
        if resume_txt.isdigit():
            self._config.omero.table_id = int(resume_txt)

        self._log.clear()
        self._progress.setVisible(True)
        self._run_btn.setEnabled(False)
        self._set_status("Running annotation pipeline…")

        # Stage 1: OMERO setup in a background thread (no Qt widgets created)
        worker = AnnotationWorker(self._config, self._conn)
        worker.signals.ready_to_annotate.connect(self._on_setup_complete)
        worker.signals.error.connect(self._on_pipeline_error)
        worker.signals.log.connect(self._append_log)
        self._thread_pool.start(worker)

    def _save_yaml(self) -> None:
        self._apply_ui_to_config()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save annotation config", "annotation_config.yaml", "YAML files (*.yaml)"
        )
        if not path:
            return
        try:
            self._config.save_yaml(path)
            self._set_status(f"Config saved to {Path(path).name}")
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Error saving YAML: {exc}")

    # ------------------------------------------------------------------
    # Pipeline callbacks
    # ------------------------------------------------------------------

    def _on_setup_complete(self, pipeline: object, table_id: int) -> None:
        """Called on the main thread once OMERO setup is done.

        Now it is safe to call image_series_annotator because we are on the
        main (Qt/GL) thread.
        """
        self._append_log("Starting micro-SAM annotation (main thread)…")
        try:
            # Inject existing viewer so image_series_annotator reuses it
            pipeline._napari_viewer = self._viewer  # type: ignore[attr-defined]
            final_table_id, _ = pipeline.run_microsam_annotation()
            self._on_pipeline_finished(final_table_id if final_table_id else table_id)
        except Exception as exc:  # noqa: BLE001
            self._on_pipeline_error(str(exc))

    def _on_pipeline_finished(self, table_id: int) -> None:
        self._progress.setVisible(False)
        self._run_btn.setEnabled(True)
        msg = f"Done. Table ID: {table_id}" if table_id and table_id > 0 else "Done (read-only mode)."
        self._append_log(msg)
        self._set_status(msg)

    def _on_pipeline_error(self, error_msg: str) -> None:
        self._progress.setVisible(False)
        self._run_btn.setEnabled(True)
        self._append_log(f"ERROR: {error_msg}")
        self._set_status(f"Error: {error_msg}")

    def _append_log(self, text: str) -> None:
        self._log.appendPlainText(text)
        self._log.verticalScrollBar().setValue(
            self._log.verticalScrollBar().maximum()
        )

    # ------------------------------------------------------------------
    # Config sync helpers
    # ------------------------------------------------------------------

    def _apply_ui_to_config(self) -> None:
        """Push UI values into self._config."""
        # OMERO
        self._config.name = self._name_edit.text().strip() or "annotation"
        self._config.omero.container_type = self._container_type.currentText()
        cid_txt = self._container_id.text().strip()
        if cid_txt.isdigit():
            self._config.omero.container_id = int(cid_txt)
        out = self._output_dir.text().strip()
        if out:
            self._config.output.output_directory = out

        # Spatial coverage — use list fields, not singular aliases
        sc = self._config.spatial_coverage
        sc.channels = [self._channel.value()]
        sc.timepoint_mode = self._timepoint_mode.currentText()
        if sc.timepoint_mode == "specific":
            sc.timepoints = [int(x.strip()) for x in self._timepoints_edit.text().split(",") if x.strip().isdigit()]
        sc.z_slice_mode = self._z_slice_mode.currentText()
        if sc.z_slice_mode == "specific":
            sc.z_slices = [int(x.strip()) for x in self._z_slices_edit.text().split(",") if x.strip().isdigit()]
        sc.three_d = self._three_d.isChecked()
        if sc.three_d:
            sc.z_range_start = self._z_range_start.value()
            sc.z_range_end = self._z_range_end.value()
        sc.use_patches = self._use_patches.isChecked()
        if sc.use_patches:
            sc.patches_per_image = self._patches_per_image.value()
            sc.random_patches = self._random_patches.isChecked()
            parts = [p.strip() for p in self._patch_size_edit.text().split(",") if p.strip().isdigit()]
            if len(parts) == 2:
                sc.patch_size = [int(parts[0]), int(parts[1])]

        # Training
        tr = self._config.training
        tr.segment_all = self._segment_all.isChecked()
        tr.train_n = self._train_n.value()
        tr.validate_n = self._validate_n.value()
        tr.test_n = self._test_n.value()

        # Technical / workflow
        self._config.ai_model.pretrained_from = self._model_type.currentText()
        self._config.workflow.batch_size = self._batch_size.value()
        self._config.workflow.read_only_mode = self._read_only.isChecked()

    def _sync_ui_from_config(self) -> None:
        """Populate UI fields from self._config (after loading YAML)."""
        self._name_edit.setText(self._config.name or "")

        idx = self._container_type.findText(self._config.omero.container_type or "dataset")
        if idx >= 0:
            self._container_type.setCurrentIndex(idx)
        cid = self._config.omero.get_primary_container_id()
        self._container_id.setText(str(cid) if cid else "")
        if self._config.output.output_directory:
            self._output_dir.setText(str(self._config.output.output_directory))

        # Spatial
        sc = self._config.spatial_coverage
        if sc.channels:
            self._channel.setValue(sc.channels[0])
        idx = self._timepoint_mode.findText(sc.timepoint_mode or "specific")
        if idx >= 0:
            self._timepoint_mode.setCurrentIndex(idx)
        self._timepoints_edit.setText(",".join(str(t) for t in (sc.timepoints or [0])))
        idx = self._z_slice_mode.findText(sc.z_slice_mode or "specific")
        if idx >= 0:
            self._z_slice_mode.setCurrentIndex(idx)
        self._z_slices_edit.setText(",".join(str(z) for z in (sc.z_slices or [0])))
        self._three_d.setChecked(sc.three_d or False)
        if sc.z_range_start is not None:
            self._z_range_start.setValue(sc.z_range_start)
        if sc.z_range_end is not None:
            self._z_range_end.setValue(sc.z_range_end)
        self._use_patches.setChecked(sc.use_patches or False)
        self._patches_per_image.setValue(sc.patches_per_image or 1)
        if sc.patch_size and len(sc.patch_size) == 2:
            self._patch_size_edit.setText(f"{sc.patch_size[0]},{sc.patch_size[1]}")
        self._random_patches.setChecked(sc.random_patches if sc.random_patches is not None else True)

        # Training
        tr = self._config.training
        self._segment_all.setChecked(tr.segment_all or False)
        self._train_n.setValue(tr.train_n or 3)
        self._validate_n.setValue(tr.validate_n or 2)
        self._test_n.setValue(tr.test_n or 0)

        # Technical / workflow
        idx = self._model_type.findText(self._config.ai_model.pretrained_from or "vit_b_lm")
        if idx >= 0:
            self._model_type.setCurrentIndex(idx)
        self._batch_size.setValue(self._config.workflow.batch_size or 0)
        self._read_only.setChecked(self._config.workflow.read_only_mode or False)
        if self._config.omero.table_id:
            self._resume_id.setText(str(self._config.omero.table_id))

    # ------------------------------------------------------------------

    def _set_status(self, msg: str) -> None:
        self._status.setText(msg)
