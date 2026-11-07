# OMERO Annotation Web Plugin

A Django web application for OMERO.web that provides an intuitive interface for configuring AI-powered annotation workflows using SAM (Segment Anything Model). Users can configure annotation parameters, generate YAML configurations, and save them as OMERO FileAnnotations for use with the omero-annotate-ai CLI tools.

## Features

- **Multi-step workflow interface** - Guided 4-step process (Container → Tables → Configuration → Save)
- **Container support** - Works with Projects, Datasets, Plates, and Screens
- **Table management** - Scan existing annotation tables or create new timestamped ones
- **Complete parameter coverage** - All annotation settings from the core library
- **Live YAML preview** - Real-time configuration preview with validation
- **OMERO integration** - Direct connection to OMERO server, saves configs as FileAnnotations
- **Simple deployment** - No build step required, pure HTML/JavaScript/CSS

## Quick Start

### Installation

**Using pip:**
```bash
cd gui/omero_web_plugin
pip install -e .
```

**Using uv (recommended):**
```bash
cd gui/omero_web_plugin
uv pip install -e .
```

### OMERO.web Configuration

**Basic Setup (Standalone Page):**
```bash
# Add the plugin to OMERO.web
omero config append omero.web.apps '"omero_annotate_web"'

# Optional: Add to top menu bar
omero config append omero.web.ui.top_links '["Annotation Workflow", "omero_annotate_web:workflow", {"title": "AI Annotation Workflows"}]'

# Deploy static files and restart
omero web static
omero web restart
```

**Center Panel Integration (Like omero.parade):**
```bash
# Add as center plugin (integrates into OMERO.webclient)
omero config append omero.web.ui.center_plugins '["Annotation Workflow", "omero_annotate_web/index.html", "omero_annotate_workflow"]'

# Also add to apps if not already done
omero config append omero.web.apps '"omero_annotate_web"'

# Deploy and restart
omero web static
omero web restart
```

### Access

**Standalone mode:**
Navigate to: `http://<your-omero-server>/omero_annotate_web/`

**Center panel mode:**
1. Log into OMERO.webclient
2. Select any Project, Dataset, Plate, or Screen
3. Click "Annotation Workflow" in the center panel tabs

## Workflow

**Step 1: Container Selection**
- Select container type and specific container
- View metadata (image count, dimensions, channels, etc.)

**Step 2: Table Management**
- Scan for existing annotation tables
- Choose to resume from existing table or create new one
- Generate unique table names

**Step 3: Configuration**
Three tabs with all annotation parameters:
- **Annotation Settings**: training/validation counts, channels, timepoints, Z-slices, 3D, patches
- **Technical Settings**: batch size, SAM model (ViT-B/L/H)
- **Workflow Settings**: read-only mode, resume from table

**Step 4: Save**
- Preview generated YAML
- Save to OMERO as FileAnnotation
- Download for CLI use

## Architecture

### Backend (Django)
- `views.py` - 9 API endpoints for workflow operations
- `utils.py` - OMERO operations, config conversion, YAML generation
- `urls.py` - URL routing

### Frontend (HTML/JavaScript)
- Component-based JavaScript architecture
- Progressive enhancement with jQuery
- Responsive CSS matching OMERO.web style

### File Structure
```
omero_annotate_web/
├── views.py, urls.py, utils.py     # Backend
├── static/omero_annotate_web/
│   ├── css/workflow.css            # Styling
│   └── js/
│       ├── api-client.js           # API wrapper
│       ├── workflow-app.js         # Main controller
│       └── components/             # Step components
└── templates/omero_annotate_web/
    ├── workflow.html               # Main page
    └── partials/                   # Step templates
```

## API Endpoints

The plugin exposes the following REST API endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Main workflow interface |
| `GET` | `/api/containers/<type>/` | List containers of specified type |
| `GET` | `/api/container/<type>/<id>/info/` | Get detailed container metadata |
| `POST` | `/api/tables/scan/` | Scan for existing annotation tables |
| `POST` | `/api/tables/generate-name/` | Generate unique table name |
| `POST` | `/api/config/validate/` | Validate configuration parameters |
| `POST` | `/api/config/preview/` | Generate YAML preview |
| `POST` | `/api/config/save/` | Save configuration to OMERO |

## Configuration Files

Configurations are saved as OMERO FileAnnotations with:
- **Namespace**: `omero.annotate_ai.config.v1`
- **Format**: YAML
- **Attachment**: Linked to the selected container (Project/Dataset/Plate/Screen)

These configurations can be:
- Viewed in OMERO.web interface
- Downloaded and used with omero-annotate-ai CLI
- Shared across your OMERO instance
- Versioned by timestamp in filename

## Development

### Local Development

**Without full OMERO installation:**
```bash
# Run Django dev server directly
cd /path/to/omero-web
export OMERO_WEB_DEV_MODE=true
export OMERODIR=/path/to/omero
python omeroweb/manage.py runserver 4080
```

**With OMERO.web:**
```bash
# Development mode with auto-reload
omero web start --development

# Check installation status
omero config get omero.web.apps
omero web status
```

### After Code Changes

```bash
# Reinstall plugin to pick up changes
cd gui/omero_web_plugin
uv pip install -e . --force-reinstall --no-deps

# Restart OMERO.web
omero web restart
```

## Troubleshooting

### Plugin Not Appearing

```bash
# Verify plugin is installed
omero config get omero.web.apps
# Should show: [..., "omero_annotate_web"]

# Reinstall if needed
cd gui/omero_web_plugin
uv pip install -e . --force-reinstall --no-deps

# Re-collect static files
omero web static

# Restart server
omero web restart
```

### Connection Issues

**No OMERO username showing:**
- Verify you're logged into OMERO.web first
- Plugin uses OMERO.web session authentication
- Check Django logs for authentication errors

**API endpoints returning 404:**
- Ensure package was reinstalled after code changes
- Verify URL patterns in browser DevTools Network tab
- Check that `omero.web.prefix` is correctly set

**Static files not loading (CSS/JS):**
- Run `omero web static` to collect static files
- Check browser console (F12) for 404 errors
- Verify `static/omero_annotate_web/` directory structure

**Logs to check:**
- OMERO.web: `/path/to/omero/var/log/OMEROweb.log`
- Browser console: F12 → Console tab
- Network requests: F12 → Network tab

### Debug Mode

Enable verbose logging:
```python
# In views.py, temporarily set logger level
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
```

See [INSTALL.md](INSTALL.md) and [LOCAL_DEV_SETUP.md](LOCAL_DEV_SETUP.md) for detailed troubleshooting.

## Requirements

- **OMERO.web** 5.6.0 or higher
- **Python** 3.8 or higher (tested with 3.13)
- **omero-annotate-ai** - Core annotation library
- **PyYAML** - For YAML config generation
- **Django** 3.2+ (provided by OMERO.web)

## Technology Stack

- **Backend**: Django (OMERO.web framework)
- **Frontend**: Vanilla JavaScript + jQuery
- **Styling**: CSS3 (OMERO.web compatible)
- **Data Format**: YAML
- **OMERO API**: BlitzGateway (Python)

## Project Structure

```
gui/omero_web_plugin/
├── omero_annotate_web/          # Main Django app
│   ├── __init__.py             # App config
│   ├── views.py                # API endpoints (9 views)
│   ├── urls.py                 # URL routing
│   ├── utils.py                # OMERO helpers
│   ├── static/
│   │   └── omero_annotate_web/
│   │       ├── css/
│   │       │   └── workflow.css
│   │       └── js/
│   │           ├── api-client.js
│   │           ├── workflow-app.js
│   │           └── components/
│   │               ├── container-selector.js
│   │               ├── table-manager.js
│   │               ├── config-editor.js
│   │               └── yaml-preview.js
│   └── templates/
│       └── omero_annotate_web/
│           ├── base.html
│           ├── workflow.html
│           └── partials/
│               ├── step_container.html
│               ├── step_tables.html
│               ├── step_config.html
│               └── step_save.html
├── setup.py                     # Package config
├── MANIFEST.in                  # Static file inclusion
├── README.md                    # This file
├── INSTALL.md                   # Detailed install guide
├── LOCAL_DEV_SETUP.md          # Development guide
├── OMERO_CONFIG_REFERENCE.md   # OMERO config commands
└── STATUS.md                    # Implementation status
```

## Future Enhancements

- Right-panel integration in OMERO.webclient
- Configuration template library with presets
- Direct annotation job execution from web interface
- Progress tracking for running jobs
- React + Blueprint.js UI upgrade (optional)
- Batch processing for multiple containers

## Documentation

- **[INSTALL.md](INSTALL.md)** - Detailed installation and configuration
- **[LOCAL_DEV_SETUP.md](LOCAL_DEV_SETUP.md)** - Local development guide
- **[OMERO_CONFIG_REFERENCE.md](OMERO_CONFIG_REFERENCE.md)** - OMERO config commands
- **[STATUS.md](STATUS.md)** - Current implementation status
- **[CLAUDE.md](../../CLAUDE.md)** - OMERO.web development knowledge base

## Contributing

This plugin follows OMERO.web development patterns. Key guidelines:

1. Use `@login_required()` decorator from `omeroweb.decorators`
2. Accept `conn=None, **kwargs` in all views
3. Static files must be in `static/omero_annotate_web/`
4. Reinstall package after changes: `uv pip install -e . --force-reinstall --no-deps`
5. See [CLAUDE.md](../../CLAUDE.md) for detailed development guidelines

## License

Apache License 2.0 (same as omero-annotate-ai parent project)

## Support

- **Issues**: [GitHub Issues](https://github.com/your-org/omero-annotate-ai/issues)
- **Documentation**: See docs/ directory in parent repository
- **OMERO.web docs**: https://docs.openmicroscopy.org/omero/latest/developers/Web.html
