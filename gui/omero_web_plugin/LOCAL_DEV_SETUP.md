# Local Development Setup (Without omero CLI)

Quick guide for testing the plugin with your local OMERO.web development setup.

## Prerequisites

- OMERO.web source code at: `/var/home/maartenpaul/Documents/GitHub/omero-web`
- Virtual environment: `omero-web` (uv managed)
- Plugin installed: ✅ (you just did this)

## Setup Steps

### 1. Configure the Plugin in OMERO.web Settings

You have two options:

#### Option A: Using Environment Variables (Recommended for Dev)

```bash
cd /var/home/maartenpaul/Documents/GitHub/omero-web

# Set the app in environment variable
export OMERO_WEB_APPS='["omero_annotate_web"]'

# Optional: Add to your shell profile for persistence
echo 'export OMERO_WEB_APPS='"'"'["omero_annotate_web"]'"'"'' >> ~/.bashrc
```

#### Option B: Edit settings.py Directly

Edit `/var/home/maartenpaul/Documents/GitHub/omero-web/omeroweb/settings.py`:

Find the `INSTALLED_APPS` section and add:

```python
INSTALLED_APPS = [
    'django.contrib.staticfiles',
    'corsheaders',
    'omeroweb.connector',
    'omeroweb.webadmin',
    'omeroweb.webclient',
    'omeroweb.webgateway',
    # Add your plugin here:
    'omero_annotate_web',
]
```

### 2. Collect Static Files

```bash
cd /var/home/maartenpaul/Documents/GitHub/omero-web

# Collect all static files (CSS, JS)
python manage.py collectstatic --noinput
```

### 3. Add URL Configuration

Edit `/var/home/maartenpaul/Documents/GitHub/omero-web/omeroweb/urls.py`:

Add your plugin's URLs:

```python
from django.urls import include, path

urlpatterns = [
    # ... existing patterns ...

    # Add this line:
    path('omero_annotate_web/', include('omero_annotate_web.urls')),
]
```

### 4. Start the Development Server

```bash
cd /var/home/maartenpaul/Documents/GitHub/omero-web

# Start Django development server
python manage.py runserver 4080
```

### 5. Access Your Plugin

Open your browser to:
```
http://localhost:4080/omero_annotate_web/
```

## Quick Commands Reference

```bash
# Navigate to OMERO.web directory
cd /var/home/maartenpaul/Documents/GitHub/omero-web

# Activate virtual environment (if not already active)
source .venv/bin/activate

# Collect static files
python manage.py collectstatic --noinput

# Run development server
python manage.py runserver 4080

# Check for errors
python manage.py check

# Create migrations (if you add models later)
python manage.py makemigrations omero_annotate_web

# Apply migrations
python manage.py migrate
```

## Troubleshooting

### Plugin not found

**Check installation:**
```bash
python -c "import omero_annotate_web; print(omero_annotate_web.__file__)"
```

**Should output something like:**
```
/var/home/maartenpaul/Documents/GitHub/omero_annotate_ai.worktrees/origin-gui_dev/gui/omero_web_plugin/omero_annotate_web/__init__.py
```

### Static files not loading

```bash
cd /var/home/maartenpaul/Documents/GitHub/omero-web
python manage.py collectstatic --noinput

# Check if files were collected
ls -la static/omero_annotate_web/
```

### URL not found (404 error)

Make sure you added the URL pattern to `omeroweb/urls.py` (see Step 3 above).

### Module import errors

```bash
# Reinstall the plugin
cd /var/home/maartenpaul/Documents/GitHub/omero_annotate_ai.worktrees/origin-gui_dev/gui/omero_web_plugin
uv pip install -e . --force-reinstall
```

## Testing Without OMERO Server

**Note:** Your plugin requires OMERO server connection for full functionality. However, you can:

1. ✅ **Test the UI** - Pages will load, forms will render
2. ✅ **Test JavaScript** - Frontend logic works independently
3. ❌ **Test API calls** - Will fail without OMERO connection
4. ❌ **Save configs** - Requires OMERO server

### Mock Testing (Optional)

To test API endpoints without OMERO server, you could add mock responses. Create a test mode in `views.py`:

```python
import os

MOCK_MODE = os.environ.get('MOCK_OMERO', 'false').lower() == 'true'

if MOCK_MODE:
    # Return mock data for testing
    pass
else:
    # Use real OMERO connection
    pass
```

Then run with:
```bash
export MOCK_OMERO=true
python manage.py runserver 4080
```

## Development Workflow

1. **Make changes** to your plugin code
2. **Restart server** (Django auto-reloads Python files)
3. **Refresh browser** to see changes
4. **For static file changes** (CSS/JS):
   ```bash
   python manage.py collectstatic --noinput
   # Then refresh browser with Ctrl+Shift+R (hard refresh)
   ```

## File Watching for Auto-Reload

For automatic static file collection on changes, install `watchdog`:

```bash
uv pip install watchdog
python manage.py runserver 4080 --settings=omeroweb.settings_development
```

## Next Steps

Once the basic setup works:

1. Connect to an OMERO server (configure in `omeroweb/settings.py`)
2. Test the full workflow with real data
3. Debug any OMERO connection issues
4. Iterate on the UI/functionality

## Connecting to OMERO Server

If you have access to an OMERO server, configure it in settings:

```python
# In omeroweb/settings.py or as environment variable
OMERO_HOST = 'your-omero-server.example.com'
OMERO_PORT = 4064
```

Or use environment variables:
```bash
export OMERO_HOST=your-omero-server.example.com
export OMERO_PORT=4064
```

## Production Deployment

For production deployment on a real OMERO.web server, see [INSTALL.md](INSTALL.md).
