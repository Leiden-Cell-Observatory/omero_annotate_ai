# Installation Guide - OMERO Annotation Web Plugin

This guide provides detailed step-by-step instructions for installing and configuring the OMERO Annotation Web Plugin.

## Prerequisites

### System Requirements

- OMERO.web server (version 5.6.0 or later)
- Python 3.8 or later
- pip package manager
- Access to OMERO server with admin privileges (for configuration)

### Required Packages

- `omero-annotate-ai` - The main annotation package
- Django (installed with OMERO.web)
- PyYAML

## Installation Steps

### Step 1: Install omero-annotate-ai

If not already installed, install the omero-annotate-ai package:

```bash
pip install omero-annotate-ai
```

Or install from source in editable mode:

```bash
cd /path/to/omero_annotate_ai
pip install -e .
```

### Step 2: Locate the Plugin Directory

The plugin is located at:

```
/path/to/omero_annotate_ai/gui/omero_web_plugin/omero_annotate_web/
```

### Step 3: Make Plugin Available to OMERO.web

Option A: Install as Python Package (Recommended)

Add the plugin directory to your Python path by creating a development installation:

```bash
cd /path/to/omero_annotate_ai/gui/omero_web_plugin
pip install -e .
```

Option B: Symlink (Alternative)

Create a symlink in your OMERO.web site-packages:

```bash
ln -s /path/to/omero_annotate_ai/gui/omero_web_plugin/omero_annotate_web \
      /path/to/venv/lib/python3.X/site-packages/omero_annotate_web
```

### Step 4: Configure OMERO.web

Add the app to OMERO.web's installed apps:

```bash
omero config append omero.web.apps '"omero_annotate_web"'
```

Verify the configuration:

```bash
omero config get omero.web.apps
```

You should see `"omero_annotate_web"` in the list.

### Step 5: Add Navigation Link (Optional)

Add a link to the top navigation menu:

```bash
omero config append omero.web.ui.top_links \
  '["Annotation Workflow", "omero_annotate_web:workflow", {"title": "Configure AI Annotation Workflows"}]'
```

### Step 6: Configure URL Prefix (Optional)

**Note:** The URL prefix is rarely needed and affects ALL OMERO.web apps. Skip this step unless you specifically need it.

If required, set the prefix:

```bash
omero config set omero.web.prefix '/omero'
```

To check current prefix:
```bash
omero config get omero.web.prefix
```

To remove the prefix:
```bash
omero config set omero.web.prefix
```

### Step 7: Collect Static Files

Collect all static files (CSS, JavaScript) to the OMERO.web static directory:

```bash
omero web static
```

This will copy files from `omero_annotate_web/static/` to the OMERO.web static directory.

### Step 8: Restart OMERO.web

Restart the web server to apply changes:

```bash
omero web restart
```

Or if running as a service:

```bash
systemctl restart omero-web
```

### Step 9: Verify Installation

1. Open your browser and navigate to your OMERO.web instance
2. Log in with your OMERO credentials
3. Access the plugin at: `https://<your-omero-server>/omero_annotate_web/`
4. You should see the "AI Annotation Workflow Configuration" page

## Verification Checklist

Run these commands to verify your installation:

```bash
# Check if app is registered
omero config get omero.web.apps
# Should include: "omero_annotate_web"

# Check if top link is added
omero config get omero.web.ui.top_links
# Should include: annotation workflow link

# Check OMERO.web status
omero web status

# Check static files were collected
ls -la /path/to/omero/lib/python/omeroweb/static/omero_annotate_web/
```

## Troubleshooting

### Issue: Plugin not appearing in navigation

**Solution:**
1. Verify the app is in the config:
   ```bash
   omero config get omero.web.apps
   ```
2. Check that you ran `omero web static`
3. Clear browser cache
4. Restart OMERO.web

### Issue: "Module not found" error

**Solution:**
1. Verify the plugin is installed:
   ```bash
   python -c "import omero_annotate_web; print(omero_annotate_web.__file__)"
   ```
2. Check Python path includes the plugin location
3. Reinstall with `pip install -e .`

### Issue: Static files not loading (CSS/JS missing)

**Solution:**
1. Run `omero web static` again
2. Check file permissions in static directory
3. Verify OMERO.web static file configuration:
   ```bash
   omero config get omero.web.static_url
   ```

### Issue: OMERO connection errors

**Solution:**
1. Verify you're logged into OMERO.web
2. Check OMERO.web session configuration:
   ```bash
   omero config get omero.web.session_engine
   ```
3. Check OMERO server is running and accessible

### Issue: Permission errors when saving configs

**Solution:**
1. Verify OMERO user has write permissions on containers
2. Check OMERO server logs for detailed error messages
3. Ensure file annotation creation is allowed

### Issue: "No JSON object could be decoded" when using config commands

**Solution:**
OMERO config commands require proper quoting:

```bash
# For simple string values (NO nested quotes)
omero config set omero.web.prefix '/omero'

# For list/array items (YES nested quotes required)
omero config append omero.web.apps '"omero_annotate_web"'

# Incorrect examples that will fail:
omero config set omero.web.prefix '"/omero"'      # Wrong - adds quotes to value!
omero config append omero.web.apps "omero_annotate_web"  # Wrong - missing nested quotes
```

To remove config values, use the exact same syntax as when adding:
```bash
omero config remove omero.web.apps '"omero_annotate_web"'
```

## Advanced Configuration

### Custom Static Files Location

If you need to customize the static files location:

```bash
omero config set omero.web.static_root '/custom/path/to/static'
omero web static
```

### Debug Mode (Development Only)

Enable debug mode for development:

```bash
omero config set omero.web.debug True
```

**WARNING:** Never use debug mode in production!

### Session Configuration

Configure session timeout (in seconds):

```bash
omero config set omero.web.session_cookie_age 86400
```

### CORS Configuration (if needed)

If accessing from different domains:

```bash
omero config append omero.web.middleware \
  '{"index": 0.5, "class": "django.middleware.common.CommonMiddleware"}'
```

## Uninstallation

To remove the plugin:

```bash
# Remove from OMERO.web apps
omero config remove omero.web.apps '"omero_annotate_web"'

# Remove top link (if added)
omero config remove omero.web.ui.top_links \
  '["Annotation Workflow", "omero_annotate_web:workflow", {"title": "Configure AI Annotation Workflows"}]'

# Remove static files
rm -rf /path/to/omero/lib/python/omeroweb/static/omero_annotate_web/

# Restart OMERO.web
omero web restart
```

## Upgrading

To upgrade to a new version:

```bash
# Pull latest code
cd /path/to/omero_annotate_ai
git pull

# Reinstall
cd gui/omero_web_plugin
pip install -e . --upgrade

# Collect updated static files
omero web static

# Restart
omero web restart
```

## Docker Deployment (Advanced)

For containerized deployment, see the Docker documentation in the main repository.

## Production Considerations

### Performance

- Use a production WSGI server (gunicorn, uWSGI)
- Configure proper caching
- Use a CDN for static files if needed

### Security

- Always use HTTPS in production
- Configure CSRF protection properly
- Set secure session cookies:
  ```bash
  omero config set omero.web.session_cookie_secure True
  ```

### Monitoring

- Monitor OMERO.web logs: `/path/to/omero/var/log/OMEROweb.log`
- Set up error notifications
- Monitor server resources

## Getting Help

- Check OMERO.web logs for error details
- Review Django app logs
- Consult OMERO.web documentation: https://docs.openmicroscopy.org/omero/latest/sysadmins/unix/install-web/
- Open an issue on the GitHub repository

## Next Steps

After installation:
1. Test the workflow with a sample dataset
2. Configure annotation parameters for your use case
3. Train your team on using the interface
4. Set up automated backups of configurations
