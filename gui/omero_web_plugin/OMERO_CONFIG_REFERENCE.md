# OMERO Configuration Command Reference

Quick reference for managing the OMERO.web plugin configuration.

## Installation Commands

```bash
# Add the app to OMERO.web
omero config append omero.web.apps '"omero_annotate_web"'

# Add navigation link (optional)
omero config append omero.web.ui.top_links '["Annotation Workflow", "omero_annotate_web:workflow", {"title": "AI Annotation Workflows"}]'

# Collect static files
omero web static

# Restart OMERO.web
omero web restart
```

## Verification Commands

```bash
# Check if app is registered
omero config get omero.web.apps

# Check navigation links
omero config get omero.web.ui.top_links

# Check OMERO.web status
omero web status

# View all OMERO.web config
omero config get --show-password
```

## Uninstallation Commands

```bash
# Remove the app (use exact same quoting as when adding!)
omero config remove omero.web.apps '"omero_annotate_web"'

# Remove navigation link (if added)
omero config remove omero.web.ui.top_links '["Annotation Workflow", "omero_annotate_web:workflow", {"title": "AI Annotation Workflows"}]'

# Remove static files
rm -rf /path/to/omero/lib/python/omeroweb/static/omero_annotate_web/

# Restart OMERO.web
omero web restart
```

## Important Quoting Rules

### ✅ Correct Syntax

```bash
# Simple string values (no nested quotes needed)
omero config set omero.web.prefix '/omero'

# String values in lists/arrays need NESTED quotes
omero config append omero.web.apps '"omero_annotate_web"'

# JSON/list values use single quotes around the JSON
omero config append omero.web.ui.top_links '["Link Name", "app:view", {"title": "Description"}]'
```

### ❌ Incorrect Syntax (will fail)

```bash
# Simple strings with nested quotes (adds quotes to value!)
omero config set omero.web.prefix '"/omero"'  # Wrong - creates "/omero" with quotes

# Missing nested quotes for list items
omero config append omero.web.apps "omero_annotate_web"  # FAILS

# Wrong quote order
omero config append omero.web.apps "'omero_annotate_web'"  # FAILS
```

## Common Operations

### Check Current Configuration

```bash
# List all installed apps
omero config get omero.web.apps

# Check URL prefix
omero config get omero.web.prefix

# Check session settings
omero config get omero.web.session_cookie_age
```

### Modify Settings

```bash
# Set session timeout (24 hours)
omero config set omero.web.session_cookie_age 86400

# Set URL prefix (rarely needed)
omero config set omero.web.prefix '/omero'

# Enable debug mode (DEVELOPMENT ONLY!)
omero config set omero.web.debug True
```

### Development vs Production

```bash
# Development mode (auto-reload, detailed errors)
omero web start --development

# Production mode
omero web start

# Check which mode is running
ps aux | grep "omero web"
```

## Troubleshooting

### "No JSON object could be decoded"

This error means improper quoting. Always use nested quotes:

```bash
# Correct
omero config append omero.web.apps '"omero_annotate_web"'

# Incorrect (causes error)
omero config append omero.web.apps "omero_annotate_web"
```

### "Plugin not appearing"

```bash
# Verify registration
omero config get omero.web.apps

# Should output list containing:
# "omero_annotate_web"

# If not there, add it again
omero config append omero.web.apps '"omero_annotate_web"'
```

### "Static files not loading"

```bash
# Re-collect static files
omero web static

# Check if files exist
ls -la /path/to/omero/lib/python/omeroweb/static/omero_annotate_web/

# Verify static file configuration
omero config get omero.web.static_root
omero config get omero.web.static_url
```

## Complete Setup Example

```bash
# 1. Install plugin
cd gui/omero_web_plugin
pip install -e .

# 2. Configure OMERO.web
omero config append omero.web.apps '"omero_annotate_web"'

# 3. (Optional) Add navigation link
omero config append omero.web.ui.top_links '["Annotation Workflow", "omero_annotate_web:workflow", {"title": "AI Annotation Workflows"}]'

# 4. Collect static files
omero web static

# 5. Restart server
omero web restart

# 6. Verify
omero config get omero.web.apps
omero web status
```

## Complete Removal Example

```bash
# 1. Remove from config (exact same quotes as when adding!)
omero config remove omero.web.apps '"omero_annotate_web"'

# 2. Remove navigation link (if added)
omero config remove omero.web.ui.top_links '["Annotation Workflow", "omero_annotate_web:workflow", {"title": "AI Annotation Workflows"}]'

# 3. Remove static files
rm -rf $(omero config get omero.web.static_root)/omero_annotate_web/

# 4. Restart server
omero web restart

# 5. Verify removal
omero config get omero.web.apps
```

## Additional Resources

- [OMERO.web documentation](https://docs.openmicroscopy.org/omero/latest/sysadmins/unix/install-web/)
- [OMERO config documentation](https://docs.openmicroscopy.org/omero/latest/sysadmins/config.html)
- [Plugin INSTALL.md](INSTALL.md) - Detailed installation guide
