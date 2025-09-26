# React + OMERO.web Plugin Framework

This folder contains the framework for a production React-based OMERO.web plugin, designed after validating the Streamlit prototype.

## Architecture

Following the NL-BIOMERO pattern for professional OMERO.web integration:

```
omero_web_plugin/
├── omero_annotate_web/           # Django app for OMERO.web
│   ├── __init__.py
│   ├── apps.py                   # Django app configuration
│   ├── views.py                  # Backend views and API endpoints
│   ├── urls.py                   # URL routing
│   ├── models.py                 # Database models (if needed)
│   ├── static/                   # Frontend assets
│   │   ├── css/
│   │   ├── js/                   # React components
│   │   │   ├── components/
│   │   │   │   ├── OMEROConnection.jsx
│   │   │   │   ├── WorkflowSetup.jsx
│   │   │   │   ├── ConfigReview.jsx
│   │   │   │   └── PipelineRunner.jsx
│   │   │   ├── App.jsx           # Main React app
│   │   │   └── index.js          # Entry point
│   │   └── dist/                 # Built assets
│   └── templates/
│       └── omero_annotate_web/
│           └── annotation_interface.html
├── setup.py                      # Plugin installation
├── package.json                  # Node.js dependencies
├── webpack.config.js            # Build configuration
└── docker/                      # Deployment configuration
    ├── Dockerfile
    └── docker-compose.yml
```

## Features

- **Native OMERO.web Integration**: Uses Django framework
- **Shared Authentication**: Leverages OMERO.web's auth system
- **React Frontend**: Modern, responsive UI components
- **API Backend**: RESTful endpoints for annotation workflow
- **Docker Deployment**: Professional containerized deployment
- **BIOMERO Integration**: Ready for cluster-based training

## Development Workflow

1. **Prototype Phase**: Validate with Streamlit ✅
2. **Framework Phase**: Create Django + React structure (Current)
3. **Development Phase**: Build production components
4. **Integration Phase**: BIOMERO cluster integration
5. **Deployment Phase**: Docker + institutional deployment

## Next Steps

This framework will be populated with:
- Django views that expose your `omero-annotate-ai` functionality
- React components that mirror the validated Streamlit interface
- API endpoints for real-time communication
- Integration with BIOMERO for scalable training