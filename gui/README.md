# 🎨 GUI Development Framework

This directory contains the complete GUI development framework for converting the OMERO annotation notebook into web applications.

## 📁 Structure

```
gui/
├── streamlit_prototype/     # ✅ Fully functional Streamlit prototype
├── omero_web_plugin/       # 🏗️ React + OMERO.web plugin framework
├── shared/                 # 🔄 Shared components and utilities
└── README.md              # 📖 This file
```

## 🚀 Quick Start

### Streamlit Prototype (Ready to Use)
```bash
# Option 1: Using pixi (recommended)
pixi run streamlit-app

# Option 2: Manual setup
cd gui/streamlit_prototype
pip install -r requirements.txt
streamlit run app.py
```

### React Plugin (Framework Ready)
```bash
cd gui/omero_web_plugin
# Development instructions coming soon
```

## 🎯 Development Roadmap

### ✅ Phase 1: Streamlit Prototype (COMPLETED)
**Fully functional web application that integrates with your existing code**

- **Real OMERO Connection**: Uses `create_omero_connection_widget()`
- **Real Workflow Setup**: Uses `create_workflow_widget()`
- **Real Pipeline Execution**: Uses `create_pipeline()` and `run_full_micro_sam_workflow()`
- **Napari Integration**: Launches napari for annotation exactly like the notebook
- **Pixi Integration**: `pixi run streamlit-app` for easy installation

**Purpose**: Validate notebook → webapp conversion and test user experience

### 🏗️ Phase 2: React + OMERO.web Plugin (FRAMEWORK READY)
**Professional production deployment following NL-BIOMERO patterns**

**Architecture**:
- **Django Backend**: Native OMERO.web integration with shared authentication
- **React Frontend**: Modern, responsive UI components
- **API Endpoints**: RESTful communication between frontend/backend
- **Docker Deployment**: Professional containerized deployment

**Integration Benefits**:
- 🔗 Shared OMERO sessions from OMERO.web
- 🔐 Built-in authentication via OMERO.web
- 📦 Docker deployment like NL-BIOMERO
- 🔧 Ready for BIOMERO cluster integration

### 🌐 Phase 3: Production Deployment (PLANNED)
**Enterprise-ready institutional deployment**

- **BIOMERO Integration**: Cluster-based training via SLURM
- **Container Orchestration**: Kubernetes deployment options
- **Monitoring & Logging**: Production observability
- **Security & Compliance**: Enterprise security standards

## 🔄 Conversion Strategy

**Streamlit → React Migration Path**:

| Component | Streamlit Implementation | React Implementation |
|-----------|-------------------------|---------------------|
| **OMERO Connection** | `create_omero_connection_widget()` display | Django view + React form |
| **Workflow Setup** | `create_workflow_widget()` display | API endpoints + React components |
| **Configuration Review** | Streamlit dataframes/displays | React tables and cards |
| **Pipeline Execution** | Direct `pipeline.run_full_micro_sam_workflow()` | API endpoint + progress tracking |

## 🏃‍♂️ Next Steps

### For Testing:
1. **Test Streamlit prototype** with your workflows
2. **Validate user experience** and interface design
3. **Identify improvements** for React conversion

### For Production:
1. **Populate React framework** with validated designs
2. **Integrate BIOMERO** for cluster training
3. **Deploy with Docker** following NL-BIOMERO patterns

## 🎭 Benefits of This Approach

### ✅ **Rapid Prototyping**:
- Streamlit prototype working in hours, not weeks
- Real integration with your existing `omero-annotate-ai` code
- Immediate user testing and feedback

### ✅ **Production Ready**:
- React framework follows proven OMERO.web patterns
- Professional deployment with Docker
- Integration with BIOMERO for scalable training

### ✅ **Risk Mitigation**:
- Validate concepts before heavy React development
- Test user workflows in real environment
- Ensure technical feasibility before production investment

## 🤝 Integration Points

Both interfaces integrate seamlessly with your existing codebase:

```python
# The same code works in both Streamlit and React contexts:
from omero_annotate_ai import (
    create_omero_connection_widget,
    create_workflow_widget,
    create_pipeline
)

# 1. OMERO Connection
conn_widget = create_omero_connection_widget()
conn = conn_widget.get_connection()

# 2. Workflow Configuration
workflow_widget = create_workflow_widget(connection=conn)
config = workflow_widget.get_config()

# 3. Pipeline Execution
pipeline = create_pipeline(config, conn)
table_id, updated_config = pipeline.run_full_micro_sam_workflow()
```

The transition from Streamlit to React preserves all functionality while adding enterprise-grade deployment capabilities!