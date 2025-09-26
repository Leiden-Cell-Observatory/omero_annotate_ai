# 🔬 OMERO Annotation Workflow - Streamlit Prototype

A fully functional web application that converts the Jupyter notebook workflow into an interactive web interface using your existing `omero-annotate-ai` package.

## ✨ Features

- **🔌 Real OMERO Connection**: Uses your actual `create_omero_connection_widget()`
- **🔬 Real Workflow Setup**: Uses your actual `create_workflow_widget()`
- **🚀 Real Pipeline Execution**: Uses your actual `create_pipeline()` and `run_full_micro_sam_workflow()`
- **🎨 Napari Integration**: Launches napari for interactive annotation exactly like the notebook
- **📋 Step-by-step Workflow**: Guided 4-step process matching the notebook structure

## 🚀 Quick Start

**Option 1: Using Pixi (Recommended)**
```bash
# Install streamlit environment and run
pixi run streamlit-app
```

**Option 2: Manual Setup**
```bash
cd gui/streamlit_prototype
pip install -r requirements.txt
streamlit run app.py
```

## 📱 Web Interface

The app provides a step-by-step workflow:

1. **🔌 OMERO Connection** - Connect using your existing widget
2. **🔬 Workflow Setup** - Configure using your existing workflow widget
3. **📋 Configuration Review** - Review all settings before execution
4. **🚀 Run Pipeline** - Execute the full annotation workflow with napari

## 🎯 Workflow Steps

This exactly mirrors your notebook workflow:

| Notebook Cell | Streamlit Step | Functionality |
|---------------|----------------|---------------|
| Cell 4-5 | Step 1 | OMERO connection using `create_omero_connection_widget()` |
| Cell 7 | Step 2 | Workflow configuration using `create_workflow_widget()` |
| Cell 10-12 | Step 3 | Configuration review and validation |
| Cell 14 | Step 4 | Pipeline execution with `run_full_micro_sam_workflow()` |

## 🔧 How It Works

The app integrates directly with your existing code:

```python
# Step 1: Real OMERO connection
conn_widget = create_omero_connection_widget()
conn = conn_widget.get_connection()

# Step 2: Real workflow configuration
workflow_widget = create_workflow_widget(connection=conn)
config = workflow_widget.get_config()

# Step 3: Real pipeline execution
pipeline = create_pipeline(config, conn)
table_id, updated_config = pipeline.run_full_micro_sam_workflow()
```

## 🎭 Next Steps

This prototype validates the notebook → webapp conversion. Once tested, we'll convert this to:

- **React + OMERO.web plugin** for production deployment
- **Integration with BIOMERO** for cluster-based training
- **Professional institutional deployment**

## 🐛 Requirements

- All your existing `omero-annotate-ai` dependencies
- Streamlit for the web interface
- Working OMERO connection
- Napari + micro-sam for annotation