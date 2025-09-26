/**
 * Main React App for OMERO Annotation Workflow
 *
 * This mirrors the Streamlit prototype structure but as a React application
 * integrated with OMERO.web's Django backend.
 */

const { useState, useEffect } = React;

// Main App Component
function AnnotationWorkflowApp() {
    const [currentStep, setCurrentStep] = useState(1);
    const [omeroConnection, setOmeroConnection] = useState(null);
    const [workflowConfig, setWorkflowConfig] = useState(null);
    const [pipelineId, setPipelineId] = useState(null);

    const steps = [
        { id: 1, name: "🔌 OMERO Connection", component: OMEROConnectionStep },
        { id: 2, name: "🔬 Workflow Setup", component: WorkflowSetupStep },
        { id: 3, name: "📋 Configuration Review", component: ConfigReviewStep },
        { id: 4, name: "🚀 Run Pipeline", component: PipelineRunnerStep }
    ];

    const currentStepComponent = steps.find(step => step.id === currentStep)?.component;

    return (
        <div className="annotation-workflow-app">
            {/* Progress Indicator */}
            <div className="workflow-progress mb-4">
                <div className="row">
                    {steps.map(step => (
                        <div key={step.id} className="col-3">
                            <div className={`step-indicator ${step.id === currentStep ? 'active' : step.id < currentStep ? 'completed' : 'pending'}`}>
                                <div className="step-number">{step.id}</div>
                                <div className="step-name">{step.name}</div>
                            </div>
                        </div>
                    ))}
                </div>
                <div className="progress">
                    <div
                        className="progress-bar"
                        style={{ width: `${((currentStep - 1) / (steps.length - 1)) * 100}%` }}
                    ></div>
                </div>
            </div>

            {/* Current Step Content */}
            <div className="step-content">
                {currentStepComponent && React.createElement(currentStepComponent, {
                    currentStep,
                    setCurrentStep,
                    omeroConnection,
                    setOmeroConnection,
                    workflowConfig,
                    setWorkflowConfig,
                    pipelineId,
                    setPipelineId
                })}
            </div>
        </div>
    );
}

// Step 1: OMERO Connection
function OMEROConnectionStep({ setCurrentStep, setOmeroConnection }) {
    const [connectionStatus, setConnectionStatus] = useState('checking');

    useEffect(() => {
        checkOMEROConnection();
    }, []);

    const checkOMEROConnection = async () => {
        try {
            const response = await fetch('/omero_annotate_web/api/omero/connection/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': getCsrfToken()
                }
            });

            const data = await response.json();

            if (data.success) {
                setConnectionStatus('connected');
                setOmeroConnection(data);
            } else {
                setConnectionStatus('disconnected');
            }
        } catch (error) {
            console.error('Connection check failed:', error);
            setConnectionStatus('error');
        }
    };

    const proceedToNext = () => {
        if (connectionStatus === 'connected') {
            setCurrentStep(2);
        }
    };

    return (
        <div className="omero-connection-step">
            <h2>🔌 OMERO Server Connection</h2>
            <p>Checking your OMERO connection status...</p>

            <div className="connection-status-card">
                {connectionStatus === 'checking' && (
                    <div className="alert alert-info">
                        <div className="spinner-border spinner-border-sm me-2"></div>
                        Checking OMERO connection...
                    </div>
                )}

                {connectionStatus === 'connected' && (
                    <div className="alert alert-success">
                        <strong>✅ Connected to OMERO!</strong>
                        <p className="mb-0">You are successfully connected to OMERO.</p>
                    </div>
                )}

                {connectionStatus === 'disconnected' && (
                    <div className="alert alert-warning">
                        <strong>⚠️ No OMERO Connection</strong>
                        <p className="mb-0">Please log into OMERO.web first.</p>
                    </div>
                )}

                {connectionStatus === 'error' && (
                    <div className="alert alert-danger">
                        <strong>❌ Connection Error</strong>
                        <p className="mb-0">Failed to check OMERO connection.</p>
                    </div>
                )}
            </div>

            <div className="step-actions mt-4">
                <button
                    type="button"
                    className="btn btn-primary"
                    onClick={proceedToNext}
                    disabled={connectionStatus !== 'connected'}
                >
                    Continue to Workflow Setup →
                </button>

                <button
                    type="button"
                    className="btn btn-outline-secondary ms-2"
                    onClick={checkOMEROConnection}
                >
                    🔄 Refresh
                </button>
            </div>
        </div>
    );
}

// Step 2: Workflow Setup
function WorkflowSetupStep({ setCurrentStep, setWorkflowConfig }) {
    const [config, setConfig] = useState({
        name: 'my_annotation_project',
        container_type: 'dataset',
        container_id: '',
        model_type: 'vit_b_lm',
        output_directory: './omero_annotations',
        read_only_mode: true
    });

    const [containers, setContainers] = useState([]);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        if (config.container_type) {
            loadContainers(config.container_type);
        }
    }, [config.container_type]);

    const loadContainers = async (containerType) => {
        setLoading(true);
        try {
            const response = await fetch(`/omero_annotate_web/api/omero/containers/${containerType}s/`);
            const data = await response.json();

            if (data.success) {
                setContainers(data.containers);
            }
        } catch (error) {
            console.error('Failed to load containers:', error);
        }
        setLoading(false);
    };

    const handleConfigChange = (field, value) => {
        setConfig(prev => ({ ...prev, [field]: value }));
    };

    const proceedToNext = () => {
        setWorkflowConfig(config);
        setCurrentStep(3);
    };

    return (
        <div className="workflow-setup-step">
            <h2>🔬 Annotation Workflow Setup</h2>
            <p>Configure your annotation workflow parameters.</p>

            <form className="workflow-config-form">
                <div className="row">
                    <div className="col-md-6">
                        <div className="mb-3">
                            <label className="form-label">Training Set Name</label>
                            <input
                                type="text"
                                className="form-control"
                                value={config.name}
                                onChange={(e) => handleConfigChange('name', e.target.value)}
                            />
                        </div>

                        <div className="mb-3">
                            <label className="form-label">Container Type</label>
                            <select
                                className="form-select"
                                value={config.container_type}
                                onChange={(e) => handleConfigChange('container_type', e.target.value)}
                            >
                                <option value="dataset">Dataset</option>
                                <option value="project">Project</option>
                                <option value="screen">Screen</option>
                                <option value="plate">Plate</option>
                            </select>
                        </div>

                        <div className="mb-3">
                            <label className="form-label">Container</label>
                            <select
                                className="form-select"
                                value={config.container_id}
                                onChange={(e) => handleConfigChange('container_id', e.target.value)}
                                disabled={loading}
                            >
                                <option value="">Select a {config.container_type}</option>
                                {containers.map(container => (
                                    <option key={container.id} value={container.id}>
                                        {container.name}
                                    </option>
                                ))}
                            </select>
                            {loading && <small className="text-muted">Loading...</small>}
                        </div>
                    </div>

                    <div className="col-md-6">
                        <div className="mb-3">
                            <label className="form-label">Model Type</label>
                            <select
                                className="form-select"
                                value={config.model_type}
                                onChange={(e) => handleConfigChange('model_type', e.target.value)}
                            >
                                <option value="vit_b_lm">ViT-B (Lightweight)</option>
                                <option value="vit_l_lm">ViT-L (Large)</option>
                                <option value="vit_h_lm">ViT-H (Huge)</option>
                            </select>
                        </div>

                        <div className="mb-3">
                            <label className="form-label">Output Directory</label>
                            <input
                                type="text"
                                className="form-control"
                                value={config.output_directory}
                                onChange={(e) => handleConfigChange('output_directory', e.target.value)}
                            />
                        </div>

                        <div className="mb-3">
                            <div className="form-check">
                                <input
                                    type="checkbox"
                                    className="form-check-input"
                                    checked={config.read_only_mode}
                                    onChange={(e) => handleConfigChange('read_only_mode', e.target.checked)}
                                />
                                <label className="form-check-label">
                                    Read-only Mode (save locally instead of uploading to OMERO)
                                </label>
                            </div>
                        </div>
                    </div>
                </div>
            </form>

            <div className="step-actions mt-4">
                <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={() => setCurrentStep(1)}
                >
                    ← Back
                </button>

                <button
                    type="button"
                    className="btn btn-primary ms-2"
                    onClick={proceedToNext}
                    disabled={!config.container_id}
                >
                    Continue to Review →
                </button>
            </div>
        </div>
    );
}

// Step 3: Configuration Review
function ConfigReviewStep({ setCurrentStep, workflowConfig, setPipelineId }) {
    const createPipeline = async () => {
        try {
            const response = await fetch('/omero_annotate_web/api/pipeline/create/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': getCsrfToken()
                },
                body: JSON.stringify(workflowConfig)
            });

            const data = await response.json();

            if (data.success) {
                setPipelineId(data.pipeline_id);
                setCurrentStep(4);
            } else {
                alert('Failed to create pipeline: ' + data.message);
            }
        } catch (error) {
            console.error('Pipeline creation failed:', error);
            alert('Failed to create pipeline');
        }
    };

    return (
        <div className="config-review-step">
            <h2>📋 Configuration Review</h2>
            <p>Review your settings before running the annotation pipeline.</p>

            <div className="config-summary">
                <div className="row">
                    <div className="col-md-6">
                        <h5>📁 Data Settings</h5>
                        <ul className="list-unstyled">
                            <li><strong>Container:</strong> {workflowConfig.container_type} (ID: {workflowConfig.container_id})</li>
                            <li><strong>Training Set:</strong> {workflowConfig.name}</li>
                            <li><strong>Output:</strong> {workflowConfig.output_directory}</li>
                            <li><strong>Read-only Mode:</strong> {workflowConfig.read_only_mode ? 'Yes' : 'No'}</li>
                        </ul>
                    </div>
                    <div className="col-md-6">
                        <h5>🤖 Model Settings</h5>
                        <ul className="list-unstyled">
                            <li><strong>Model:</strong> {workflowConfig.model_type}</li>
                            <li><strong>Mode:</strong> {workflowConfig.read_only_mode ? 'Local Export' : 'OMERO Upload'}</li>
                        </ul>
                    </div>
                </div>
            </div>

            <div className="step-actions mt-4">
                <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={() => setCurrentStep(2)}
                >
                    ← Back to Configuration
                </button>

                <button
                    type="button"
                    className="btn btn-success ms-2"
                    onClick={createPipeline}
                >
                    Create Pipeline →
                </button>
            </div>
        </div>
    );
}

// Step 4: Pipeline Runner
function PipelineRunnerStep({ setCurrentStep, pipelineId }) {
    const [launching, setLaunching] = useState(false);
    const [launchResult, setLaunchResult] = useState(null);

    const launchAnnotation = async () => {
        setLaunching(true);
        try {
            const response = await fetch('/omero_annotate_web/api/annotation/launch/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': getCsrfToken()
                },
                body: JSON.stringify({ pipeline_id: pipelineId })
            });

            const data = await response.json();
            setLaunchResult(data);
        } catch (error) {
            console.error('Launch failed:', error);
            setLaunchResult({ success: false, message: 'Launch failed' });
        }
        setLaunching(false);
    };

    return (
        <div className="pipeline-runner-step">
            <h2>🚀 Run Annotation Pipeline</h2>
            <p>Ready to start the annotation workflow. This will launch napari for interactive annotation.</p>

            <div className="pipeline-info">
                <div className="alert alert-info">
                    <strong>ℹ️ Note:</strong> This will launch napari on your local machine for interactive annotation.
                    Make sure you have napari and micro-sam installed.
                </div>
            </div>

            {launchResult && (
                <div className={`alert ${launchResult.success ? 'alert-success' : 'alert-danger'}`}>
                    <strong>{launchResult.success ? '✅' : '❌'} {launchResult.message}</strong>
                </div>
            )}

            <div className="step-actions mt-4">
                <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={() => setCurrentStep(3)}
                    disabled={launching}
                >
                    ← Back to Review
                </button>

                <button
                    type="button"
                    className="btn btn-success ms-2"
                    onClick={launchAnnotation}
                    disabled={launching}
                >
                    {launching ? (
                        <>
                            <span className="spinner-border spinner-border-sm me-2"></span>
                            Launching...
                        </>
                    ) : (
                        '🚀 Launch Pipeline'
                    )}
                </button>
            </div>
        </div>
    );
}

// Utility function to get CSRF token
function getCsrfToken() {
    return document.querySelector('[name=csrfmiddlewaretoken]')?.value || '';
}