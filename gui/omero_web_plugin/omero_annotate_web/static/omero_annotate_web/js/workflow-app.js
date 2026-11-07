/**
 * Workflow App Controller
 * Main application controller for the multi-step workflow
 */

class WorkflowApp {
    constructor() {
        this.currentStep = 1;
        this.totalSteps = 4;

        this.prevBtn = $('#btn-previous');
        this.nextBtn = $('#btn-next');
        this.statusMessage = $('#status-message');

        // Component instances
        this.containerSelector = null;
        this.tableManager = null;
        this.configEditor = null;
        this.yamlPreview = null;
    }

    init() {
        // Initialize all components
        this.containerSelector = new ContainerSelector();
        this.tableManager = new TableManager();
        this.configEditor = new ConfigEditor();
        this.yamlPreview = new YamlPreview();

        // Setup navigation
        this.prevBtn.on('click', () => this.previousStep());
        this.nextBtn.on('click', () => this.nextStep());

        this.updateNavigation();
    }

    showStep(stepNumber) {
        // Hide all steps
        $('.workflow-step').hide();

        // Show current step
        $(`#step-${stepNumber}`).show();

        // Update progress indicator
        $('.progress-step').removeClass('active');
        $(`.progress-step[data-step="${stepNumber}"]`).addClass('active');

        this.currentStep = stepNumber;
        this.updateNavigation();
    }

    updateNavigation() {
        // Show/hide previous button
        if (this.currentStep === 1) {
            this.prevBtn.hide();
        } else {
            this.prevBtn.show();
        }

        // Update next button text
        if (this.currentStep === this.totalSteps) {
            this.nextBtn.hide(); // Hide on final step
        } else {
            this.nextBtn.show();
            this.nextBtn.text('Next');
        }
    }

    previousStep() {
        if (this.currentStep > 1) {
            this.showStep(this.currentStep - 1);
        }
    }

    async nextStep() {
        // Validate current step before proceeding
        if (!await this.validateCurrentStep()) {
            return;
        }

        if (this.currentStep < this.totalSteps) {
            this.showStep(this.currentStep + 1);

            // Auto-actions for certain steps
            if (this.currentStep === 2) {
                // Auto-scan tables when entering step 2
                this.tableManager.scanTables();
            }
        }
    }

    async validateCurrentStep() {
        switch (this.currentStep) {
            case 1: // Container selection
                if (!this.containerSelector.isValid()) {
                    this.showMessage('Please select a container and wait for info to load', 'error');
                    return false;
                }
                return true;

            case 2: // Table management
                if (!this.tableManager.isValid()) {
                    this.showMessage('Please select or create a table', 'error');
                    return false;
                }
                return true;

            case 3: // Configuration
                if (!this.configEditor.isValid()) {
                    this.showMessage('Please check your configuration settings', 'error');
                    return false;
                }
                return true;

            case 4: // Save
                return true;

            default:
                return true;
        }
    }

    showMessage(message, type = 'info') {
        this.statusMessage.removeClass('message-success message-error message-info');
        this.statusMessage.addClass(`message-${type}`);
        this.statusMessage.find('.message-text').text(message);
        this.statusMessage.slideDown();

        // Auto-hide after 5 seconds
        setTimeout(() => {
            this.statusMessage.slideUp();
        }, 5000);
    }

    getAllConfigData() {
        const containerData = this.containerSelector.getData();
        const tableData = this.tableManager.getData();
        const configData = this.configEditor.getData();

        return {
            ...containerData,
            ...tableData,
            ...configData
        };
    }
}
