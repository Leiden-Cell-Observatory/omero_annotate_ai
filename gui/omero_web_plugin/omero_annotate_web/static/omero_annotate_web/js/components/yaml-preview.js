/**
 * YAML Preview Component
 * Handles YAML preview generation and configuration saving
 */

class YamlPreview {
    constructor() {
        this.yamlContent = null;
        this.fileAnnotationId = null;

        this.previewBtn = $('#btn-preview-yaml');
        this.saveBtn = $('#btn-save-config');
        this.newConfigBtn = $('#btn-new-config');
        this.previewSection = $('#yaml-preview-section');
        this.previewTextarea = $('#yaml-preview');
        this.saveResult = $('#save-result');

        this.init();
    }

    init() {
        this.previewBtn.on('click', () => this.previewYaml());
        this.saveBtn.on('click', () => this.saveConfig());
        this.newConfigBtn.on('click', () => this.createNewConfig());
    }

    async previewYaml() {
        try {
            // Gather all configuration data
            const containerData = window.workflowApp.containerSelector.getData();
            const tableData = window.workflowApp.tableManager.getData();
            const configData = window.workflowApp.configEditor.getData();

            const fullConfig = {
                ...containerData,
                ...tableData,
                ...configData
            };

            // Request YAML preview
            const data = await window.apiClient.previewYaml(fullConfig);

            if (data.success) {
                this.yamlContent = data.yaml;
                this.previewTextarea.val(this.yamlContent);
                this.previewSection.show();
                window.workflowApp.showMessage('Configuration preview generated', 'success');
            }
        } catch (error) {
            window.workflowApp.showMessage('Error previewing YAML: ' + error.message, 'error');
        }
    }

    async saveConfig() {
        if (!this.yamlContent) {
            window.workflowApp.showMessage('Please preview the configuration first', 'error');
            return;
        }

        try {
            // Disable button to prevent double-clicking
            this.saveBtn.prop('disabled', true).text('Saving...');

            // Gather all configuration data
            const containerData = window.workflowApp.containerSelector.getData();
            const tableData = window.workflowApp.tableManager.getData();
            const configData = window.workflowApp.configEditor.getData();

            const fullConfig = {
                ...containerData,
                ...tableData,
                ...configData
            };

            // Save configuration
            const data = await window.apiClient.saveConfig(fullConfig);

            if (data.success) {
                this.fileAnnotationId = data.file_annotation_id;
                this.displaySaveResult(data);
                window.workflowApp.showMessage('Configuration saved successfully!', 'success');
            }
        } catch (error) {
            window.workflowApp.showMessage('Error saving configuration: ' + error.message, 'error');
            this.saveBtn.prop('disabled', false).text('Save Configuration to OMERO');
        }
    }

    displaySaveResult(data) {
        $('#file-ann-id').text(data.file_annotation_id);
        $('#download-config-link').attr('href', data.download_url);

        this.previewSection.hide();
        this.saveResult.show();
    }

    createNewConfig() {
        // Reset the workflow to start over
        window.location.reload();
    }
}
