/**
 * Container Selector Component
 * Handles container type selection and container listing
 */

class ContainerSelector {
    constructor() {
        this.selectedType = null;
        this.selectedId = null;
        this.metadata = null;

        this.typeSelect = $('#container-type');
        this.containerList = $('#container-list');
        this.refreshBtn = $('#btn-refresh-containers');
        this.infoSection = $('#container-info');

        this.init();
    }

    init() {
        // Event listeners
        this.typeSelect.on('change', () => this.onTypeChange());
        this.containerList.on('change', () => this.onContainerChange());
        this.refreshBtn.on('click', () => this.loadContainers());
    }

    async onTypeChange() {
        this.selectedType = this.typeSelect.val();

        if (!this.selectedType) {
            this.containerList.prop('disabled', true);
            this.containerList.html('<option value="">-- Select container type first --</option>');
            this.infoSection.hide();
            return;
        }

        this.containerList.prop('disabled', false);
        await this.loadContainers();
    }

    async loadContainers() {
        if (!this.selectedType) return;

        try {
            this.containerList.html('<option value="">Loading...</option>');

            const data = await window.apiClient.listContainers(this.selectedType);

            if (data.success && data.containers.length > 0) {
                let options = '<option value="">-- Select a container --</option>';
                data.containers.forEach(container => {
                    options += `<option value="${container.id}">${container.name} (ID: ${container.id})</option>`;
                });
                this.containerList.html(options);
            } else {
                this.containerList.html('<option value="">No containers found</option>');
            }
        } catch (error) {
            this.containerList.html('<option value="">Error loading containers</option>');
            window.workflowApp.showMessage('Error loading containers: ' + error.message, 'error');
        }
    }

    async onContainerChange() {
        this.selectedId = parseInt(this.containerList.val());

        if (!this.selectedId) {
            this.infoSection.hide();
            return;
        }

        await this.loadContainerInfo();
    }

    async loadContainerInfo() {
        if (!this.selectedType || !this.selectedId) return;

        try {
            const data = await window.apiClient.getContainerInfo(this.selectedType, this.selectedId);

            if (data.success) {
                this.metadata = data.metadata;
                this.displayContainerInfo();
            }
        } catch (error) {
            window.workflowApp.showMessage('Error loading container info: ' + error.message, 'error');
        }
    }

    displayContainerInfo() {
        if (!this.metadata) return;

        $('#info-name').text(this.metadata.name || 'N/A');
        $('#info-id').text(this.metadata.id || 'N/A');
        $('#info-description').text(this.metadata.description || 'No description');
        $('#info-image-count').text(this.metadata.image_count || '0');

        if (this.metadata.size_x) {
            const dims = `${this.metadata.size_x} × ${this.metadata.size_y} × ${this.metadata.size_z || 1}`;
            $('#info-dimensions').text(dims);
        } else {
            $('#info-dimensions').text('N/A');
        }

        $('#info-channels').text(this.metadata.size_c || 'N/A');
        $('#info-timepoints').text(this.metadata.size_t || 'N/A');

        this.infoSection.show();

        // Update slider maximums based on image count
        const imageCount = this.metadata.image_count || 10;
        $('#train-n').attr('max', imageCount);
        $('#train-n-slider').attr('max', imageCount);
        $('#validate-n').attr('max', imageCount);
        $('#validate-n-slider').attr('max', imageCount);
    }

    isValid() {
        return this.selectedType && this.selectedId && this.metadata;
    }

    getData() {
        return {
            container_type: this.selectedType,
            container_id: this.selectedId,
            metadata: this.metadata
        };
    }
}
