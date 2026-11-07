/**
 * Table Manager Component
 * Handles annotation table scanning and selection
 */

class TableManager {
    constructor() {
        this.existingTables = [];
        this.selectedTable = null;
        this.tableName = null;

        this.scanBtn = $('#btn-scan-tables');
        this.generateNameBtn = $('#btn-generate-name');
        this.tablesList = $('#tables-list');
        this.tableNameInput = $('#table-name-input');
        this.selectedTableInfo = $('#selected-table-info');

        this.init();
    }

    init() {
        // Event listeners
        this.scanBtn.on('click', () => this.scanTables());
        this.generateNameBtn.on('click', () => this.generateTableName());
        $('input[name="table-option"]').on('change', (e) => this.onTableOptionChange(e));
        this.tableNameInput.on('input', () => this.updateSelectedTable());
    }

    async scanTables() {
        const containerData = window.workflowApp.containerSelector.getData();

        if (!containerData || !containerData.container_id) {
            window.workflowApp.showMessage('Please select a container first', 'error');
            return;
        }

        try {
            const data = await window.apiClient.scanTables(
                containerData.container_type,
                containerData.container_id
            );

            if (data.success) {
                this.existingTables = data.tables || [];
                this.displayTables();
                window.workflowApp.showMessage(`Found ${this.existingTables.length} table(s)`, 'success');
            }
        } catch (error) {
            window.workflowApp.showMessage('Error scanning tables: ' + error.message, 'error');
        }
    }

    displayTables() {
        $('#existing-tables').show();

        if (this.existingTables.length === 0) {
            this.tablesList.html('<p class="info-text">No existing tables found.</p>');
            return;
        }

        let html = '<div class="table-items">';
        this.existingTables.forEach(table => {
            html += `
                <div class="table-item" data-table-id="${table.id}">
                    <label class="radio-label">
                        <input type="radio" name="existing-table" value="${table.id}">
                        <strong>${table.name}</strong>
                        <span class="table-desc">${table.description || ''}</span>
                    </label>
                </div>
            `;
        });
        html += '</div>';

        this.tablesList.html(html);

        // Add event listener for table selection
        $('input[name="existing-table"]').on('change', (e) => {
            const tableId = parseInt($(e.target).val());
            this.selectedTable = this.existingTables.find(t => t.id === tableId);
            this.updateSelectedTable();
        });
    }

    async generateTableName() {
        const containerData = window.workflowApp.containerSelector.getData();

        if (!containerData || !containerData.container_id) {
            window.workflowApp.showMessage('Please select a container first', 'error');
            return;
        }

        const baseName = this.tableNameInput.val() || 'annotation_table';

        try {
            const data = await window.apiClient.generateTableName(
                containerData.container_type,
                containerData.container_id,
                baseName
            );

            if (data.success) {
                this.tableNameInput.val(data.table_name);
                this.updateSelectedTable();
                window.workflowApp.showMessage('Generated unique table name', 'success');
            }
        } catch (error) {
            window.workflowApp.showMessage('Error generating table name: ' + error.message, 'error');
        }
    }

    onTableOptionChange(e) {
        const option = $(e.target).val();

        if (option === 'new') {
            $('#new-table-section').show();
            $('input[name="existing-table"]').prop('checked', false);
            this.selectedTable = null;
        } else {
            $('#new-table-section').hide();
        }

        this.updateSelectedTable();
    }

    updateSelectedTable() {
        const option = $('input[name="table-option"]:checked').val();

        if (option === 'new') {
            this.tableName = this.tableNameInput.val() || 'annotation_table';
            this.selectedTableInfo.show();
            $('#selected-table-name').text(this.tableName + ' (new)');
            $('#resume-from-table').prop('checked', false);
        } else if (option === 'existing' && this.selectedTable) {
            this.tableName = this.selectedTable.name;
            this.selectedTableInfo.show();
            $('#selected-table-name').text(this.tableName + ' (existing)');
            $('#resume-from-table').prop('checked', true);
        } else {
            this.selectedTableInfo.hide();
        }
    }

    isValid() {
        return this.tableName !== null && this.tableName !== '';
    }

    getData() {
        const option = $('input[name="table-option"]:checked').val();
        return {
            table_name: this.tableName,
            resume_from_table: option === 'existing'
        };
    }
}
