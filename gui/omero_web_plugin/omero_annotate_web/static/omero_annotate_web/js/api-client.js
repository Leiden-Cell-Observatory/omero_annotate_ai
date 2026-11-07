/**
 * API Client for OMERO Annotation Web Plugin
 *
 * Provides wrapper functions for all backend API endpoints with
 * proper error handling and CSRF token management.
 */

class ApiClient {
    constructor() {
        this.baseUrl = '/omero_annotate_web';
        this.csrfToken = this.getCsrfToken();
    }

    /**
     * Get CSRF token from cookie
     */
    getCsrfToken() {
        const name = 'csrftoken';
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }

    /**
     * Generic API request handler
     */
    async request(url, options = {}) {
        const headers = {
            'X-CSRFToken': this.csrfToken,
            ...options.headers
        };

        if (options.body && typeof options.body === 'object') {
            headers['Content-Type'] = 'application/json';
            options.body = JSON.stringify(options.body);
        }

        try {
            const response = await fetch(url, {
                ...options,
                headers,
                credentials: 'same-origin'
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.message || `HTTP error! status: ${response.status}`);
            }

            return data;
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    }

    /**
     * List containers of a specific type
     */
    async listContainers(containerType) {
        return this.request(`${this.baseUrl}/api/containers/${containerType}/`);
    }

    /**
     * Get detailed container information
     */
    async getContainerInfo(containerType, containerId) {
        return this.request(`${this.baseUrl}/api/container/${containerType}/${containerId}/info/`);
    }

    /**
     * Scan for existing annotation tables
     */
    async scanTables(containerType, containerId) {
        return this.request(`${this.baseUrl}/api/tables/scan/`, {
            method: 'POST',
            body: { container_type: containerType, container_id: containerId }
        });
    }

    /**
     * Generate unique table name
     */
    async generateTableName(containerType, containerId, baseName = 'annotation_table') {
        return this.request(`${this.baseUrl}/api/tables/generate-name/`, {
            method: 'POST',
            body: {
                container_type: containerType,
                container_id: containerId,
                base_name: baseName
            }
        });
    }

    /**
     * Validate configuration
     */
    async validateConfig(configData) {
        return this.request(`${this.baseUrl}/api/config/validate/`, {
            method: 'POST',
            body: configData
        });
    }

    /**
     * Preview YAML configuration
     */
    async previewYaml(configData) {
        return this.request(`${this.baseUrl}/api/config/preview/`, {
            method: 'POST',
            body: configData
        });
    }

    /**
     * Save configuration to OMERO
     */
    async saveConfig(configData) {
        return this.request(`${this.baseUrl}/api/config/save/`, {
            method: 'POST',
            body: configData
        });
    }
}

// Create global instance
window.apiClient = new ApiClient();
