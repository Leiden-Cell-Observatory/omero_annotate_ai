/**
 * Config Editor Component
 * Handles all configuration form inputs with dynamic enable/disable logic
 */

class ConfigEditor {
    constructor() {
        this.init();
    }

    init() {
        // Tab switching
        $('.tab-button').on('click', (e) => {
            const tab = $(e.target).data('tab');
            this.switchTab(tab);
        });

        // Segment all checkbox
        $('#segment-all').on('change', (e) => {
            const checked = $(e.target).is(':checked');
            $('#image-selection-section').toggle(!checked);
        });

        // Slider synchronization
        this.syncSlider('#train-n', '#train-n-slider');
        this.syncSlider('#validate-n', '#validate-n-slider');

        // Timepoint mode
        $('input[name="timepoint-mode"]').on('change', (e) => {
            const mode = $(e.target).val();
            $('#timepoints').prop('disabled', mode !== 'specific');
        });

        // Z-slice mode
        $('input[name="z-mode"]').on('change', (e) => {
            const mode = $(e.target).val();
            $('#z-slices').prop('disabled', mode !== 'specific');
        });

        // 3D processing
        $('#process-3d').on('change', (e) => {
            const checked = $(e.target).is(':checked');
            $('#3d-settings').toggle(checked);
        });

        // Patch processing
        $('#use-patches').on('change', (e) => {
            const checked = $(e.target).is(':checked');
            $('#patch-settings').toggle(checked);
        });
    }

    syncSlider(inputId, sliderId) {
        $(inputId).on('input', function() {
            $(sliderId).val($(this).val());
        });

        $(sliderId).on('input', function() {
            $(inputId).val($(this).val());
        });
    }

    switchTab(tabName) {
        $('.tab-button').removeClass('active');
        $(`.tab-button[data-tab="${tabName}"]`).addClass('active');

        $('.tab-content').removeClass('active');
        $(`#tab-${tabName}`).addClass('active');
    }

    getData() {
        const config = {};

        // Annotation settings
        config.segment_all = $('#segment-all').is(':checked');

        if (!config.segment_all) {
            config.train_n = parseInt($('#train-n').val());
            config.validate_n = parseInt($('#validate-n').val());
        }

        config.channels = $('#channels').val();

        // Timepoint mode
        const timepointMode = $('input[name="timepoint-mode"]:checked').val();
        config.timepoint_mode = timepointMode;
        if (timepointMode === 'specific') {
            config.timepoints = $('#timepoints').val();
        }

        // Z-slice mode
        const zMode = $('input[name="z-mode"]:checked').val();
        config.z_mode = zMode;
        if (zMode === 'specific') {
            config.z_slices = $('#z-slices').val();
        }

        // 3D processing
        config.process_3d = $('#process-3d').is(':checked');
        if (config.process_3d) {
            config.z_range_start = parseInt($('#z-range-start').val());
            config.z_range_end = parseInt($('#z-range-end').val());
        }

        // Patch processing
        config.use_patches = $('#use-patches').is(':checked');
        if (config.use_patches) {
            config.patch_size = parseInt($('#patch-size').val());
            config.patch_overlap = parseInt($('#patch-overlap').val());
        }

        // Technical settings
        config.batch_size = parseInt($('#batch-size').val());
        config.sam_model = $('#sam-model').val();

        // Workflow settings
        config.read_only = $('#read-only').is(':checked');
        config.resume_from_table = $('#resume-from-table').is(':checked');

        return config;
    }

    isValid() {
        // Basic validation
        if (!$('#segment-all').is(':checked')) {
            const trainN = parseInt($('#train-n').val());
            const validateN = parseInt($('#validate-n').val());

            if (trainN < 1 || validateN < 1) {
                return false;
            }
        }

        return true;
    }
}
