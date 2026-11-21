"""The New training set form shared by the Jupyter widgets and the napari dock."""

import pytest

from omero_annotate_ai.forms import (
    NewSetForm,
    Planes,
    channel_rows,
    matches_search,
    move_label,
    well_condition_keys,
    well_value_choices,
    wells_kept,
)


@pytest.mark.unit
class TestNewSetForm:
    def test_the_form_becomes_a_config(self):
        rows = channel_rows(["DAPI", "GFP", "Actin"], label=1)
        rows[0].train = True
        rows[2].show, rows[2].colormap = True, "red"
        config = NewSetForm(
            name=" nuclei ",
            container_type="plate",
            container_ids=[4, 5],
            label_channel=1,
            channels=rows,
            timepoints=Planes("random", n=3),
            z_slices=Planes("specific", "2, 5-6"),
            patch_size=256,
            patches_per_image=2,
            counts={"training": 4, "validation": 1, "test": 1},
            seed="7",
            storage="local",
            model="vit_l_lm",
            batch_size=3,
            well_filters={"CellLine": ["HeLa"]},
            well_filter_mode="exclude",
        ).to_config()

        assert config.name == "nuclei"
        assert (config.omero.container_id, config.omero.container_ids) == (4, [4, 5])
        assert config.omero.well_filters == {"CellLine": ["HeLa"]}
        assert config.omero.well_filter_mode == "exclude"
        assert config.label_channel() == 1
        assert config.training_channels() == [0, 1]
        assert config.display.specs() == [(2, "Actin", "red")]
        assert config.display.label_colormap == "gray"  # the annotated row's default
        assert config.sampling.timepoints.mode == "random"
        assert config.sampling.z_slices.values == [2, 5, 6]
        assert config.sampling.patches.width == 256
        assert (config.training.train_n, config.training.test_n) == (4, 1)
        assert config.split.seed == 7 and config.storage == "local"
        assert config.ai_model.pretrained_from == "vit_l_lm"
        assert config.workflow.batch_size == 3

    def test_one_container_and_defaults(self):
        config = NewSetForm(container_ids=[9]).to_config()
        assert (config.omero.container_id, config.omero.container_ids) == (9, None)
        assert config.sampling.input_channels is None  # the annotated channel
        assert config.display.label_colormap == "gray"
        assert config.omero.well_filters is None  # datasets have no wells
        assert config.split.seed is None

    @pytest.mark.parametrize(
        ("form", "message"),
        [
            (NewSetForm(seed="abc"), "Seed"),
            (NewSetForm(timepoints=Planes("specific", "3-1")), "Timepoints"),
            (NewSetForm(channels=channel_rows(["a"], label=5)), "at least one"),
        ],
    )
    def test_a_bad_field_is_named(self, form, message):
        with pytest.raises(ValueError, match=message):
            form.to_config()


@pytest.mark.unit
class TestPickerLogic:
    def test_search_matches_name_or_exact_id(self):
        assert matches_search("HeLa nuclei", 12, "nucl")
        assert matches_search("HeLa nuclei", 12, "12")
        assert not matches_search("HeLa nuclei", 123, "12")
        assert matches_search("anything", 1, "  ")

    def test_the_annotated_channel_has_a_colour_too(self):
        from omero_annotate_ai.forms import display_config

        rows = channel_rows(["DAPI", "GFP"], label=1)
        assert [r.colormap for r in rows] == ["blue", "gray"]
        rows[1].colormap = "green"
        assert display_config(rows, 1).label_colormap == "green"

    def test_training_follows_the_annotated_channel_unless_changed(self):
        rows = channel_rows(["a", "b", "c"])
        rows[1].show = True
        move_label(rows, 0, 1)
        assert [r.train for r in rows] == [False, True, False]
        assert not rows[1].show  # the annotated channel is not also context
        assert [r.colormap for r in rows] == ["blue", "gray", "green"]  # gray moves
        rows[2].train = True  # chosen by hand: kept when the label moves
        rows[1].colormap = "red"
        move_label(rows, 1, 0)
        assert [r.train for r in rows] == [False, True, True]
        assert [r.colormap for r in rows] == ["blue", "red", "green"]  # picked: kept


@pytest.mark.unit
class TestWellLogic:
    WELLS = {
        i: {"CellLine": "HeLa" if i % 2 else "U2OS", "Well": f"W{i}", "Other": "x"}
        for i in range(12)
    }

    def test_mihcsme_keys_first_and_well_names_left_out(self):
        keys = well_condition_keys(self.WELLS, {"CellLine"})
        assert keys == [("CellLine (MIHCSME)", "CellLine"), ("Other", "Other")]

    def test_values_with_counts_and_kept_wells(self):
        assert well_value_choices(self.WELLS, "CellLine") == [
            ("HeLa (6 wells)", "HeLa"),
            ("U2OS (6 wells)", "U2OS"),
        ]
        hela = {"CellLine": ["HeLa"]}
        assert wells_kept(self.WELLS, hela, "include") == 6
        assert wells_kept(self.WELLS, hela, "exclude") == 6
        assert wells_kept(self.WELLS, None, "include") == 12
        both = {"CellLine": ["HeLa"], "Other": ["y"]}
        assert wells_kept(self.WELLS, both, "include") == 0
