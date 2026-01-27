"""Tests for TrainingConfig conditional validation based on segment_all mode.

These tests verify that validation correctly enforces constraints based on the
segment_all flag, catching invalid configurations that would cause runtime errors.
"""

import pytest
from pydantic import ValidationError
from omero_annotate_ai.core.annotation_config import TrainingConfig


class TestFractionModeValidation:
    """Test validation when segment_all=True (fraction-based mode)."""

    def test_fractions_exceeding_one_raises_error(self):
        """Fractions summing to > 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="must sum to <= 1.0"):
            TrainingConfig(
                segment_all=True,
                train_fraction=0.6,
                validation_fraction=0.5,
                test_fraction=0.0,  # Total = 1.1
            )

    def test_fractions_slightly_over_one_raises_error(self):
        """Even small overages should be caught."""
        with pytest.raises(ValueError, match="must sum to <= 1.0"):
            TrainingConfig(
                segment_all=True,
                train_fraction=0.34,
                validation_fraction=0.33,
                test_fraction=0.34,  # Total = 1.01
            )

    def test_all_fractions_at_maximum_raises_error(self):
        """All fractions at 1.0 should fail (total = 3.0)."""
        with pytest.raises(ValueError, match="must sum to <= 1.0"):
            TrainingConfig(
                segment_all=True,
                train_fraction=1.0,
                validation_fraction=1.0,
                test_fraction=1.0,
            )

    def test_train_n_zero_allowed_in_fraction_mode(self):
        """train_n=0 should NOT raise error when segment_all=True."""
        # This would fail with old validation that always required train_n >= 1
        config = TrainingConfig(
            segment_all=True,
            train_n=0,
            validate_n=0,
            test_n=0,
        )
        assert config.train_n == 0

    def test_fractions_exactly_one_allowed(self):
        """Fractions summing to exactly 1.0 should be valid."""
        config = TrainingConfig(
            segment_all=True,
            train_fraction=0.6,
            validation_fraction=0.3,
            test_fraction=0.1,
        )
        total = config.train_fraction + config.validation_fraction + config.test_fraction
        assert abs(total - 1.0) < 0.001

    def test_fractions_less_than_one_allowed(self):
        """Fractions summing to < 1.0 should be valid (remainder unassigned)."""
        config = TrainingConfig(
            segment_all=True,
            train_fraction=0.5,
            validation_fraction=0.3,
            test_fraction=0.0,
        )
        total = config.train_fraction + config.validation_fraction + config.test_fraction
        assert total < 1.0

    def test_inference_only_workflow_allowed(self):
        """train_fraction=0 should be valid for inference-only workflows."""
        config = TrainingConfig(
            segment_all=True,
            train_fraction=0.0,
            validation_fraction=1.0,
            test_fraction=0.0,
        )
        assert config.train_fraction == 0.0

    def test_train_n_negative_raises_pydantic_error_fraction_mode(self):
        """Negative train_n should be caught by Pydantic even in fraction mode."""
        with pytest.raises(ValidationError):
            TrainingConfig(
                segment_all=True,
                train_n=-1,
            )


class TestCountModeValidation:
    """Test validation when segment_all=False (count-based mode)."""

    def test_train_n_zero_raises_error(self):
        """train_n=0 should raise ValueError when segment_all=False."""
        with pytest.raises(ValueError, match="train_n must be at least 1"):
            TrainingConfig(
                segment_all=False,
                train_n=0,
                validate_n=2,
                test_n=0,
            )

    def test_train_n_negative_raises_pydantic_error(self):
        """Negative train_n should be caught by Pydantic's ge=0 constraint."""
        with pytest.raises(ValidationError):
            TrainingConfig(
                segment_all=False,
                train_n=-1,
            )

    def test_fractions_over_one_ignored_in_count_mode(self):
        """Invalid fractions should NOT raise error when segment_all=False.

        In count mode, fractions are ignored, so even invalid fractions
        should not cause validation errors.
        """
        # This would fail if fraction validation ran in count mode
        config = TrainingConfig(
            segment_all=False,
            train_n=3,
            train_fraction=0.9,
            validation_fraction=0.9,
            test_fraction=0.9,  # Total = 2.7, but should be ignored
        )
        assert config.train_n == 3

    def test_train_n_one_is_minimum(self):
        """train_n=1 should be the minimum valid value."""
        config = TrainingConfig(
            segment_all=False,
            train_n=1,
        )
        assert config.train_n == 1


class TestModeSwitch:
    """Test that switching segment_all changes which validation applies."""

    def test_same_config_valid_in_fraction_mode_invalid_in_count_mode(self):
        """train_n=0 valid with segment_all=True, invalid with segment_all=False."""
        # Valid in fraction mode
        config = TrainingConfig(
            segment_all=True,
            train_n=0,
        )
        assert config.train_n == 0

        # Invalid in count mode
        with pytest.raises(ValueError, match="train_n must be at least 1"):
            TrainingConfig(
                segment_all=False,
                train_n=0,
            )

    def test_invalid_fractions_only_caught_in_fraction_mode(self):
        """Fraction validation only runs when segment_all=True."""
        # Invalid fractions should fail in fraction mode
        with pytest.raises(ValueError, match="must sum to <= 1.0"):
            TrainingConfig(
                segment_all=True,
                train_fraction=0.8,
                validation_fraction=0.8,
            )

        # Same invalid fractions should pass in count mode (ignored)
        config = TrainingConfig(
            segment_all=False,
            train_n=5,
            train_fraction=0.8,
            validation_fraction=0.8,
        )
        assert config.train_n == 5


class TestDefaultBehavior:
    """Test that defaults work correctly with validation."""

    def test_default_config_is_valid(self):
        """Default TrainingConfig should pass all validation."""
        config = TrainingConfig()
        assert config.segment_all is False
        assert config.train_n == 3
        assert config.train_fraction == 0.7

    def test_default_is_count_mode(self):
        """Default segment_all=False means count-based validation applies."""
        # Default config uses count mode, so train_n >= 1 is required
        # but default train_n=3 satisfies this
        config = TrainingConfig()
        assert config.segment_all is False
        assert config.train_n >= 1
