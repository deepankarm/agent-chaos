"""Tests for TriggerConfig Pydantic model."""

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from agent_chaos.chaos.base import TriggerConfig


class TestTriggerConfigValidation:
    """Test TriggerConfig field validation."""

    def test_valid_config(self):
        """Test creating a valid config."""
        config = TriggerConfig(on_call=2, probability=0.5)
        assert config.on_call == 2
        assert config.probability == 0.5

    def test_empty_config(self):
        """Test creating empty config (all defaults)."""
        config = TriggerConfig()
        assert config.on_call is None
        assert config.after_calls is None
        assert config.probability is None
        assert config.always is False

    def test_probability_must_be_between_0_and_1(self):
        """Test probability validation."""
        # Valid values
        TriggerConfig(probability=0.0)
        TriggerConfig(probability=0.5)
        TriggerConfig(probability=1.0)

        # Invalid values
        with pytest.raises(ValidationError):
            TriggerConfig(probability=-0.1)

        with pytest.raises(ValidationError):
            TriggerConfig(probability=1.1)

    def test_on_call_must_be_positive(self):
        """Test on_call validation."""
        TriggerConfig(on_call=1)
        TriggerConfig(on_call=100)

        with pytest.raises(ValidationError):
            TriggerConfig(on_call=0)

        with pytest.raises(ValidationError):
            TriggerConfig(on_call=-1)

    def test_after_calls_must_be_positive(self):
        """Test after_calls validation."""
        TriggerConfig(after_calls=1)

        with pytest.raises(ValidationError):
            TriggerConfig(after_calls=0)

    def test_on_turn_must_be_positive(self):
        """Test on_turn validation."""
        TriggerConfig(on_turn=1)

        with pytest.raises(ValidationError):
            TriggerConfig(on_turn=0)

    def test_after_turns_must_be_positive(self):
        """Test after_turns validation."""
        TriggerConfig(after_turns=1)

        with pytest.raises(ValidationError):
            TriggerConfig(after_turns=0)

    def test_between_turns_tuple(self):
        """Test between_turns accepts tuple."""
        config = TriggerConfig(between_turns=(1, 3))
        assert config.between_turns == (1, 3)


class TestTriggerConfigShouldTrigger:
    """Test TriggerConfig.should_trigger() logic."""

    def test_always_triggers(self):
        """Test always=True triggers regardless of call number."""
        config = TriggerConfig(always=True)
        assert config.should_trigger(1) is True
        assert config.should_trigger(5) is True
        assert config.should_trigger(100) is True

    def test_on_call_triggers_at_exact_call(self):
        """Test on_call triggers only at exact call number."""
        config = TriggerConfig(on_call=3)
        assert config.should_trigger(1) is False
        assert config.should_trigger(2) is False
        assert config.should_trigger(3) is True
        assert config.should_trigger(4) is False

    def test_after_calls_triggers_after_threshold(self):
        """Test after_calls triggers after N calls."""
        config = TriggerConfig(after_calls=2)
        assert config.should_trigger(1) is False
        assert config.should_trigger(2) is False
        assert config.should_trigger(3) is True
        assert config.should_trigger(4) is True

    def test_probability_triggers_randomly(self):
        """Test probability-based triggering."""
        config = TriggerConfig(probability=0.5)

        # Mock random to control outcomes
        with patch("random.random", return_value=0.3):
            assert config.should_trigger(1) is True

        with patch("random.random", return_value=0.7):
            assert config.should_trigger(1) is False

    def test_probability_zero_never_triggers(self):
        """Test probability=0 never triggers."""
        config = TriggerConfig(probability=0.0)
        with patch("random.random", return_value=0.0):
            # Even with random=0.0, probability=0.0 should not trigger
            assert config.should_trigger(1) is False

    def test_probability_one_always_triggers(self):
        """Test probability=1.0 always triggers."""
        config = TriggerConfig(probability=1.0)
        with patch("random.random", return_value=0.99):
            assert config.should_trigger(1) is True

    def test_provider_filter_matches(self):
        """Test provider filtering when provider matches."""
        config = TriggerConfig(always=True, provider="anthropic")
        assert config.should_trigger(1, provider="anthropic") is True
        assert config.should_trigger(1, provider="openai") is False
        assert config.should_trigger(1, provider=None) is False

    def test_provider_filter_none_matches_all(self):
        """Test that provider=None matches any provider."""
        config = TriggerConfig(always=True, provider=None)
        assert config.should_trigger(1, provider="anthropic") is True
        assert config.should_trigger(1, provider="openai") is True
        assert config.should_trigger(1, provider=None) is True

    def test_on_turn_triggers_on_specific_turn(self):
        """Test on_turn triggers on specific turn."""
        config = TriggerConfig(on_turn=2)
        assert config.should_trigger(1, current_turn=1) is False
        assert config.should_trigger(1, current_turn=2) is True
        assert config.should_trigger(1, current_turn=3) is False

    def test_after_turns_triggers_after_completed(self):
        """Test after_turns triggers after N turns complete."""
        config = TriggerConfig(after_turns=2)
        assert config.should_trigger(1, completed_turns=0) is False
        assert config.should_trigger(1, completed_turns=1) is False
        assert config.should_trigger(1, completed_turns=2) is True
        assert config.should_trigger(1, completed_turns=3) is True

    def test_between_turns_triggers_in_range(self):
        """Test between_turns triggers between specified turns."""
        config = TriggerConfig(between_turns=(1, 3))
        # Should trigger after turn 1 is complete, before turn 3 starts
        # (when current_turn=0 means we're between turns)
        assert config.should_trigger(1, completed_turns=0, current_turn=0) is False
        assert config.should_trigger(1, completed_turns=1, current_turn=0) is True
        assert config.should_trigger(1, completed_turns=2, current_turn=0) is True

    def test_combined_on_turn_and_on_call(self):
        """Test combining on_turn and on_call."""
        config = TriggerConfig(on_turn=2, on_call=3)
        # Must be turn 2 AND call 3
        assert config.should_trigger(1, current_turn=2) is False
        assert config.should_trigger(3, current_turn=1) is False
        assert config.should_trigger(3, current_turn=2) is True

    def test_combined_on_turn_and_after_calls(self):
        """Test combining on_turn and after_calls."""
        config = TriggerConfig(on_turn=2, after_calls=2)
        # Must be turn 2 AND after 2 calls
        assert config.should_trigger(1, current_turn=2) is False
        assert config.should_trigger(2, current_turn=2) is False
        assert config.should_trigger(3, current_turn=2) is True
        assert config.should_trigger(3, current_turn=1) is False

    def test_default_does_not_trigger(self):
        """Test empty config never triggers."""
        config = TriggerConfig()
        assert config.should_trigger(1) is False
        assert config.should_trigger(100) is False


class TestTriggerConfigSerialization:
    """Test TriggerConfig Pydantic serialization."""

    def test_model_dump(self):
        """Test serializing to dict."""
        config = TriggerConfig(on_call=2, probability=0.5)
        data = config.model_dump()
        assert data["on_call"] == 2
        assert data["probability"] == 0.5
        assert data["always"] is False

    def test_model_json(self):
        """Test serializing to JSON."""
        config = TriggerConfig(on_call=2)
        json_str = config.model_dump_json()
        assert '"on_call":2' in json_str or '"on_call": 2' in json_str

    def test_model_from_dict(self):
        """Test creating from dict."""
        config = TriggerConfig.model_validate({"on_call": 3, "always": True})
        assert config.on_call == 3
        assert config.always is True
