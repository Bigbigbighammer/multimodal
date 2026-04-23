"""Tests for the interactive REPL module."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from interactive import (
    CommandParser,
    ParsedCommand,
    CommandType,
    InteractiveSession,
)


class TestCommandParser:
    """Test command parsing."""

    def setup_method(self):
        self.parser = CommandParser()

    def test_parse_go_command(self):
        """Test 'go' navigation command."""
        result = self.parser.parse("go to the kitchen")
        assert result.type == CommandType.NAVIGATE
        assert result.action == "go"
        assert result.target == "the kitchen"

    def test_parse_find_command(self):
        """Test 'find' navigation command."""
        result = self.parser.parse("find the chair")
        assert result.type == CommandType.NAVIGATE
        assert result.action == "find"
        assert result.target == "the chair"

    def test_parse_explore_command(self):
        """Test 'explore' command."""
        result = self.parser.parse("explore")
        assert result.type == CommandType.NAVIGATE
        assert result.action == "explore"
        assert result.target is None

    def test_parse_pickup_command(self):
        """Test 'pick up' action command."""
        result = self.parser.parse("pick up the apple")
        assert result.type == CommandType.ACTION
        assert result.action == "pickup"
        assert result.target == "the apple"

    def test_parse_grab_command(self):
        """Test 'grab' action command."""
        result = self.parser.parse("grab the mug")
        assert result.type == CommandType.ACTION
        assert result.action == "pickup"
        assert result.target == "the mug"

    def test_parse_open_command(self):
        """Test 'open' action command."""
        result = self.parser.parse("open the door")
        assert result.type == CommandType.ACTION
        assert result.action == "open"
        assert result.target == "the door"

    def test_parse_close_command(self):
        """Test 'close' action command."""
        result = self.parser.parse("close the fridge")
        assert result.type == CommandType.ACTION
        assert result.action == "close"
        assert result.target == "the fridge"

    def test_parse_look_command(self):
        """Test 'look' vision command."""
        result = self.parser.parse("look")
        assert result.type == CommandType.VISION
        assert result.action == "look"

    def test_parse_scan_command(self):
        """Test 'scan' vision command."""
        result = self.parser.parse("scan")
        assert result.type == CommandType.VISION
        assert result.action == "scan"

    def test_parse_turn_left_command(self):
        """Test 'turn left' command."""
        result = self.parser.parse("turn left")
        assert result.type == CommandType.VISION
        assert result.action == "turn"
        assert result.target == "left"

    def test_parse_turn_right_command(self):
        """Test 'turn right' command."""
        result = self.parser.parse("turn right")
        assert result.type == CommandType.VISION
        assert result.action == "turn"
        assert result.target == "right"

    def test_parse_turn_degrees_command(self):
        """Test 'turn <degrees>' command."""
        result = self.parser.parse("turn 90")
        assert result.type == CommandType.VISION
        assert result.action == "turn_deg"
        assert result.target == "90"

    def test_parse_turn_negative_degrees(self):
        """Test 'turn -45' command."""
        result = self.parser.parse("turn -45")
        assert result.type == CommandType.VISION
        assert result.action == "turn_deg"
        assert result.target == "-45"

    def test_parse_help_command(self):
        """Test 'help' command."""
        result = self.parser.parse("help")
        assert result.type == CommandType.CONTROL
        assert result.action == "help"

    def test_parse_status_command(self):
        """Test 'status' command."""
        result = self.parser.parse("status")
        assert result.type == CommandType.CONTROL
        assert result.action == "status"

    def test_parse_quit_command(self):
        """Test 'quit' command."""
        result = self.parser.parse("quit")
        assert result.type == CommandType.CONTROL
        assert result.action == "quit"

    def test_parse_exit_command(self):
        """Test 'exit' command (alias for quit)."""
        result = self.parser.parse("exit")
        assert result.type == CommandType.CONTROL
        assert result.action == "quit"

    def test_parse_reset_command(self):
        """Test 'reset' command."""
        result = self.parser.parse("reset")
        assert result.type == CommandType.CONTROL
        assert result.action == "reset"

    def test_parse_stop_command(self):
        """Test 'stop' command."""
        result = self.parser.parse("stop")
        assert result.type == CommandType.CONTROL
        assert result.action == "stop"

    def test_parse_unknown_command(self):
        """Test unknown command."""
        result = self.parser.parse("xyzzy")
        assert result.type == CommandType.UNKNOWN
        assert result.action == "unknown"

    def test_parse_case_insensitive(self):
        """Test case insensitive parsing."""
        result = self.parser.parse("FIND THE CHAIR")
        assert result.type == CommandType.NAVIGATE
        assert result.action == "find"
        assert result.target == "the chair"

    def test_parse_extra_whitespace(self):
        """Test parsing with extra whitespace."""
        result = self.parser.parse("  find   the   chair  ")
        assert result.type == CommandType.NAVIGATE
        assert result.action == "find"
        assert result.target == "the   chair"


class TestInteractiveSession:
    """Test InteractiveSession class."""

    def test_session_initialization(self):
        """Test session initializes correctly."""
        session = InteractiveSession(
            scene="FloorPlan1",
            use_llm_planner=False
        )
        assert session._scene == "FloorPlan1"
        assert session._use_llm_planner is False
        assert session._running is True

    def test_parser_available(self):
        """Test parser is available."""
        session = InteractiveSession(scene="FloorPlan1")
        assert session._parser is not None
        assert isinstance(session._parser, CommandParser)


class TestParsedCommand:
    """Test ParsedCommand dataclass."""

    def test_parsed_command_defaults(self):
        """Test default values."""
        cmd = ParsedCommand(type=CommandType.UNKNOWN, action="test")
        assert cmd.target is None
        assert cmd.args == {}

    def test_parsed_command_with_values(self):
        """Test with explicit values."""
        cmd = ParsedCommand(
            type=CommandType.NAVIGATE,
            action="go",
            target="kitchen",
            args={"key": "value"}
        )
        assert cmd.type == CommandType.NAVIGATE
        assert cmd.action == "go"
        assert cmd.target == "kitchen"
        assert cmd.args == {"key": "value"}
