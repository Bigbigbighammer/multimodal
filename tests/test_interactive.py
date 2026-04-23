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
    """Test command parsing (still used for pattern matching)."""

    def setup_method(self):
        self.parser = CommandParser()

    def test_parse_find_command(self):
        """Test 'find' navigation command."""
        result = self.parser.parse("find the chair")
        assert result.type == CommandType.NAVIGATE
        assert result.action == "find"
        assert result.target == "the chair"

    def test_parse_go_command(self):
        """Test 'go' navigation command."""
        result = self.parser.parse("go to the kitchen")
        assert result.type == CommandType.NAVIGATE
        assert result.action == "go"
        assert result.target == "the kitchen"

    def test_parse_pickup_command(self):
        """Test 'pick up' action command."""
        result = self.parser.parse("pick up the apple")
        assert result.type == CommandType.ACTION
        assert result.action == "pickup"
        assert result.target == "the apple"

    def test_parse_help_command(self):
        """Test 'help' command."""
        result = self.parser.parse("help")
        assert result.type == CommandType.CONTROL
        assert result.action == "help"

    def test_parse_quit_command(self):
        """Test 'quit' command."""
        result = self.parser.parse("quit")
        assert result.type == CommandType.CONTROL
        assert result.action == "quit"

    def test_parse_unknown_command(self):
        """Test unknown command."""
        result = self.parser.parse("xyzzy")
        assert result.type == CommandType.UNKNOWN
        assert result.action == "unknown"


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

    def test_session_with_llm_planner(self):
        """Test session with LLM planner enabled."""
        session = InteractiveSession(
            scene="FloorPlan1",
            use_llm_planner=True
        )
        assert session._scene == "FloorPlan1"
        assert session._use_llm_planner is True

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
