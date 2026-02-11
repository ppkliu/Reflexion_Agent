"""Tests for ToolRegistry — centralized tool registration and execution."""

import pytest

from src.tools.registry import ToolRegistry, ToolSpec, _infer_parameters


class TestToolRegistry:
    """Test core ToolRegistry operations."""

    def setup_method(self):
        self.registry = ToolRegistry()

    def test_register_and_execute(self):
        """Register a sync tool and execute it."""
        def add(a: int, b: int) -> int:
            return a + b

        self.registry.add(ToolSpec(
            name="add",
            description="Add two numbers.",
            parameters={"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}},
            callable=add,
            tags=["math"],
        ))

        assert "add" in self.registry
        assert len(self.registry) == 1
        result = self.registry.execute("add", a=3, b=5)
        assert result == 8

    def test_register_decorator(self):
        """Register a tool using the decorator pattern."""
        @self.registry.register(
            name="greet",
            description="Generate a greeting.",
            tags=["text"],
        )
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        assert "greet" in self.registry
        result = self.registry.execute("greet", name="World")
        assert result == "Hello, World!"

    def test_get_schemas(self):
        """get_schemas returns OpenAI-compatible tool schemas."""
        @self.registry.register(name="echo", description="Echo input.")
        def echo(text: str) -> str:
            return text

        schemas = self.registry.get_schemas()
        assert len(schemas) == 1
        schema = schemas[0]
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "echo"
        assert schema["function"]["description"] == "Echo input."
        assert "properties" in schema["function"]["parameters"]

    def test_remove_tool(self):
        """Remove a registered tool."""
        @self.registry.register(name="tmp", description="Temporary.")
        def tmp():
            pass

        assert "tmp" in self.registry
        removed = self.registry.remove("tmp")
        assert removed is True
        assert "tmp" not in self.registry
        assert self.registry.remove("tmp") is False

    def test_execute_missing_tool_raises(self):
        """Executing a non-existent tool raises KeyError."""
        with pytest.raises(KeyError, match="Tool not found"):
            self.registry.execute("nonexistent")

    def test_list_tools_with_tag_filter(self):
        """list_tools filters by tags."""
        @self.registry.register(name="t1", description="t1", tags=["a", "b"])
        def t1():
            pass

        @self.registry.register(name="t2", description="t2", tags=["b", "c"])
        def t2():
            pass

        @self.registry.register(name="t3", description="t3", tags=["c"])
        def t3():
            pass

        assert len(self.registry.list_tools(tags=["a"])) == 1
        assert len(self.registry.list_tools(tags=["b"])) == 2
        assert len(self.registry.list_tools(tags=["c"])) == 2
        assert len(self.registry.list_tools(tags=["d"])) == 0

    def test_get_names(self):
        """get_names returns all registered tool names."""
        @self.registry.register(name="foo", description="foo")
        def foo():
            pass

        @self.registry.register(name="bar", description="bar")
        def bar():
            pass

        names = self.registry.get_names()
        assert set(names) == {"foo", "bar"}


class TestInferParameters:
    """Test automatic parameter inference from function signatures."""

    def test_basic_types(self):
        def func(name: str, count: int, ratio: float, flag: bool):
            pass

        schema = _infer_parameters(func)
        assert schema["type"] == "object"
        props = schema["properties"]
        assert props["name"]["type"] == "string"
        assert props["count"]["type"] == "integer"
        assert props["ratio"]["type"] == "number"
        assert props["flag"]["type"] == "boolean"
        assert set(schema["required"]) == {"name", "count", "ratio", "flag"}

    def test_default_values(self):
        def func(query: str, limit: int = 10):
            pass

        schema = _infer_parameters(func)
        assert "query" in schema["required"]
        assert "limit" not in schema["required"]
        assert schema["properties"]["limit"]["default"] == 10

    def test_skips_self_and_kwargs(self):
        def method(self, data: str, **kwargs):
            pass

        schema = _infer_parameters(method)
        assert "self" not in schema["properties"]
        assert "kwargs" not in schema["properties"]
        assert "data" in schema["properties"]
