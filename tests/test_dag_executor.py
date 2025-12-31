"""Tests for DAG executor."""
from __future__ import annotations

import pytest
from typing import Any, Dict

from core.base_handler import BaseHandler
from core.dag_executor import DAGExecutor, DAGNode, ExecutionResult, NodeStatus


class SimpleHandler(BaseHandler):
    """Simple test handler."""

    def __init__(self, name: str, output_key: str, output_value: Any) -> None:
        self._name = name
        self._output_key = output_key
        self._output_value = output_value

    @property
    def name(self) -> str:
        return self._name

    def handle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        context[self._output_key] = self._output_value
        return context


class FailingHandler(BaseHandler):
    """Handler that always fails."""

    def __init__(self, name: str = "FailingHandler") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def handle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        raise RuntimeError("Intentional failure")


class TestDAGNode:
    """Tests for DAGNode."""

    def test_node_creation(self) -> None:
        """Test DAGNode creation."""
        handler = SimpleHandler("test", "key", "value")
        node = DAGNode(handler=handler)

        assert node.name == "test"
        assert node.status == NodeStatus.PENDING
        assert node.dependencies == []
        assert node.error is None

    def test_node_with_dependencies(self) -> None:
        """Test DAGNode with dependencies."""
        handler = SimpleHandler("test", "key", "value")
        node = DAGNode(handler=handler, dependencies=["dep1", "dep2"])

        assert node.dependencies == ["dep1", "dep2"]


class TestDAGExecutor:
    """Tests for DAGExecutor."""

    def test_simple_execution(self) -> None:
        """Test simple DAG execution."""
        handler = SimpleHandler("Handler1", "result", 42)
        node = DAGNode(handler=handler)
        executor = DAGExecutor([node])

        context, result = executor.execute({})

        assert context["result"] == 42
        assert "Handler1" in result.completed_nodes
        assert len(result.failed_nodes) == 0

    def test_sequential_execution(self) -> None:
        """Test sequential DAG execution."""
        handler1 = SimpleHandler("H1", "step1", "done1")
        handler2 = SimpleHandler("H2", "step2", "done2")

        nodes = [
            DAGNode(handler=handler1),
            DAGNode(handler=handler2, dependencies=["H1"]),
        ]
        executor = DAGExecutor(nodes)

        context, result = executor.execute({})

        assert context["step1"] == "done1"
        assert context["step2"] == "done2"
        assert result.completed_nodes == ["H1", "H2"]

    def test_parallel_execution(self) -> None:
        """Test parallel DAG execution."""
        handler1 = SimpleHandler("P1", "p1", "val1")
        handler2 = SimpleHandler("P2", "p2", "val2")
        handler3 = SimpleHandler("P3", "p3", "val3")

        nodes = [
            DAGNode(handler=handler1),
            DAGNode(handler=handler2),  # No dependencies - parallel with P1
            DAGNode(handler=handler3, dependencies=["P1", "P2"]),  # After both
        ]
        executor = DAGExecutor(nodes)

        context, result = executor.execute({})

        assert context["p1"] == "val1"
        assert context["p2"] == "val2"
        assert context["p3"] == "val3"
        assert len(result.completed_nodes) == 3

    def test_cycle_detection(self) -> None:
        """Test cycle detection in DAG."""
        handler1 = SimpleHandler("A", "a", 1)
        handler2 = SimpleHandler("B", "b", 2)

        nodes = [
            DAGNode(handler=handler1, dependencies=["B"]),
            DAGNode(handler=handler2, dependencies=["A"]),
        ]

        with pytest.raises(ValueError, match="cycle"):
            DAGExecutor(nodes)

    def test_missing_dependency(self) -> None:
        """Test missing dependency detection."""
        handler = SimpleHandler("A", "a", 1)
        node = DAGNode(handler=handler, dependencies=["NonExistent"])

        with pytest.raises(ValueError, match="unknown node"):
            DAGExecutor([node])

    def test_execution_with_failure(self) -> None:
        """Test execution with handler failure."""
        handler1 = SimpleHandler("OK", "ok", "done")
        handler2 = FailingHandler("FAIL")
        handler3 = SimpleHandler("After", "after", "after")

        nodes = [
            DAGNode(handler=handler1),
            DAGNode(handler=handler2, dependencies=["OK"]),
            DAGNode(handler=handler3, dependencies=["FAIL"]),
        ]
        executor = DAGExecutor(nodes)

        with pytest.raises(RuntimeError, match="Pipeline failed"):
            executor.execute({})

    def test_execution_continue_on_error(self) -> None:
        """Test execution continues on error when stop_on_error=False."""
        handler1 = SimpleHandler("OK", "ok", "done")
        handler2 = FailingHandler("FAIL")

        nodes = [
            DAGNode(handler=handler1),
            DAGNode(handler=handler2),  # Parallel with OK
        ]
        executor = DAGExecutor(nodes)

        context, result = executor.execute({}, stop_on_error=False)

        # OK handler should complete
        assert "OK" in result.completed_nodes
        assert "FAIL" in result.failed_nodes

    def test_execution_plan(self) -> None:
        """Test getting execution plan."""
        h1 = SimpleHandler("A", "a", 1)
        h2 = SimpleHandler("B", "b", 2)
        h3 = SimpleHandler("C", "c", 3)

        nodes = [
            DAGNode(handler=h1),
            DAGNode(handler=h2),
            DAGNode(handler=h3, dependencies=["A", "B"]),
        ]
        executor = DAGExecutor(nodes)

        plan = executor.get_execution_plan()

        # Layer 0: A, B (parallel)
        # Layer 1: C
        assert len(plan) == 2
        assert set(plan[0]) == {"A", "B"}
        assert plan[1] == ["C"]

    def test_node_timings(self) -> None:
        """Test that node timings are recorded."""
        handler = SimpleHandler("Timed", "result", 42)
        node = DAGNode(handler=handler)
        executor = DAGExecutor([node])

        context, result = executor.execute({})

        assert "Timed" in result.node_timings
        assert result.node_timings["Timed"] >= 0
        assert result.total_time >= 0


class TestExecutionResult:
    """Tests for ExecutionResult."""

    def test_result_creation(self) -> None:
        """Test ExecutionResult creation."""
        result = ExecutionResult(
            completed_nodes=["A", "B"],
            failed_nodes=["C"],
            skipped_nodes=["D"],
            total_time=1.5,
            layer_timings={0: 0.5, 1: 1.0},
            node_timings={"A": 0.3, "B": 0.2, "C": 0.1},
        )

        assert len(result.completed_nodes) == 2
        assert len(result.failed_nodes) == 1
        assert result.total_time == 1.5
