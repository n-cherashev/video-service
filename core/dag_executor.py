"""
DAG Executor V2 - исполнитель DAG с контрактами и безопасным мерджем.

Поддерживает:
1. Валидация контрактов (requires/provides)
2. Иммутабельное состояние (StateView вместо dict.copy())
3. Детерминированный merge с политикой конфликтов
4. Обратная совместимость с legacy handlers
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from core.base_handler import BaseHandler


class NodeStatus(Enum):
    """Статус узла DAG."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class DAGNode:
    """Узел DAG с handler'ом и зависимостями."""
    handler: BaseHandler
    dependencies: List[str] = field(default_factory=list)
    status: NodeStatus = NodeStatus.PENDING
    error: Optional[str] = None
    execution_time: float = 0.0

    @property
    def name(self) -> str:
        return self.handler.name

    def __hash__(self) -> int:
        return hash(self.name)


@dataclass
class ExecutionResult:
    """Результат выполнения DAG."""
    completed_nodes: List[str]
    failed_nodes: List[str]
    skipped_nodes: List[str]
    total_time: float
    layer_timings: Dict[int, float]
    node_timings: Dict[str, float]
    contract_errors: List[str] = field(default_factory=list)
    merge_errors: List[str] = field(default_factory=list)


class DAGExecutor:
    """Исполнитель DAG с поддержкой контрактов.

    Два режима работы:
    1. Legacy mode (use_contracts=False): как раньше, dict.copy()
    2. Contract mode (use_contracts=True): StateView + NodePatch + merge policy
    """

    def __init__(
        self,
        nodes: List[DAGNode],
        max_workers: int = 5,
        use_contracts: bool = False,  # Пока по умолчанию legacy
        strict_contracts: bool = False,  # Fail on contract violations
        on_node_start: Optional[Callable[[str], None]] = None,
        on_node_complete: Optional[Callable[[str, float], None]] = None,
        on_node_error: Optional[Callable[[str, Exception], None]] = None,
    ) -> None:
        self.nodes: Dict[str, DAGNode] = {n.name: n for n in nodes}
        self.max_workers = max_workers
        self.use_contracts = use_contracts
        self.strict_contracts = strict_contracts
        self.on_node_start = on_node_start
        self.on_node_complete = on_node_complete
        self.on_node_error = on_node_error

        self._validate_dag()

    def _validate_dag(self) -> None:
        """Валидирует DAG: зависимости и циклы."""
        for name, node in self.nodes.items():
            for dep in node.dependencies:
                if dep not in self.nodes:
                    raise ValueError(f"Node '{name}' depends on unknown node '{dep}'")

        if self._has_cycle():
            raise ValueError("DAG contains a cycle")

    def _has_cycle(self) -> bool:
        """Проверяет наличие цикла в DAG."""
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def dfs(node_name: str) -> bool:
            visited.add(node_name)
            rec_stack.add(node_name)

            node = self.nodes[node_name]
            for dep in node.dependencies:
                if dep not in visited:
                    if dfs(dep):
                        return True
                elif dep in rec_stack:
                    return True

            rec_stack.remove(node_name)
            return False

        for name in self.nodes:
            if name not in visited:
                if dfs(name):
                    return True
        return False

    def _topological_sort(self) -> List[List[str]]:
        """Топологическая сортировка в слои для параллельного выполнения."""
        in_degree: Dict[str, int] = {name: 0 for name in self.nodes}
        dependents: Dict[str, List[str]] = defaultdict(list)

        for name, node in self.nodes.items():
            for dep in node.dependencies:
                dependents[dep].append(name)
                in_degree[name] += 1

        layers: List[List[str]] = []
        ready: deque[str] = deque(
            name for name, degree in in_degree.items() if degree == 0
        )

        while ready:
            current_layer: List[str] = []
            next_ready: List[str] = []

            while ready:
                node_name = ready.popleft()
                current_layer.append(node_name)

            # Сортируем для детерминированного порядка
            current_layer.sort()

            for node_name in current_layer:
                for dependent in dependents[node_name]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        next_ready.append(dependent)

            layers.append(current_layer)
            ready.extend(sorted(next_ready))  # Детерминированный порядок

        total_nodes = sum(len(layer) for layer in layers)
        if total_nodes != len(self.nodes):
            raise ValueError("Topological sort failed - possible cycle detected")

        return layers

    def _execute_node_legacy(
        self,
        node: DAGNode,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Выполняет узел в legacy режиме."""
        if self.on_node_start:
            self.on_node_start(node.name)

        start_time = time.monotonic()
        try:
            node.status = NodeStatus.RUNNING
            result = node.handler.handle(context)
            node.status = NodeStatus.COMPLETED
            node.execution_time = time.monotonic() - start_time

            if self.on_node_complete:
                self.on_node_complete(node.name, node.execution_time)

            return result
        except Exception as e:
            node.status = NodeStatus.FAILED
            node.error = str(e)
            node.execution_time = time.monotonic() - start_time

            if self.on_node_error:
                self.on_node_error(node.name, e)

            raise

    def _execute_node_with_contracts(
        self,
        node: DAGNode,
        state: "PipelineStateV2",
    ) -> "NodePatch":
        """Выполняет узел с контрактами."""
        from models.contracts import StateView, NodePatch

        if self.on_node_start:
            self.on_node_start(node.name)

        start_time = time.monotonic()
        try:
            node.status = NodeStatus.RUNNING

            # Создаём read-only view
            state_view = StateView(state)

            # Выполняем handler
            patch = node.handler.run(state_view)

            if patch.error:
                node.status = NodeStatus.FAILED
                node.error = patch.error
            else:
                node.status = NodeStatus.COMPLETED

            node.execution_time = time.monotonic() - start_time
            patch.execution_time_seconds = node.execution_time

            if self.on_node_complete and not patch.error:
                self.on_node_complete(node.name, node.execution_time)

            return patch

        except Exception as e:
            node.status = NodeStatus.FAILED
            node.error = str(e)
            node.execution_time = time.monotonic() - start_time

            if self.on_node_error:
                self.on_node_error(node.name, e)

            return NodePatch(
                handler_name=node.name,
                error=str(e),
                execution_time_seconds=node.execution_time,
            )

    def execute(
        self,
        context: Dict[str, Any],
        stop_on_error: bool = True,
    ) -> tuple[Dict[str, Any], ExecutionResult]:
        """Выполняет DAG (legacy mode для обратной совместимости)."""
        total_start = time.monotonic()
        layers = self._topological_sort()

        layer_timings: Dict[int, float] = {}
        node_timings: Dict[str, float] = {}
        completed_nodes: List[str] = []
        failed_nodes: List[str] = []
        skipped_nodes: List[str] = []

        # Reset node statuses
        for node_name in self.nodes:
            self.nodes[node_name].status = NodeStatus.PENDING
            self.nodes[node_name].error = None

        for layer_idx, layer in enumerate(layers):
            layer_start = time.monotonic()
            print(f"\n[Layer {layer_idx + 1}/{len(layers)}] Executing: {', '.join(layer)}")

            # Skip nodes with failed dependencies
            nodes_to_skip = []
            if failed_nodes:
                for node_name in layer:
                    node = self.nodes[node_name]
                    for dep in node.dependencies:
                        if dep in failed_nodes:
                            nodes_to_skip.append(node_name)
                            break

            if stop_on_error:
                for node_name in nodes_to_skip:
                    self.nodes[node_name].status = NodeStatus.SKIPPED
                    skipped_nodes.append(node_name)

            nodes_to_run = [
                name for name in layer
                if self.nodes[name].status == NodeStatus.PENDING
            ]

            if len(nodes_to_run) == 1:
                # Sequential execution
                node = self.nodes[nodes_to_run[0]]
                try:
                    context = self._execute_node_legacy(node, context)
                    completed_nodes.append(node.name)
                    node_timings[node.name] = node.execution_time
                except Exception as e:
                    failed_nodes.append(node.name)
                    node_timings[node.name] = node.execution_time
                    if stop_on_error:
                        raise RuntimeError(f"Pipeline failed at {node.name}: {e}") from e

            elif len(nodes_to_run) > 1:
                # Parallel execution
                with ThreadPoolExecutor(max_workers=min(self.max_workers, len(nodes_to_run))) as executor:
                    futures: Dict[Future, str] = {}

                    for node_name in nodes_to_run:
                        node = self.nodes[node_name]
                        # Shallow copy для параллельных узлов
                        future = executor.submit(self._execute_node_legacy, node, context.copy())
                        futures[future] = node_name

                    results: Dict[str, Dict[str, Any]] = {}
                    errors: List[tuple[str, Exception]] = []

                    for future in as_completed(futures):
                        node_name = futures[future]
                        try:
                            result = future.result()
                            results[node_name] = result
                            completed_nodes.append(node_name)
                            node_timings[node_name] = self.nodes[node_name].execution_time
                        except Exception as e:
                            errors.append((node_name, e))
                            failed_nodes.append(node_name)
                            node_timings[node_name] = self.nodes[node_name].execution_time

                    # Детерминированный merge (по имени узла)
                    for node_name in sorted(results.keys()):
                        result = results[node_name]
                        for key, value in result.items():
                            # Перезаписываем только новые/изменённые ключи
                            if key not in context or context[key] != value:
                                context[key] = value

                    if errors and stop_on_error:
                        error_msgs = [f"{name}: {e}" for name, e in errors]
                        raise RuntimeError(f"Pipeline failed: {'; '.join(error_msgs)}")

            layer_timings[layer_idx] = time.monotonic() - layer_start

        total_time = time.monotonic() - total_start

        # Add execution metadata to context
        context["completed_stages"] = set(completed_nodes)
        context["layer_timings"] = layer_timings
        context["node_timings"] = node_timings

        result = ExecutionResult(
            completed_nodes=completed_nodes,
            failed_nodes=failed_nodes,
            skipped_nodes=skipped_nodes,
            total_time=total_time,
            layer_timings=layer_timings,
            node_timings=node_timings,
        )

        return context, result

    def execute_with_state(
        self,
        state: "PipelineStateV2",
        stop_on_error: bool = True,
    ) -> tuple["PipelineStateV2", ExecutionResult]:
        """Выполняет DAG с PipelineStateV2 и контрактами."""
        from models.pipeline_state import PipelineStateV2
        from models.contracts import NodePatch

        total_start = time.monotonic()
        layers = self._topological_sort()

        layer_timings: Dict[int, float] = {}
        node_timings: Dict[str, float] = {}
        completed_nodes: List[str] = []
        failed_nodes: List[str] = []
        skipped_nodes: List[str] = []
        contract_errors: List[str] = []
        merge_errors: List[str] = []

        state.mark_started()

        # Reset node statuses
        for node_name in self.nodes:
            self.nodes[node_name].status = NodeStatus.PENDING
            self.nodes[node_name].error = None

        for layer_idx, layer in enumerate(layers):
            layer_start = time.monotonic()
            print(f"\n[Layer {layer_idx + 1}/{len(layers)}] Executing: {', '.join(layer)}")

            # Skip nodes with failed dependencies
            nodes_to_skip = []
            if failed_nodes:
                for node_name in layer:
                    node = self.nodes[node_name]
                    for dep in node.dependencies:
                        if dep in failed_nodes:
                            nodes_to_skip.append(node_name)
                            break

            if stop_on_error:
                for node_name in nodes_to_skip:
                    self.nodes[node_name].status = NodeStatus.SKIPPED
                    skipped_nodes.append(node_name)

            nodes_to_run = [
                name for name in layer
                if self.nodes[name].status == NodeStatus.PENDING
            ]

            if len(nodes_to_run) == 1:
                # Sequential
                node = self.nodes[nodes_to_run[0]]
                state.mark_stage_started(node.name)

                patch = self._execute_node_with_contracts(node, state)

                if patch.error:
                    failed_nodes.append(node.name)
                    node_timings[node.name] = patch.execution_time_seconds
                    if stop_on_error:
                        state.mark_failed(patch.error)
                        raise RuntimeError(f"Pipeline failed at {node.name}: {patch.error}")
                else:
                    # Apply patch
                    errors = state.apply_patch(patch)
                    if errors:
                        merge_errors.extend(errors)
                        if self.strict_contracts:
                            raise RuntimeError(f"Merge errors: {errors}")

                    completed_nodes.append(node.name)
                    node_timings[node.name] = patch.execution_time_seconds

            elif len(nodes_to_run) > 1:
                # Parallel - собираем patches и мерджим детерминированно
                patches: Dict[str, NodePatch] = {}
                errors_list: List[tuple[str, str]] = []

                with ThreadPoolExecutor(max_workers=min(self.max_workers, len(nodes_to_run))) as executor:
                    futures: Dict[Future, str] = {}

                    for node_name in nodes_to_run:
                        node = self.nodes[node_name]
                        state.mark_stage_started(node_name)
                        future = executor.submit(self._execute_node_with_contracts, node, state)
                        futures[future] = node_name

                    for future in as_completed(futures):
                        node_name = futures[future]
                        patch = future.result()
                        patches[node_name] = patch

                        if patch.error:
                            errors_list.append((node_name, patch.error))
                            failed_nodes.append(node_name)
                        else:
                            completed_nodes.append(node_name)

                        node_timings[node_name] = patch.execution_time_seconds

                # Детерминированный merge patches (по имени узла)
                for node_name in sorted(patches.keys()):
                    patch = patches[node_name]
                    if not patch.error:
                        merge_errs = state.apply_patch(patch)
                        if merge_errs:
                            merge_errors.extend(merge_errs)

                if errors_list and stop_on_error:
                    error_msgs = [f"{name}: {err}" for name, err in errors_list]
                    state.mark_failed("; ".join(error_msgs))
                    raise RuntimeError(f"Pipeline failed: {'; '.join(error_msgs)}")

            layer_timings[layer_idx] = time.monotonic() - layer_start
            state.metrics.layer_timings[layer_idx] = layer_timings[layer_idx]

        total_time = time.monotonic() - total_start

        # Update state metrics
        state.metrics.node_timings = node_timings
        state.mark_completed()

        result = ExecutionResult(
            completed_nodes=completed_nodes,
            failed_nodes=failed_nodes,
            skipped_nodes=skipped_nodes,
            total_time=total_time,
            layer_timings=layer_timings,
            node_timings=node_timings,
            contract_errors=contract_errors,
            merge_errors=merge_errors,
        )

        return state, result

    def get_execution_plan(self) -> List[List[str]]:
        """Возвращает план выполнения (слои)."""
        return self._topological_sort()

    def print_execution_plan(self) -> None:
        """Печатает план выполнения."""
        layers = self._topological_sort()
        print("\nExecution Plan:")
        print("-" * 50)
        for i, layer in enumerate(layers):
            parallel_indicator = " (parallel)" if len(layer) > 1 else ""
            print(f"Layer {i + 1}{parallel_indicator}:")
            for node_name in layer:
                node = self.nodes[node_name]
                deps = f" <- [{', '.join(node.dependencies)}]" if node.dependencies else ""
                print(f"  - {node_name}{deps}")
        print("-" * 50)


def build_dag_from_handlers(
    handlers: List[BaseHandler],
    dependencies: Optional[Dict[str, List[str]]] = None,
) -> DAGExecutor:
    """Создаёт DAGExecutor из списка handlers и зависимостей."""
    dependencies = dependencies or {}

    nodes = []
    for handler in handlers:
        name = handler.name
        deps = dependencies.get(name, [])
        nodes.append(DAGNode(handler=handler, dependencies=deps))

    return DAGExecutor(nodes)


# Type hints for imports
if True:  # TYPE_CHECKING workaround
    from models.pipeline_state import PipelineStateV2
    from models.contracts import NodePatch
