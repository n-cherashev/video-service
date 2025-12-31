"""Core package for video service.

Экспортирует:
- BaseHandler, ExtractorHandler, AnalyzerHandler: базовые классы handlers
- DAGExecutor, DAGNode, ExecutionResult: исполнитель DAG
- run_pipeline: legacy линейный пайплайн
"""

from core.base_handler import BaseHandler, ExtractorHandler, AnalyzerHandler
from core.pipeline import run_pipeline
from core.dag_executor import (
    DAGExecutor,
    DAGNode,
    ExecutionResult,
    NodeStatus,
    build_dag_from_handlers,
)

__all__ = [
    # Handlers
    "BaseHandler",
    "ExtractorHandler",
    "AnalyzerHandler",
    # Pipeline
    "run_pipeline",
    # DAG
    "DAGExecutor",
    "DAGNode",
    "ExecutionResult",
    "NodeStatus",
    "build_dag_from_handlers",
]
