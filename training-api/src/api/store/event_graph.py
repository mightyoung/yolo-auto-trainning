"""Bounded event-graph helpers for Training API task history."""

from __future__ import annotations

from datetime import datetime
from typing import Any

MAX_GRAPH_NODES = 20
MAX_GRAPH_EDGES = 20


def build_graph_node(
    node_id: str,
    node_type: str,
    *,
    label: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "type": node_type,
        "label": label or node_type,
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata or {},
    }


def build_graph_edge(
    source: str,
    target: str,
    relation: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "source": source,
        "target": target,
        "type": relation,
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata or {},
    }


def normalize_event_graph(graph: dict[str, Any] | None) -> dict[str, Any]:
    normalized = dict(graph or {})
    normalized["nodes"] = list(normalized.get("nodes") or [])[-MAX_GRAPH_NODES:]
    normalized["edges"] = list(normalized.get("edges") or [])[-MAX_GRAPH_EDGES:]
    normalized["latest_edge"] = normalized["edges"][-1] if normalized["edges"] else None
    normalized["latest_node"] = normalized["nodes"][-1] if normalized["nodes"] else None
    return normalized


def append_graph_event(
    graph: dict[str, Any] | None,
    *,
    source: str,
    target: str,
    relation: str,
    node_type: str | None = None,
    label: str | None = None,
    metadata: dict[str, Any] | None = None,
    target_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = normalize_event_graph(graph)
    nodes = list(normalized.get("nodes") or [])
    edges = list(normalized.get("edges") or [])
    source_node = build_graph_node(source, node_type or relation, label=label, metadata=metadata)
    target_node = build_graph_node(target, relation, metadata=target_metadata)
    edge = build_graph_edge(source, target, relation, metadata=metadata)
    nodes.extend([source_node, target_node])
    edges.append(edge)
    normalized["nodes"] = nodes[-MAX_GRAPH_NODES:]
    normalized["edges"] = edges[-MAX_GRAPH_EDGES:]
    normalized["latest_node"] = normalized["nodes"][-1]
    normalized["latest_edge"] = normalized["edges"][-1]
    return normalized
