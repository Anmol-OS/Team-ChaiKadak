from langgraph.graph import StateGraph, START, END
from nodes import ForensicNodes, ForensicState

def route_next_exam(state: ForensicState) -> str:
    """
    Decide which exam to run next based on remaining planned_run.

    HARD INVARIANT:
    - If planned_run is empty → go to confidence
    """
    planned = state.get("planned_run", [])

    if not planned:
        return "confidence"

    if "visual_environment" in planned:
        return "visual_environment"
    if "ela" in planned:
        return "ela"
    if "osint" in planned:
        return "osint"

    return "confidence"


def exam_router(state: ForensicState):
    """
    Pure routing node.

    IMPORTANT:
    - Must NOT mutate state
    - Exists only to centralize routing
    """
    return state


def create_forensic_graph():
    """
    Adaptive forensic audit graph.
    HARD INVARIANTS:
    - metadata runs exactly once
    - planner runs exactly once
    - each exam runs at most once
    - routing space strictly shrinks
    - graph always reaches END
    """

    engine = ForensicNodes()
    graph = StateGraph(ForensicState)


    graph.add_node("metadata", engine.metadata_node)
    graph.add_node("planner", engine.planner_node)
    graph.add_node("visual_environment", engine.visual_environment_node)
    graph.add_node("ela", engine.ela_node)
    graph.add_node("osint", engine.osint_node)
    graph.add_node("exam_router", exam_router)
    graph.add_node("confidence", engine.confidence_node)
    graph.add_node("report", engine.report_node)
    graph.add_edge(START, "metadata")
    graph.add_edge("metadata", "planner")

    graph.add_conditional_edges(
        "planner",
        route_next_exam,
        {
            "visual_environment": "visual_environment",
            "ela": "ela",
            "osint": "osint",
            "confidence": "confidence",
        },
    )

    graph.add_edge("visual_environment", "exam_router")
    graph.add_edge("ela", "exam_router")
    graph.add_edge("osint", "exam_router")

    graph.add_conditional_edges(
        "exam_router",
        route_next_exam,
        {
            "visual_environment": "visual_environment",
            "ela": "ela",
            "osint": "osint",
            "confidence": "confidence",
        },
    )

    graph.add_edge("confidence", "report")
    graph.add_edge("report", END)

    return graph.compile()


if __name__ == "__main__":
    app = create_forensic_graph()
    print("✅ Forensic graph compiled successfully (recursion-safe).")
    