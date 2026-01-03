from langgraph.graph import StateGraph, START, END
from nodes import ForensicNodes, ForensicState

def route_next_exam(state: ForensicState) -> str:
    """
    Decide which exam to run next based on remaining planned_run.

    HARD INVARIANT:
    - If planned_run is empty -> go to confidence
    """
    planned = state.get("planned_run", [])

    if not planned:
        return "confidence"

    # Priority routing based on available plans
    if "visual_environment" in planned:
        return "visual_environment"
    if "ela" in planned:
        return "ela"
    if "osint" in planned:
        return "osint"

    return "confidence"


def exam_router(state: ForensicState):
    """
    Routing node that also functions as a cleanup step.
    
    It checks the 'conclusions' and 'skipped_exams' to see what has 
    just finished, and removes those items from 'planned_run'.
    This ensures the graph progresses and prevents infinite loops.
    """
    planned = state.get("planned_run", [])
    conclusions = state.get("conclusions", {})
    skipped = state.get("skipped_exams", {})
    
    # Rebuild planned_run excluding completed items
    remaining = []
    for exam in planned:
        # If the exam has a result or a skip reason, it's done.
        if exam in conclusions or exam in skipped:
            continue
        remaining.append(exam)
    
    state["planned_run"] = remaining
    return state


def create_forensic_graph():
    """
    Adaptive forensic audit graph.
    
    Flow:
    1. Metadata (Always)
    2. Planner (Always) -> Decides queue
    3. Loop: [Exam -> Router -> Next Exam]
    4. Confidence (Once queue empty)
    5. Report (Final)
    """

    engine = ForensicNodes()
    graph = StateGraph(ForensicState)

    # Add all nodes
    graph.add_node("metadata", engine.metadata_node)
    graph.add_node("planner", engine.planner_node)
    
    graph.add_node("visual_environment", engine.visual_environment_node)
    graph.add_node("ela", engine.ela_node)
    graph.add_node("osint", engine.osint_node)
    
    graph.add_node("exam_router", exam_router)
    graph.add_node("confidence", engine.confidence_node)
    graph.add_node("report", engine.report_node)

    # Define Edges
    graph.add_edge(START, "metadata")
    graph.add_edge("metadata", "planner")

    # Conditional logic from Planner
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

    # Exams always return to the router to update state/queue
    graph.add_edge("visual_environment", "exam_router")
    graph.add_edge("ela", "exam_router")
    graph.add_edge("osint", "exam_router")

    # Router decides where to go next (loop back or finish)
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

    # Finalization flow
    graph.add_edge("confidence", "report")
    graph.add_edge("report", END)

    return graph.compile()


if __name__ == "__main__":
    app = create_forensic_graph()
    print("✅ Forensic graph compiled successfully (recursion-safe).")