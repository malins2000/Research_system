import json
from typing import TypedDict, List, Optional, Any

# Import agents and tools
from agents import (
    PlannerAgent,
    CriticAgent,
    RetrievalAgent,
    AnalyticAgent,
    ExpertAgent,
    ExpertForge,
    OutputGenerationAgent,
    TopicExplorerAgent,
    PlanUpdaterAgent,
    SummaryAgent
)
from tools import PlanManager, Blackboard, PersonaLoader, RAGSystem
from mock_llm import MockLLMClient


# Define the state for the graph
class GraphState(TypedDict):
    user_prompt: str
    plan_manager: PlanManager
    blackboard: Blackboard
    persona_loader: PersonaLoader
    rag_system: RAGSystem
    llm_client: Any
    current_plan_node_id: Optional[str]
    feedback: Optional[dict]
    run_log: List[str]
    final_summary: Optional[str]
    last_completed_node: Optional[str]


# --- Node Functions ---

def planning_node(state: GraphState) -> dict:
    """
    Creates the initial research plan.
    """
    print("--- Executing Planning Node ---")
    log = state.get("run_log", [])

    planner = PlannerAgent(state["llm_client"])
    planner.execute(state["user_prompt"], state["plan_manager"])

    log.append("Initial plan created.")
    return {"run_log": log, "last_completed_node": "planning_node"}


def research_node(state: GraphState) -> dict:
    """
    Conducts research for the current plan node.
    This involves analyzing requirements, creating experts, and gathering data.
    """
    print("--- Executing Research Node ---")
    log = state.get("run_log", [])
    plan_manager = state["plan_manager"]
    blackboard = state["blackboard"]
    persona_loader = state["persona_loader"]
    llm_client = state["llm_client"]

    # Get the next pending node from the plan
    next_node = plan_manager.get_next_pending_node()
    if not next_node:
        log.append("All plan nodes completed.")
        return {"run_log": log, "current_plan_node_id": None}

    node_id = next_node.id
    plan_manager.update_node_status(node_id, "in-progress")
    log.append(f"Starting research for plan node: '{next_node.title}' (ID: {node_id})")

    # Clear blackboard for the new loop
    blackboard.clear_section("retrieved_data")
    blackboard.clear_section("expert_insights")
    blackboard.clear_section("output_draft")

    # 1. Analyze task and determine required experts
    analytic_agent = AnalyticAgent(llm_client, persona_loader)
    required_roles = analytic_agent.execute(next_node.description)
    log.append(f"Required experts identified: {required_roles}")

    # 2. Create expert agents
    expert_forge = ExpertForge(llm_client, persona_loader)
    experts = expert_forge.create_experts(required_roles)

    # 3. Gather information
    rag_system = state["rag_system"]
    retrieval_agent = RetrievalAgent(llm_client, rag_system)
    retrieved_docs = retrieval_agent.execute(next_node.title)
    blackboard.post("retrieved_data", "docs", retrieved_docs)
    log.append(f"Retrieved {len(retrieved_docs)} documents.")

    # 4. Get insights from experts (sequentially for simplicity)
    for expert in experts:
        insights = expert.execute(next_node.description, retrieved_docs)
        blackboard.post("expert_insights", expert.name, insights)
        log.append(f"Got insights from expert: {expert.name}")

    return {
        "run_log": log,
        "current_plan_node_id": node_id,
        "last_completed_node": "research_node"
    }


def writing_node(state: GraphState) -> dict:
    """
    Synthesizes expert insights into a coherent draft.
    """
    print("--- Executing Writing Node ---")
    log = state.get("run_log", [])
    blackboard = state["blackboard"]
    plan_manager = state["plan_manager"]
    current_node_id = state["current_plan_node_id"]

    if not current_node_id:
        log.append("Writing Node: No current plan node ID found. Skipping.")
        return {"run_log": log}

    current_node = plan_manager._find_node_by_id(plan_manager.plan, current_node_id)
    topic_description = current_node.description

    insights_section = blackboard.get_section("expert_insights")
    insights = list(insights_section.values()) if insights_section else []

    if not insights:
        log.append("Writing Node: No insights found on blackboard to synthesize.")
        return {"run_log": log, "last_completed_node": "writing_node"}

    # Generate the output
    output_agent = OutputGenerationAgent(state["llm_client"])
    draft_text = output_agent.execute(topic_description, insights)

    # Post the draft to the blackboard
    blackboard.post("output_draft", current_node_id, draft_text)
    log.append(f"Draft written for node: {current_node_id}")

    return {"run_log": log, "last_completed_node": "writing_node"}


def exploration_node(state: GraphState) -> dict:
    """
    Explores new research topics based on the latest draft and retrieved data.
    """
    print("--- Executing Exploration Node ---")
    log = state.get("run_log", [])
    blackboard = state["blackboard"]
    plan_manager = state["plan_manager"]
    current_node_id = state["current_plan_node_id"]

    if not current_node_id:
        log.append("Exploration Node: No current plan node ID found. Skipping.")
        return {"run_log": log}

    draft = blackboard.get("output_draft", current_node_id)
    retrieved_docs = blackboard.get("retrieved_data", "docs")

    if not draft or not retrieved_docs:
        log.append("Exploration Node: Missing draft or documents for exploration.")
        return {"run_log": log}

    # Explore new topics
    explorer_agent = TopicExplorerAgent(state["llm_client"])
    proposals = explorer_agent.execute(draft, retrieved_docs)

    # Update the plan with new topics
    if proposals:
        updater_agent = PlanUpdaterAgent(state["llm_client"])
        updater_agent.execute(proposals, plan_manager, current_node_id)
        log.append(f"Plan updated with {len(proposals)} new topics.")

    return {"run_log": log, "last_completed_node": "exploration_node"}


def critique_node(state: GraphState) -> dict:
    """
    Evaluates the most recent output (plan or draft).
    """
    print("--- Executing Critique Node ---")
    log = state.get("run_log", [])
    last_node = state["last_completed_node"]

    content_to_review = None
    evaluation_criteria = ""

    if last_node == "planning_node":
        content_to_review = state["plan_manager"].plan.model_dump()
        evaluation_criteria = "Evaluate the logical structure, completeness, and feasibility of this research plan."
    elif last_node == "writing_node" or last_node == "exploration_node":
        node_id = state["current_plan_node_id"]
        content_to_review = state["blackboard"].get("output_draft", node_id)
        evaluation_criteria = "Evaluate the clarity, coherence, and accuracy of the generated text based on standard research principles."

    if not content_to_review:
        log.append("Critique Node: No content found to review.")
        # Default to approved to avoid getting stuck
        return {"run_log": log, "feedback": {"approved": True, "feedback": "No content to review."}}

    critic_agent = CriticAgent(state["llm_client"])
    feedback = critic_agent.execute(content_to_review, evaluation_criteria)
    log.append(f"Critique complete. Approved: {feedback.get('approved')}")

    # The router function will use the 'last_completed_node' from before this critique.
    return {"run_log": log, "feedback": feedback}


def summarize_node(state: GraphState) -> dict:
    """
    Generates a final summary of the entire research report.
    """
    print("--- Executing Summarize Node ---")
    log = state.get("run_log", [])

    # In a real system, we would load all approved content. Here, we'll just use what's on the blackboard.
    # A better approach would be to save approved drafts to a file or a dedicated blackboard section.
    final_content_section = state["blackboard"].get_section("final_content")
    full_text = "\n\n".join(final_content_section.values()) if final_content_section else ""

    if not full_text:
        log.append("Summarize Node: No final content found to summarize.")
        return {"run_log": log, "final_summary": "No content was generated."}

    summary_agent = SummaryAgent(state["llm_client"])
    summary = summary_agent.execute(full_text)

    log.append("Final summary generated.")
    return {"run_log": log, "final_summary": summary}


# --- Edge Functions ---

from langgraph.graph import StateGraph, END

def after_critique_router(state: GraphState) -> str:
    """
    Routes the workflow after the critique node based on the last completed node and feedback.
    """
    last_completed_node = state.get("last_completed_node")
    feedback = state.get("feedback")

    if last_completed_node == "planning_node":
        print("--- Deciding After Plan Critique ---")
        if feedback and not feedback.get("approved", False):
            print("Plan rejected. Looping back to planning.")
            return "planning_node"
        else:
            print("Plan approved. Proceeding to research.")
            return "research_node"

    elif last_completed_node in ["writing_node", "exploration_node"]:
        print("--- Deciding After Research Critique ---")
        if feedback and not feedback.get("approved", False):
            print("Draft rejected. Retrying research for the current node.")
            return "research_node"

        # If approved, mark the current node as 'completed'
        plan_manager = state["plan_manager"]
        current_node_id = state["current_plan_node_id"]
        if current_node_id:
            plan_manager.update_node_status(current_node_id, "completed")
            print(f"Node {current_node_id} marked as completed.")
            # Persist the final content for the summary
            draft = state["blackboard"].get("output_draft", current_node_id)
            state["blackboard"].post("final_content", current_node_id, draft)

        # Check for more pending nodes
        if plan_manager.get_next_pending_node():
            print("More pending nodes found. Continuing research loop.")
            return "research_node"
        else:
            print("All plan nodes are complete. Proceeding to summarization.")
            return "summarize_node"
    else:
        # Fallback just in case
        print("Unknown state after critique. Ending.")
        return END

# --- Graph Assembly ---

def create_graph():
    """
    Creates and compiles the LangGraph for the research system.
    """
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("planning_node", planning_node)
    workflow.add_node("research_node", research_node)
    workflow.add_node("writing_node", writing_node)
    workflow.add_node("exploration_node", exploration_node)
    workflow.add_node("critique_node", critique_node)
    workflow.add_node("summarize_node", summarize_node)

    # Set entry point
    workflow.set_entry_point("planning_node")

    # Add edges
    workflow.add_edge("planning_node", "critique_node")
    workflow.add_edge("research_node", "writing_node")
    workflow.add_edge("writing_node", "exploration_node")
    workflow.add_edge("exploration_node", "critique_node")
    workflow.add_edge("summarize_node", END)

    # Add conditional edge from the critique node
    workflow.add_conditional_edges(
        "critique_node",
        after_critique_router,
    )

    # Compile the graph
    app = workflow.compile()
    print("Graph compiled successfully.")
    return app
