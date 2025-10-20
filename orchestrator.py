import json
from typing import TypedDict, List, Optional, Any, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

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
from tools import PlanManager, Blackboard, PersonaLoader, RAGSystem, ArxivSearchTool
from mock_llm import MockLLMClient


# Define the state for the graph
class GraphState(TypedDict):
    user_prompt: str
    plan_manager: PlanManager
    blackboard: Blackboard
    persona_loader: PersonaLoader
    rag_system: RAGSystem
    arxiv_tool: ArxivSearchTool
    llm_client: Any
    current_plan_node_id: Optional[str]
    feedback: Optional[dict]  # This is for the *research step*
    run_log: List[str]
    final_summary: Optional[str]
    last_completed_node: Optional[str]

    # --- NEW FIELDS FOR PLANNING LOOP ---
    current_plan_json: Optional[Dict]  # Stores the plan being critiqued
    planning_feedback: Optional[str]   # Stores the critic's feedback for the planner

    # --- NEW FIELD ---
    project_summary_so_far: Optional[str] # Stores the running summary

    # --- NEW FIELD ---
    research_feedback: Optional[str] # Stores critic feedback for the current research topic
    research_retry_count: int = 0    # Tracks retries for the current research topic

    # --- NEW FIELDS ---
    best_draft_so_far: Optional[str] = None # Stores the text of the best draft during retries
    best_rating_so_far: int = 0             # Stores the rating of the best draft


# --- Node Functions ---

def planning_node(state: GraphState) -> dict:
    """
    Creates or refines the initial research plan.
    """
    print("--- Executing Planning Node ---")
    log = state.get("run_log", [])

    # Read the feedback from the last critique
    feedback = state.get("planning_feedback")

    planner = PlannerAgent(state["llm_client"])
    planner.execute(state["user_prompt"], state["plan_manager"], feedback) # Pass feedback

    # Save the new plan to the state for the critic to read
    new_plan_json = state["plan_manager"].plan.model_dump()

    log.append("Initial plan created/refined.")
    return {
        "run_log": log,
        "last_completed_node": "planning_node",
        "current_plan_json": new_plan_json  # <-- Store the new plan
    }


def research_node(state: GraphState) -> dict:
    """
    Conducts parallel research and a multi-round parallel expert debate.
    """
    print("--- Executing Research Node (with Parallel Debate) ---")
    log = state.get("run_log", [])
    plan_manager = state["plan_manager"]
    blackboard = state["blackboard"]
    persona_loader = state["persona_loader"]
    llm_client = state["llm_client"]

    # --- CONFIGURATION ---
    DEBATE_ROUNDS = 3
    # ---------------------

    # --- ADD THIS: Get feedback from the last critique attempt ---
    feedback_for_experts = state.get("research_feedback")
    if feedback_for_experts:
        log.append(f"Retrying research based on critic feedback: {feedback_for_experts}")
    # --- END ADDITION ---


    # Get the next pending node from the plan
    next_node = plan_manager.get_next_pending_node()
    if not next_node:
        log.append("All plan nodes completed.")
        return {"run_log": log, "current_plan_node_id": None}

    node_id = next_node.id
    plan_manager.update_node_status(node_id, "in-progress")
    log.append(f"Starting research for plan node: '{next_node.title}' (ID: {node_id})")

    # --- ADD THIS: Clear operational blackboard sections ---
    print("Clearing operational blackboard sections for new loop...")
    blackboard.clear_section("retrieved_data")
    blackboard.clear_section("expert_discussion")
    blackboard.clear_section("output_draft")
    blackboard.clear_section("topic_proposals") # Also clear proposals
    log.append("Cleared operational blackboard sections.")
    # --- END ADDITION ---

    # --- ADD THIS: Reset best draft tracking for the new node ---
    state["best_draft_so_far"] = None
    state["best_rating_so_far"] = 0
    # We keep research_retry_count as is, it gets reset by the router on success/limit
    # --- END ADDITION ---

    # 1. Analyze task and determine required experts
    analytic_agent = AnalyticAgent(llm_client, persona_loader)
    required_roles = analytic_agent.execute(next_node.description)
    log.append(f"Required experts identified: {required_roles}")

    # 2. Create expert agents
    expert_forge = ExpertForge(llm_client, persona_loader)
    experts = expert_forge.create_experts(required_roles)
    if not experts:
        log.append("No experts were created. Skipping to next node.")
        plan_manager.update_node_status(node_id, "completed") # Mark as complete to avoid loop
        return {"run_log": log, "current_plan_node_id": node_id, "last_completed_node": "research_node"}

    # 3. Gather information (using the new parallel RetrievalAgent)
    rag_system = state["rag_system"]
    arxiv_tool = state["arxiv_tool"]
    retrieval_agent = RetrievalAgent(llm_client, rag_system, arxiv_tool)
    retrieved_docs = retrieval_agent.execute(next_node.title)
    blackboard.post("retrieved_data", "docs", retrieved_docs)
    log.append(f"Retrieved {len(retrieved_docs)} unique documents.")

    # 4. Run the Parallel Expert Debate
    discussion_history = []
    log.append(f"Starting {DEBATE_ROUNDS}-round expert debate with {len(experts)} experts...")

    for i in range(DEBATE_ROUNDS):
        print(f"--- Debate Round {i+1} ---")
        log.append(f"Starting debate round {i+1}/{DEBATE_ROUNDS}")

        round_responses = []
        with ThreadPoolExecutor(max_workers=len(experts)) as executor:
            # --- MODIFY THIS PART ---
            # Get the current summary from the state
            current_summary = state.get("project_summary_so_far")

            # Create a partial function to pass ALL arguments, INCLUDING feedback
            execute_task = partial(
                lambda expert: expert.execute(
                    next_node.description,
                    retrieved_docs,
                    discussion_history,
                    current_summary,
                    feedback_for_experts # <-- Pass the feedback here
                ),
            )

            # Map the execute function to all experts in parallel
            # This sends all requests to vLLM at once
            results = list(executor.map(execute_task, experts))

            round_responses = results

        # Add all responses from this round to the main history
        discussion_history.extend(round_responses)
        log.append(f"Round {i+1} complete. Collected {len(round_responses)} insights.")

        # Post the *entire* updated history to the blackboard
        blackboard.post("expert_discussion", "transcript", discussion_history)

    log.append("Debate finished. Full transcript saved to blackboard.")

    # --- ADD THIS: Clear the feedback after using it ---
    # Ensures feedback from a previous failure isn't reused if the next critique passes
    return {
        "run_log": log,
        "current_plan_node_id": node_id,
        "last_completed_node": "research_node",
        "research_feedback": None # Clear feedback after the debate runs
    }
    # --- END ADDITION ---


def writing_node(state: GraphState) -> dict:
    """
    Synthesizes the full expert debate transcript into a coherent draft.
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

    # Get the full debate transcript from the blackboard
    debate_transcript = blackboard.get("expert_discussion", "transcript")

    if not debate_transcript:
        log.append("Writing Node: No debate transcript found on blackboard to synthesize.")
        return {"run_log": log, "last_completed_node": "writing_node"}

    # Generate the output using the OutputGenerationAgent
    output_agent = OutputGenerationAgent(state["llm_client"])
    draft_text = output_agent.execute(topic_description, debate_transcript)

    # Post the final draft to the blackboard
    blackboard.post("output_draft", current_node_id, draft_text)
    log.append(f"Draft written for node: {current_node_id}")

    return {"run_log": log, "last_completed_node": "writing_node"}


def exploration_node(state: GraphState) -> dict:
    """
    Explores new research topics based on AI and user proposals.
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
        return {"run_log": log, "last_completed_node": "exploration_node"}

    # --- UPDATED LOGIC ---

    # 1. Get AI proposals
    explorer_agent = TopicExplorerAgent(state["llm_client"])
    ai_proposals = explorer_agent.execute(draft, retrieved_docs)
    log.append(f"AI Topic Explorer found {len(ai_proposals)} proposals.")

    # 2. Get User proposals from blackboard
    user_proposals_section = blackboard.get_section("user_proposals")
    user_proposals = list(user_proposals_section.values()) if user_proposals_section else []

    if user_proposals:
        log.append(f"Found {len(user_proposals)} user-submitted proposals on the blackboard.")
        # Clear the section so they aren't processed again
        blackboard.clear_section("user_proposals")

    # 3. Combine all proposals
    all_proposals = ai_proposals + user_proposals

    # 4. Update the plan with all new topics
    if all_proposals:
        updater_agent = PlanUpdaterAgent(state["llm_client"])
        updater_agent.execute(
            all_proposals,
            plan_manager,
            current_node_id,
            state["user_prompt"]  # Pass the main prompt
        )
        log.append(f"Plan updated with {len(all_proposals)} new topics (AI + User).")
    else:
        log.append("No new topics proposed by AI or User.")

    # --- END UPDATED LOGIC ---

    return {"run_log": log, "last_completed_node": "exploration_node"}


def critique_node(state: GraphState) -> dict:
    """
    Evaluates the most recent output (plan or draft) and tracks the best draft.
    """
    print("--- Executing Critique Node ---")
    log = state.get("run_log", [])
    last_node = state["last_completed_node"]
    current_node_id = state.get("current_plan_node_id")

    content_to_review = None
    evaluation_criteria = ""
    previous_feedback = None

    if last_node == "planning_node":
        content_to_review = state.get("current_plan_json")
        evaluation_criteria = "Evaluate the logical structure, completeness (including data gathering steps), and feasibility of this research plan."
        previous_feedback = state.get("planning_feedback")

    elif last_node in ["writing_node", "exploration_node"]:
        content_to_review = state["blackboard"].get("output_draft", current_node_id)
        evaluation_criteria = "Evaluate the clarity, coherence, and accuracy of the generated text based on standard research principles."
        previous_feedback = state.get("research_feedback")

    if not content_to_review:
        log.append("Critique Node: No content found to review.")
        return {"run_log": log, "feedback": {"rating": 0, "feedback": "No content to review."}}

    critic_agent = CriticAgent(state["llm_client"])
    feedback = critic_agent.execute(content_to_review, evaluation_criteria, previous_feedback)
    rating = feedback.get("rating", 0)
    log.append(f"Critique complete. Rating: {rating}")

    # --- ADD THIS: Track the best draft during research retries ---
    if last_node in ["writing_node", "exploration_node"]:
        current_draft = state["blackboard"].get("output_draft", current_node_id)
        best_rating_so_far = state.get("best_rating_so_far", 0)

        if current_draft and rating > best_rating_so_far:
            log.append(f"New best draft found with rating {rating} (previous best: {best_rating_so_far}).")
            state["best_draft_so_far"] = current_draft
            state["best_rating_so_far"] = rating
    # --- END ADDITION ---

    # Save feedback for the correct loop
    if last_node == "planning_node":
        return {"run_log": log, "planning_feedback": feedback.get('feedback'), "feedback": feedback}
    else:
        return {"run_log": log, "feedback": feedback} # Keep returning feedback for router


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

def _update_running_summary(llm_client: Any, previous_summary: Optional[str], new_section_title: str, new_section_text: str) -> str:
    """Uses the LLM to update the running project summary."""
    print("Updating running project summary...")

    if not previous_summary:
        prompt = (
            f"You are summarizing a research project section by section. "
            f"This is the first completed section.\n\n"
            f"**Section Title:** {new_section_title}\n"
            f"**Section Content:**\n{new_section_text}\n\n"
            f"Please provide a concise summary (1-2 paragraphs) of this section to start the overall project summary."
        )
    else:
        prompt = (
            f"You are summarizing a research project section by section. "
            f"Here is the summary of the work completed so far:\n"
            f"**Previous Summary:**\n{previous_summary}\n\n"
            f"A new section has just been completed:\n"
            f"**New Section Title:** {new_section_title}\n"
            f"**New Section Content:**\n{new_section_text}\n\n"
            f"Please provide an updated, concise summary (2-3 paragraphs) that incorporates the key points of the new section into the previous summary. "
            f"Focus on the flow and key findings."
        )

    try:
        new_summary = llm_client.query(prompt)
        print("Running summary updated.")
        return new_summary
    except Exception as e:
        print(f"Error updating running summary: {e}")
        # Fallback: just append titles or return previous summary
        return previous_summary + f"\n\n(Error summarizing: {new_section_title})"

def after_critique_router(state: GraphState) -> str:
    """
    Routes the workflow after critique, implementing a retry limit
    with a "save best draft" fallback.
    """
    last_completed_node = state.get("last_completed_node")
    feedback = state.get("feedback")
    rating = feedback.get("rating", 0)
    log = state.get("run_log", []) # Get log to append warnings

    # --- Planning Loop Logic (Unchanged, uses rating > 90) ---
    if last_completed_node == "planning_node":
        print("--- Deciding After Plan Critique ---")
        if rating >= 90:
            print(f"Plan approved with rating {rating}. Proceeding to research.")
            state["planning_feedback"] = None # Clear feedback on success
            return "research_node"
        else:
            print(f"Plan rejected with rating {rating}. Looping back for refinement.")
            # planning_feedback was already set in critique_node
            return "planning_node"

    # --- Research Loop Logic (NEW "Save Best" Implementation) ---
    elif last_completed_node in ["writing_node", "exploration_node"]:
        print("--- Deciding After Research Critique ---")
        current_node_id = state["current_plan_node_id"]
        plan_manager = state["plan_manager"]

        # --- CONFIGURATION ---
        RESEARCH_APPROVAL_THRESHOLD = 80
        MAX_RESEARCH_RETRIES = 3
        # ---------------------

        if rating > RESEARCH_APPROVAL_THRESHOLD: # Approved
            print(f"Draft approved with rating {rating}.")
            if current_node_id:
                plan_manager.update_node_status(current_node_id, "completed")
                print(f"Node {current_node_id} marked as completed.")

                # Use the *just approved* draft for final content and summary update
                approved_draft = state["blackboard"].get("output_draft", current_node_id)
                current_node = plan_manager._find_node_by_id(plan_manager.plan, current_node_id)
                node_title = current_node.title if current_node else "Unknown Section"

                if approved_draft:
                    state["blackboard"].post("final_content", current_node_id, approved_draft)
                    previous_summary = state.get("project_summary_so_far")
                    llm_client = state["llm_client"]
                    new_summary = _update_running_summary(llm_client, previous_summary, node_title, approved_draft)
                    state["project_summary_so_far"] = new_summary

            # Reset retry counters and best draft tracking on success
            state["research_retry_count"] = 0
            state["research_feedback"] = None
            state["best_draft_so_far"] = None
            state["best_rating_so_far"] = 0

            # Proceed: Check for more nodes or go to summary
            if plan_manager.get_next_pending_node():
                print("More pending nodes found. Continuing research loop.")
                return "research_node"
            else:
                print("All plan nodes are complete. Proceeding to summarization.")
                return "summarize_node"

        else: # Rejected
            current_retry_count = state.get("research_retry_count", 0)
            print(f"Draft rejected with rating {rating}. Retry attempt {current_retry_count + 1}/{MAX_RESEARCH_RETRIES}.")

            state["research_retry_count"] = current_retry_count + 1

            if state["research_retry_count"] > MAX_RESEARCH_RETRIES:
                # --- Retry Limit Reached ---
                log.append(f"WARNING: Research retry limit ({MAX_RESEARCH_RETRIES}) reached for node {current_node_id}. Using best draft found (rating: {state.get('best_rating_so_far', 0)}).")
                print(f"WARNING: Retry limit reached for node {current_node_id}. Using best draft.")

                best_draft = state.get("best_draft_so_far")
                best_rating = state.get("best_rating_so_far", 0)

                if current_node_id:
                    plan_manager.update_node_status(current_node_id, "completed") # Mark as complete anyway
                    print(f"Node {current_node_id} marked as completed (fallback).")

                    if best_draft: # Use the best draft found during retries
                        state["blackboard"].post("final_content", current_node_id, best_draft)
                        current_node = plan_manager._find_node_by_id(plan_manager.plan, current_node_id)
                        node_title = current_node.title if current_node else "Unknown Section"

                        previous_summary = state.get("project_summary_so_far")
                        llm_client = state["llm_client"]
                        new_summary = _update_running_summary(llm_client, previous_summary, node_title, best_draft)
                        state["project_summary_so_far"] = new_summary
                    else:
                        # If somehow no best draft was saved (e.g., all attempts failed parsing), save an error message
                        error_message = f"ERROR: Could not generate acceptable content for this section after {MAX_RESEARCH_RETRIES} retries."
                        state["blackboard"].post("final_content", current_node_id, error_message)
                        log.append(f"ERROR: No best draft saved for node {current_node_id}. Saving error message.")

                # Reset counters and tracking before moving on
                state["research_retry_count"] = 0
                state["research_feedback"] = None
                state["best_draft_so_far"] = None
                state["best_rating_so_far"] = 0

                # Proceed: Check for more nodes or go to summary
                if plan_manager.get_next_pending_node():
                    print("Moving to next node after fallback.")
                    return "research_node"
                else:
                    print("All plan nodes complete after fallback. Proceeding to summarization.")
                    return "summarize_node"

            else:
                # --- Limit Not Reached: Store feedback and loop back ---
                state["research_feedback"] = feedback.get('feedback') # Store feedback for next research_node run
                print("Looping back to research node for refinement.")
                return "research_node"

    else: # Fallback
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

    # Add conditional edges
    workflow.add_conditional_edges(
        "critique_node",
        after_critique_router,
    )

    # Compile the graph
    app = workflow.compile()
    print("Graph compiled successfully.")
    return app
