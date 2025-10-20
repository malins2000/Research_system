import argparse
import os
from orchestrator import create_graph, GraphState
from tools import PlanManager, Blackboard, PersonaLoader, RAGSystem, ArxivSearchTool
from mock_llm import MockLLMClient
import logging

DB_PATH = r"/media/malin/1002CB2602CB1020/ChromaDB_RAG"

def main():
    """
    The main entry point for the multi-agent research system.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run the multi-agent research system.")
    parser.add_argument("prompt", type=str, help="The research prompt to execute.")
    args = parser.parse_args()

    # Clean up previous run files if they exist to ensure a fresh start
    if os.path.exists("research_plan.json"):
        os.remove("research_plan.json")
    if os.path.exists("blackboard.json"):
        os.remove("blackboard.json")

    # 1. Initialize components
    llm_client = MockLLMClient()
    plan_manager = PlanManager("research_plan.json")
    blackboard = Blackboard("blackboard.json")
    persona_loader = PersonaLoader("./personas")
    rag_system = RAGSystem(db_path=DB_PATH)
    arxiv_tool = ArxivSearchTool()
    logging.info("All components initialized.")

    # 2. Create the graph
    app = create_graph()

    # 3. Prepare the initial state
    initial_state: GraphState = {
        "user_prompt": args.prompt,
        "plan_manager": plan_manager,
        "blackboard": blackboard,
        "persona_loader": persona_loader,
        "rag_system": rag_system,
        "arxiv_tool": arxiv_tool,
        "llm_client": llm_client,
        "current_plan_node_id": None,
        "feedback": None,
        "run_log": [],
        "final_summary": None,
        "last_completed_node": None
    }

    print("\n--- Starting Research System ---")
    print(f"Prompt: {args.prompt}")

    # 4. Invoke the graph
    final_state = app.invoke(initial_state)

    # 5. Print the final results
    print("\n--- Research System Finished ---")
    print("\n**Run Log:**")
    for i, log_entry in enumerate(final_state['run_log']):
        print(f"{i+1}. {log_entry}")

    print("\n**Final Summary:**")
    print(final_state.get('final_summary', "No summary was generated."))

    # You can also inspect the final plan and blackboard files:
    print("\nFinal plan saved to: research_plan.json")
    print("Final blackboard state saved to: blackboard.json")


if __name__ == "__main__":
    main()
