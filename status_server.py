import uvicorn
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uuid
import os
import threading
import logging

# --- Import all system components ---
from orchestrator import create_graph, GraphState
from tools import PlanManager, Blackboard, PersonaLoader, RAGSystem, ArxivSearchTool
from agents import StatusReportAgent
from mock_llm import MockLLMClient  # Or your RealLLMClient

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Research System Control API")

# --- Global Components & State ---
# Initialize all components on startup, just like main.py did
try:
    logger.info("Initializing system components...")

    # Use the same LLM client for the server and the graph
    llm_client = MockLLMClient()

    plan_manager = PlanManager("research_plan.json")
    blackboard = Blackboard("blackboard.json")
    persona_loader = PersonaLoader("./personas")
    rag_system = RAGSystem(db_path="./chroma_db_server")
    arxiv_tool = ArxivSearchTool()

    # Create the compiled LangGraph app
    app.state.graph = create_graph()

    # Global state to track the running job
    app.state.system_status = {"is_running": False, "thread": None, "current_prompt": None}
    logger.info("All components initialized successfully.")
except Exception as e:
    logger.error(f"FATAL: Error initializing components: {e}")
    app.state.graph = None
    app.state.system_status = {"is_running": False, "thread": None, "current_prompt": "INIT_FAILED"}

# --- Pydantic Models ---
class StartResearchInput(BaseModel):
    prompt: str

class TaskInput(BaseModel):
    title: str
    description: str

class StatusResponse(BaseModel):
    is_running: bool
    current_prompt: Optional[str]
    summary: str

class TaskResponse(BaseModel):
    message: str
    task_id: Optional[str] = None

# --- Background Research Task ---
def run_research_in_background(initial_state: GraphState):
    """
    The target function for the background thread.
    This runs the actual LangGraph.
    """
    logger.info(f"Background thread started for prompt: {initial_state['user_prompt']}")
    app.state.system_status["is_running"] = True
    app.state.system_status["current_prompt"] = initial_state['user_prompt']

    try:
        # This is the blocking call that will run for hours
        app.state.graph.invoke(initial_state)
        logger.info("LangGraph invocation finished successfully.")
    except Exception as e:
        logger.error(f"Exception in background research thread: {e}", exc_info=True)
    finally:
        # When done, reset the status
        logger.info("Resetting system status. Research is complete.")
        app.state.system_status["is_running"] = False
        app.state.system_status["thread"] = None
        app.state.system_status["current_prompt"] = None

# --- API Endpoints ---
@app.post("/api/start_research", response_model=TaskResponse)
async def start_new_research(payload: StartResearchInput):
    """
    (WRITE)
    Starts a new research job in a background thread.
    """
    if app.state.system_status["is_running"]:
        raise HTTPException(status_code=409, detail="A research process is already running.")

    if not app.state.graph:
        raise HTTPException(status_code=500, detail="Graph is not compiled. Check server logs.")

    logger.info(f"Received new research request: {payload.prompt}")

    # Clean up files from previous runs
    try:
        if os.path.exists("research_plan.json"):
            os.remove("research_plan.json")
        if os.path.exists("blackboard.json"):
            os.remove("blackboard.json")
        logger.info("Cleaned up old state files.")
    except Exception as e:
        logger.warning(f"Could not clean up state files: {e}")

    # Prepare the initial state (copied from main.py)
    initial_state: GraphState = {
        "user_prompt": payload.prompt,
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

    # Create and start the background thread
    thread = threading.Thread(target=run_research_in_background, args=(initial_state,))
    app.state.system_status["thread"] = thread
    thread.start()

    return TaskResponse(message="Research process started successfully in the background.")

@app.get("/api/status", response_model=StatusResponse)
async def get_system_status():
    """
    (READ)
    Reads the current state and returns an LLM-generated summary.
    """
    if not llm_client:
        raise HTTPException(status_code=500, detail="LLM Client not initialized")

    try:
        agent = StatusReportAgent(llm_client)
        summary = agent.execute()

        return StatusResponse(
            is_running=app.state.system_status["is_running"],
            current_prompt=app.state.system_status["current_prompt"],
            summary=summary
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")

@app.post("/api/add_task", response_model=TaskResponse)
async def add_new_task(task: TaskInput):
    """
    (WRITE)
    Adds a new task proposal to the blackboard for the *running* job.
    """
    if not app.state.system_status["is_running"]:
        raise HTTPException(status_code=400, detail="No research is running. Cannot add tasks.")

    try:
        task_id = f"user_task_{uuid.uuid4()}"
        task_data = {
            "title": task.title,
            "summary": task.description,
            "justification": "Submitted by user via control panel."
        }
        blackboard.post("user_proposals", task_id, task_data)

        return TaskResponse(message="Task submitted successfully. It will be added on the next exploration loop.", task_id=task_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error posting task: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting Master Control API server on http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)