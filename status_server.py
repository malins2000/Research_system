import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import str
import uuid

from tools import Blackboard
from agents import StatusReportAgent
from mock_llm import MockLLMClient  # Or your RealLLMClient

app = FastAPI(title="Research System Control API")

# --- Global Components ---
# This server will use its own LLM client and Blackboard instance
# to communicate with the main application.
try:
    # IMPORTANT: Ensure the LLM client here matches the one in main.py
    # (e.g., MockLLMClient or RealLLMClient)
    llm_client = MockLLMClient()
    blackboard = Blackboard("blackboard.json")
except Exception as e:
    print(f"Error initializing server components: {e}")
    llm_client = None
    blackboard = None

# --- Pydantic Models ---
class TaskInput(BaseModel):
    title: str
    description: str

class StatusResponse(BaseModel):
    summary: str

class TaskResponse(BaseModel):
    message: str
    task_id: str

# --- API Endpoints ---
@app.get("/api/status", response_model=StatusResponse)
async def get_system_status():
    """
    (READ)
    Reads the current state of the plan and blackboard and returns
    an LLM-generated summary.
    """
    if not llm_client:
        raise HTTPException(status_code=500, detail="LLM Client not initialized")

    try:
        agent = StatusReportAgent(llm_client)
        summary = agent.execute()
        return StatusResponse(summary=summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")

@app.post("/api/add_task", response_model=TaskResponse)
async def add_new_task(task: TaskInput):
    """
    (WRITE)
    Adds a new task proposal to the 'user_proposals' section of the
    blackboard for the orchestrator to pick up.
    """
    if not blackboard:
        raise HTTPException(status_code=500, detail="Blackboard not initialized")

    try:
        task_id = f"user_task_{uuid.uuid4()}"
        task_data = {
            "title": task.title,
            "summary": task.description,
            "justification": "Submitted by user via control panel."
        }

        # Use the thread-safe post method to add the task
        blackboard.post("user_proposals", task_id, task_data)

        return TaskResponse(message="Task submitted successfully. It will be added to the plan on the next exploration loop.", task_id=task_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error posting task: {str(e)}")

if __name__ == "__main__":
    print("Starting Status & Control API server on http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
