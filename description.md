# System Overview & Philosophy

This is an autonomous multi-agent system designed for in-depth research and content generation. Its core philosophy is based on a "divide and conquer" strategy, combined with iterative refinement and dynamic adaptation. An Orchestrator agent manages the entire workflow, delegating tasks to specialized agents who communicate and share data via a central Shared Blackboard. The system is designed for long-running, self-hosted tasks, allowing it to explore topics deeply without the constraints of token costs.

# Required Components: Agents & Tools

## Agents
- Orchestrator: The master controller and state manager of the entire system. It does not perform research but invokes all other agents, manages the workflow sequence, enforces meta-rules (e.g., time limits, depth constraints), and handles error recovery. It is the central nervous system of the operation.
- Planner: Responsible for creating the initial high-level research plan (a structured table of contents) based on the user's prompt.
- Analytic: A specialized agent that analyzes text (either the plan or generated output) to determine the required areas of expertise. Its output is a list of expert roles (e.g., ['economist', 'data_scientist']) needed for a task.
- Expert Forge: A factory agent. It receives a list of required roles from the Orchestrator and uses the Persona Library to instantiate the necessary Expert agents, each with its specific system prompt and capabilities.
- Experts (Dynamically Created): The workhorses of the system. Each Expert is a specialist LLM instance with a persona (e.g., "You are a skeptical materials scientist..."). They analyze data, provide insights, and contribute to writing from their unique perspective.
- Critic: The quality assurance agent. It evaluates the completeness, relevance, and coherence of both plans and generated content, providing structured feedback for improvement.
- Retrieval Agent: The information gatherer. This agent designs, expands, and executes queries to find relevant information. It operates in parallel across multiple sources, including the internal RAG system and external APIs (arXiv, Google Search, etc.).
- Output Generation Expert: The lead author. It synthesizes the fragmented insights and data provided by the various Experts on the Shared Blackboard into a single, coherent, and well-structured piece of text for a given plan section.
- Topic Explorer: The discovery agent. It analyzes the generated text and retrieved data to find novel, relevant, and interesting subtopics that are not yet in the plan, proposing them for expansion.
- Plan Updater: The project manager for the research plan. It evaluates proposals from the Topic Explorer and, if they meet criteria (relevance, non-duplication), uses the plan modification tools to insert them into the research plan.
- Summary Agent: The finalizer. After all plan points are completed, this agent reads the entire generated body of work and produces a concise executive summary, introduction, and conclusion.

## Tools & Data Stores
- Shared Blackboard: A centralized, short-term data store (e.g., a structured JSON file or an in-memory database like Redis). It serves as the main communication hub where agents post and read operational data like tasks, retrieved articles, expert insights, and drafts.
- RAG (Retrieval-Augmented Generation) System: The long-term memory. A vector database that stores all retrieved and generated information for semantic search across the entire project.
- Persona Library: A directory of text files, where each file defines a detailed system prompt and capability set for a specific expert role (e.g., chemist.yaml, historian.yaml).
- Plan/Notepad Tools: A set of functions that allow agents (primarily the Plan Updater and Planner) to perform Create, Read, Update, and Delete (CRUD) operations on the research plan file and auxiliary note files.
- Sandboxed Code Executor: A secure tool for running code (e.g., Python for data analysis). It uses Docker containers to execute the code in an isolated environment, capturing the output without risking the host system.
- External Search APIs: A collection of functions for the Retrieval Agent to query external knowledge sources like arXiv, PubMed, Google Scholar, etc.

# Complete System Workflow

This workflow is managed and sequenced by the Orchestrator.

## Phase 1: Planning & Setup
1. User Prompt & Initialization: The user provides the initial research topic. The Orchestrator starts up, initializes the Shared Blackboard, and loads the main project file.
2. Initial Plan Generation: The Orchestrator tasks the Planner to generate an initial table of contents. The Planner posts this plan to the blackboard. The Orchestrator then tasks the Analytic agent to review the plan and suggest initial expert roles for the overall project, which are also posted to the blackboard.
3. Initial Plan Evaluation: The Orchestrator invokes the Critic. The Critic reads the plan from the blackboard and provides feedback. If the plan is incomplete or flawed, the Orchestrator loops back to step 2 with the feedback. If approved, the plan is saved to a file, and the system proceeds.

## Phase 2: The Execution Loop (Repeats for each point in the plan)
1. Loop Initialization: The Orchestrator selects the next "pending" point from the plan file. It updates the point's status to "in-progress" and clears the operational sections of the Shared Blackboard to prepare for the new task. The description of the current plan point is posted to the blackboard.
2. Expert Assembly: The Orchestrator tasks the Analytic agent to determine the specific expert roles needed for this particular plan point. It then instructs the Expert Forge to create instances of these experts using the Persona Library.
3. Parallel Information Gathering: The Orchestrator tasks the Retrieval Agent. The agent reads the task description from the blackboard and formulates multiple queries. It then executes these queries concurrently:
   1. Thread 1: Search internal RAG system.
   2. Thread 2: Search arXiv API.
   3. Thread 3: Search Google Search API.
   4. As results come in, they are processed, chunked, and posted to the "retrieved_data" section of the Shared Blackboard. The results are also saved to the long-term RAG. 
4. Concurrent Context Generation: The Orchestrator activates the assembled Experts. Working in parallel, each Expert reads the retrieved data from the blackboard. Each one analyzes the information from its unique perspective and posts its key insights, relevant data points, and preliminary arguments to the "expert_insights" section of the blackboard. 
5. Synthesized Output Generation: The Orchestrator tasks the Output Generation Expert. It reads the task description and all the expert insights from the blackboard. It synthesizes this information into a single, coherent draft and posts it to the "output_draft" section of the blackboard. Other Experts can perform a quick review and post minor suggestions for refinement. 
6. Exploratory Topic Discovery: While the output is being finalized, the Orchestrator can concurrently task the Topic Explorer. This agent analyzes the new draft and the raw retrieved data to identify potential new subtopics. It posts any findings as structured proposals (title, summary, justification) to the "topic_proposals" section of the blackboard. 
7. Dynamic Plan Update: The Orchestrator tasks the Plan Updater. It reads the proposals from the blackboard and evaluates them against meta-rules (e.g., relevance, scope limits). For any accepted proposal, it uses the plan modification tool to insert the new topic into the main plan file at the appropriate location. 
8. Output & Process Evaluation: The Orchestrator invokes the Critic to evaluate the final draft from the blackboard. 
   1. If major issues are found (e.g., missing information), the Orchestrator can send the process back to Step 6 for more information gathering. 
   2. f minor edits are needed, it loops back to Step 8 for refinement. 
   3. If the output is approved, the Orchestrator marks the content as "finalized" and saves it. 
9. Plan Continuation Check: The Orchestrator checks for more "pending" points in the plan file. 
10. If Yes, it returns to Step 4 to begin the next loop. 
11. If No, it proceeds to the final phase.

## Phase 3: Finalization
1. Final Composition and Summary: The Orchestrator loads all finalized content sections. It tasks the Summary Agent to read the complete text and generate an executive summary, introduction, and conclusion. 
2. Report Generation: The Orchestrator assembles the final document: title page, summary, table of contents, and all the generated sections in order. The final output is then presented to the user.

