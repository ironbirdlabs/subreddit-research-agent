"""
AgentSystems Agent Template

This is a minimal working agent that demonstrates the LangGraph pattern.
To customize this agent, modify:
  1. State TypedDict - define what data flows through your agent
  2. Request/Response models - define your API contract
  3. Graph nodes - implement your business logic
  4. Prompts - customize the LLM instructions

Required endpoints (do not remove):
  POST /invoke    - Main agent logic
  GET  /health    - Container health check
  GET  /metadata  - Agent information
"""

from datetime import datetime, timezone
from typing import TypedDict, List, Dict, Any
import pathlib
import yaml

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END

from agentsystems_toolkit import get_model


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Setup and Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

load_dotenv()

# Load and merge agent metadata from agent.yaml + metadata.yaml
agent_identity = yaml.safe_load(
    pathlib.Path(__file__).with_name("agent.yaml").read_text()
)
agent_metadata = yaml.safe_load(
    pathlib.Path(__file__).with_name("metadata.yaml").read_text()
)

# Merge metadata (metadata.yaml takes precedence on conflicts)
meta: Dict[str, Any] = {**agent_identity, **agent_metadata}

app = FastAPI(title=meta.get("name", "Agent"), version=meta.get("version", "0.1.0"))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# API Models - Define your request/response contract
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class InvokeRequest(BaseModel):
    """Request payload sent to the agent."""

    date: str = "April 1"  # Example: "December 25"


class InvokeResponse(BaseModel):
    """Response returned by the agent."""

    thread_id: str  # Unique invocation ID from gateway
    date: str  # The date that was analyzed
    events: List[str]  # Historical events found
    story: str  # Narrative weaving events together
    timestamp: datetime  # When the response was generated


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Graph State - Define what data flows through your agent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class State(TypedDict):
    """
    State holds all data that flows between nodes in the graph.

    Each node receives the current state, modifies it, and returns
    the updated state. This pattern makes data flow explicit and testable.
    """

    date: str  # Input: date to analyze
    historical_events: List[str]  # Output from first node
    story: str  # Output from second node


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Model Configuration - Specify which model to use
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# get_model() does two things:
# 1. Model routing: Maps your model ID to the configured provider
# 2. Framework wrapping: Returns the right object type for your framework
#
# As an agent builder, you specify:
# - Model ID: Which model to use (e.g., "gemma3:1b", "claude-sonnet-4")
# - Framework: Which library interface to return (e.g., "langchain")
#
# The platform operator configures the actual provider (OpenAI, Anthropic,
# Bedrock, Ollama) and credentials in the AgentSystems UI.
#
# Model IDs: https://docs.agentsystems.ai/deploy-agents/supported-models
# Frameworks: https://docs.agentsystems.ai/deploy-agents/supported-frameworks
model = get_model(
    "gemma3:1b", "langchain", temperature=0
)  # 0 = deterministic, 1 = creative

# Example model IDs: "claude-sonnet-4", "gpt-5-nano", "llama3.3:70b", "nova-pro"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Prompts and Chains - Define how to interact with the LLM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Prompt for extracting historical events
events_prompt = PromptTemplate(
    template="""
You are a helpful AI that MUST return VALID JSON ONLY. No markdown.
Return exactly 3 historical events that occurred on this month/day: {date}

Expected format:

{{
  "events": [
    "Event #1",
    "Event #2",
    "Event #3"
  ]
}}

Ensure events are family-friendly and appropriate for all ages.
""",
    input_variables=["date"],
)

# Chain: prompt → model → JSON parser
# The | operator creates a pipeline that flows data left to right
json_parser = JsonOutputParser()
events_chain = events_prompt | model | json_parser

# Prompt for creating a narrative story
story_prompt = PromptTemplate(
    template="""
Compose a concise, engaging narrative (max 120 words) weaving these historical events together on {date}:
{events}

Ensure all content is family-friendly and appropriate for all ages.
""",
    input_variables=["date", "events"],
)

# Chain: prompt → model (returns raw text)
story_chain = story_prompt | model


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Graph Nodes - Implement your business logic
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def get_historical_events_node(state: State) -> State:
    """
    First node: Fetch historical events for the given date.

    This node calls the LLM to retrieve historical events and updates
    the state with the results.

    Args:
        state: Current state containing the date

    Returns:
        Updated state with historical_events populated
    """
    result = events_chain.invoke({"date": state["date"]})
    state["historical_events"] = result.get("events", [])
    return state


def create_story_node(state: State) -> State:
    """
    Second node: Create a narrative story from the historical events.

    This node takes the events from the previous node and asks the LLM
    to weave them into a coherent narrative.

    Args:
        state: Current state containing date and historical_events

    Returns:
        Updated state with story populated
    """
    result = story_chain.invoke(
        {"date": state["date"], "events": state["historical_events"]}
    )

    # LangChain models return AIMessage objects; extract the text content
    state["story"] = result.content if hasattr(result, "content") else str(result)
    return state


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Graph Construction - Define the execution flow
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Create a new graph that uses our State type
graph = StateGraph(State)

# Add nodes to the graph
graph.add_node("get_events", get_historical_events_node)
graph.add_node("create_story", create_story_node)

# Define the execution flow: get_events → create_story → END
graph.add_edge("get_events", "create_story")
graph.add_edge("create_story", END)

# Set the starting point
graph.set_entry_point("get_events")

# Compile the graph into an executable pipeline
# This validates the graph structure and prepares it for execution
pipeline = graph.compile()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FastAPI Endpoints - Required by AgentSystems platform
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@app.post("/invoke", response_model=InvokeResponse)
async def invoke(request: Request, req: InvokeRequest) -> InvokeResponse:
    """
    Main agent endpoint - executes the LangGraph pipeline.

    The AgentSystems gateway calls this endpoint and injects the
    X-Thread-Id header for request tracking and observability.

    Args:
        request: FastAPI request object (contains headers)
        req: Parsed request body

    Returns:
        InvokeResponse containing results from the graph execution
    """
    # Extract the unique thread ID injected by the gateway
    thread_id = request.headers.get("X-Thread-Id", "")

    # Initialize the state with the input data
    initial_state: State = {
        "date": req.date,
        "historical_events": [],
        "story": "",
    }

    # Execute the graph pipeline
    # The graph will run: get_events → create_story
    final_state: State = pipeline.invoke(initial_state)

    # Return the results
    return InvokeResponse(
        thread_id=thread_id,
        date=final_state["date"],
        events=final_state["historical_events"],
        story=final_state["story"],
        timestamp=datetime.now(timezone.utc),
    )


@app.get("/health")
async def health() -> Dict[str, str]:
    """
    Health check endpoint.

    The AgentSystems platform uses this to verify the container is ready
    before routing traffic to it.

    Returns:
        Status and version information
    """
    return {"status": "ok", "version": meta.get("version", "0.1.0")}


@app.get("/metadata")
async def metadata() -> Dict[str, Any]:
    """
    Metadata endpoint.

    Returns merged agent information from agent.yaml + metadata.yaml
    for display in the AgentSystems UI and for gateway routing decisions.

    Returns:
        Complete agent metadata dictionary
    """
    return meta
