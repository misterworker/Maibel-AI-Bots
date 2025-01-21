from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from typing import Literal
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Load environment variables
load_dotenv()

# Constants
DB_URI = os.getenv("DB_URI")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# FastAPI app setup
app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI LLM setup
OpenAI_llm = ChatOpenAI(
    model="gpt-4o-mini-2024-07-18",
    temperature=0.6,
    max_tokens=5000,
    api_key=OPENAI_API_KEY,
)

# Tool example for React agent
@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")

# Route to test the React agent
@app.post("/chat")
async def test_react_agent_endpoint(request: Request):
    data = await request.json()
    user_message = data.get("message", "What's the weather in nyc")

    async with AsyncConnectionPool(
        conninfo=DB_URI,
        max_size=20,
        kwargs={"autocommit": True, "prepare_threshold": 0}
    ) as pool:
        checkpointer = AsyncPostgresSaver(pool)
        # await checkpointer.setup()  # Only necessary the first time

        tools = [get_weather]
        graph = create_react_agent(OpenAI_llm, tools=tools, checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "test_thread"}}
        res = await graph.ainvoke({"messages": [("human", user_message)]}, config)

        checkpoint = await checkpointer.aget(config)

        print("Checkpoint: ", checkpoint)
        messages = checkpoint.get('channel_values', {}).get('messages', [])

        print("Messages: ", messages)
        # Filter out AIMessage instances and get the last one
        ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
        if ai_messages:
            last_ai_message = ai_messages[-1].content
            print("Last AI Message", last_ai_message)
            return JSONResponse(content={"last_ai_message": last_ai_message})
        else:
            return JSONResponse(content={"error": "No AI message found"}, status_code=404)



