from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langchain_openai import ChatOpenAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

import os

from pc_vs import VectorStoreManager
from utils import handle_intents, construct_system_prompt

load_dotenv()

# Constants and configuration
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Create FastAPI app
app = FastAPI()

# Setup LLMs for LangChain
OpenAI_llm = ChatOpenAI(
    model="gpt-4o-mini-2024-07-18",
    temperature=0.6,
    max_tokens=5000,
    timeout=None,
    max_retries=2,
    api_key=OPENAI_API_KEY,
)

nemo_nvidia_llm = ChatNVIDIA(
    model="meta/llama-3.1-70b-instruct",
    temperature=0.6,
    max_tokens=5000,
    api_key=NVIDIA_API_KEY,
)

all_llms = [nemo_nvidia_llm, OpenAI_llm]

pinecone_vs = VectorStoreManager()

async def generate_summary(user_input, bot_response, prev_summary):
    system_prompt = ("Distill the chat messages into a single summary paragraph/message. "
                    "Include as many specific details as you can.")
    content = (f"These are the summary of the chat thus far: {prev_summary}\n"
               f"This is the user input now: {user_input}, and here is the bot response: {bot_response}")
    messages = [HumanMessage(content=content), SystemMessage(content=system_prompt)]
    errors = []
    for llm in all_llms:
        try:
            response = await llm.ainvoke(messages)
            return {"messages": response}
        except Exception as e:
            errors.append(f"{type(llm).__name__}: {str(e)}")
    print(f"All LLMs failed.\nErrors: {errors}")
    raise Exception("Error: LLMs Down.")

async def call_model(user_input: str, userid: str, personality: str, summary: str, challenge: str):
    retrieved_docs = pinecone_vs.retrieve_from_vector_store(user_input, 1)
    retrieved_context = "\n".join([res.page_content for res in retrieved_docs])
    
    system_prompt = construct_system_prompt({"configurable": {"thread_id": userid, "personality": personality}}, 
                                            retrieved_context, summary, challenge)

    messages = [HumanMessage(content=user_input), SystemMessage(content=system_prompt)]
    
    errors = []
    for llm in all_llms:
        try:
            response = await llm.ainvoke(messages)
            return {"messages": response}
        except Exception as e:
            errors.append(f"{type(llm).__name__}: {str(e)}")

    print(f"All LLMs failed.\nErrors: {errors}")
    raise Exception("Error: LLMs Down.")


@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    user_input = data.get("message", "")
    userid = data.get("userid", "")
    personality = data.get("personality", "bubbly_coach")
    prev_summary = data.get("summary", "No Summary")
    challenge = data.get("challenge", "No Challenge")

    print("Previous Summary", prev_summary)

    if not user_input:
        return JSONResponse(content={"error": "Message is required"}, status_code=400)
    if not userid:
        return JSONResponse(content={"error": "No userid passed"}, status_code=400)

    try:
        response = await call_model(user_input, userid, personality, prev_summary, challenge)
        summary = await generate_summary(user_input, response, prev_summary)
        return JSONResponse(content = response, new_summary = summary)

    except Exception as e:
        return JSONResponse(content={"error": f"Internal server error: {str(e)}"}, status_code=500)
