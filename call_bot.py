from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from contextlib import asynccontextmanager
from langchain_openai import ChatOpenAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage, SystemMessage, trim_messages
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import START, MessagesState, StateGraph
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from pc_vs import VectorStoreManager
from constants import PERSONALITIES
import os

load_dotenv()

# Constants and configuration
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
DB_URI = os.getenv("DB_URI")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global postgres_pool, memory, langgraph_agent, all_llms
    
    postgres_pool = AsyncConnectionPool(
        conninfo=DB_URI,
        max_size=20,
        kwargs={"autocommit": True, "prepare_threshold": 0}
    )
    async with postgres_pool.connection() as conn:
        memory = AsyncPostgresSaver(conn=conn)
        workflow.add_node("model", call_model)
        workflow.add_edge(START, "model")
        langgraph_agent = workflow.compile(checkpointer=memory)
        yield
    
    # await memory.setup()
    await postgres_pool.close()

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tavily = TavilySearchResults(max_results=3)

# Setup workflow for LangChain
workflow = StateGraph(state_schema=MessagesState)

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

zephyr_llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=5000,
    do_sample=False,
    repetition_penalty=1.03,
    streaming=True,
)
zephyr_hf_llm = ChatHuggingFace(llm=zephyr_llm, disable_streaming=False)
all_llms = [OpenAI_llm, nemo_nvidia_llm, zephyr_hf_llm]

pinecone_vs = VectorStoreManager()

async def call_model(state: MessagesState, config):
    trimmed_state = trim_messages(state['messages'], strategy="last", token_counter=len, 
                                  max_tokens=21, start_on="human", end_on=("human"), include_system=False)  # Gets context of last 21 messages
    user_input = trimmed_state[-1].content

    retrieved_docs = pinecone_vs.retrieve_from_vector_store(user_input, 1)
    retrieved_context = "\n".join([res.page_content for res in retrieved_docs])   

    personalityId = config["configurable"].get("personalityId")
    if personalityId == "custom_coach":
        background = config["configurable"].get("background")
        personalities = config["configurable"].get("personalities")
        gender = config["configurable"].get("gender")
        name = config["configurable"].get("name", "Coach")
    else:
        personality_data = PERSONALITIES.get(personalityId, PERSONALITIES["female_coach"])
        background = personality_data.get("Background", "")
        personalities = personality_data.get("Short Description", "")
        name = personality_data.get("Name", "")
        gender = personality_data.get("Gender", "")

    system_prompt = (
        f"You are {name} ({gender}).\nThis is your background: {background}\n"
        f"Use this as contextual information:\n{retrieved_context}\n"
        f"These are your personalities: {personalities}\n"
        "When communicating with the user, remember to stay in character. "
        "Lead the conversation when neccessary, including when the user first contacts you."
    )

    messages = [SystemMessage(content=system_prompt)] + trimmed_state

    errors = []
    for llm in all_llms:
        try:
            response = await llm.ainvoke(messages)
            # print(f"{type(llm).__name__} Response: {response}")
            return {"messages": response}
        except Exception as e:
            errors.append(f"{type(llm).__name__}: {str(e)}")
            # print(f"Error from {type(llm).__name__}: ", e)

    print("All LLMs failed. Errors: ", errors)
    raise Exception("Error: LLMs Down.")


@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    user_input = data.get("message", "")
    userid = data.get("userid", "")
    personality = data.get("personality", "bubbly_coach")
    
    if not user_input:
        return JSONResponse(content={"error": "Message is required"}, status_code=400)
    if not userid:
        return JSONResponse(content={"error": "No userid passed"}, status_code=400)

    try:
        # Stream the model response back to the client
        async def message_stream():
            config = {"configurable": {"thread_id": userid, "personality": personality}}
            
            messages = {"messages": [HumanMessage(content=user_input)]}
            
            try:
                # stream_mode should be how chunks are outputted (values = state, messages is pure content)
                async for msg, metadata in langgraph_agent.astream(messages, config=config, stream_mode="messages"):
                    yield msg.content
                
            except Exception as e:
                yield f"Error: {str(e)}"

        return StreamingResponse(message_stream(), media_type="text/plain")

    except Exception as e:
        return JSONResponse(content={"error": f"Internal server error: {str(e)}"}, status_code=500)