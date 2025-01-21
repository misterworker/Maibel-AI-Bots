import os, asyncio
from langchain_openai import ChatOpenAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage, SystemMessage, trim_messages
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from langgraph.graph import START, MessagesState, StateGraph
from dotenv import load_dotenv
from pc_vs import VectorStoreManager
from utils import handle_intents, construct_system_prompt

# Load environment variables
load_dotenv()

# Constants and configuration
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
DB_URI = os.getenv("DB_URI")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize LangChain LLMs
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

# Initialize vector store manager
pinecone_vs = VectorStoreManager()

# Model call function
async def call_model(state: MessagesState, config):
    trimmed_state = trim_messages(state['messages'], strategy="last", token_counter=len,
                                  max_tokens=11, start_on="human", end_on=("human"), include_system=False)
    user_input = trimmed_state[-1].content

    retrieved_docs = pinecone_vs.retrieve_from_vector_store(user_input, 1)
    retrieved_context = "\n".join([res.page_content for res in retrieved_docs])

    system_prompt = construct_system_prompt(config, retrieved_context)
    messages = [SystemMessage(content=system_prompt)] + trimmed_state

    async def get_nemo_response():
        try:
            return await nemo_nvidia_llm.ainvoke(messages)
        except:
            asyncio.sleep(50)

    async def get_openai_response():
        await asyncio.sleep(25)
        return await OpenAI_llm.ainvoke(messages)
    
    try:
        # Create tasks for each coroutine
        nemo_task = asyncio.create_task(get_nemo_response())
        openai_task = asyncio.create_task(get_openai_response())
        done, pending = await asyncio.wait([nemo_task, openai_task], return_when=asyncio.FIRST_COMPLETED, timeout=50)

        for task in done: response = task.result(); break
        for task in pending: task.cancel()

    except asyncio.TimeoutError:
        # If neither task completes within 25 seconds, return fallback response
        print("Both tasks took too long. Returning fallback response.")
        response.content = "Response took too long. Sorry about that. Please try again."
    
    if not response.content: response.content = "Response unavailable. Sorry😛. Error te-0"
    return {"messages": response}

async def chat_endpoint(user_input, userid, coachId, personality, coachName, gender, background, challenge, challengeProgress):
    async with AsyncConnectionPool(
        conninfo=DB_URI,
        max_size=20,
        kwargs={"autocommit": True, "prepare_threshold": 0}
    ) as pool:
        checkpointer = AsyncPostgresSaver(pool)

        # Setup workflow for LangChain
        workflow = StateGraph(state_schema=MessagesState)
        workflow.add_node("model", call_model)
        workflow.add_edge(START, "model")
        langgraph_agent = workflow.compile(checkpointer=checkpointer)

        try:
            config = {"configurable": {"thread_id": userid, "coachId": coachId, "personality": personality,
                                        "coachName": coachName, "gender": gender, "background": background,
                                        "challenge": challenge, "challengeProgress": challengeProgress}}

            messages = {"messages": [HumanMessage(content=user_input)]}

            ai_msg = await langgraph_agent.ainvoke(messages, config)

            ai_messages = ai_msg['messages']
            for index in range(len(ai_messages) - 1, -1, -1):
                cur_message = ai_messages[index]
                if type(cur_message).__name__ == "AIMessage":
                    return cur_message.content

        except Exception as e:
            raise RuntimeError("An error occurred while processing the chat endpoint.") from e

    
        