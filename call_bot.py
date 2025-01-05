from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage, SystemMessage, trim_messages, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from pc_vs import VectorStoreManager
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import os

from constants import PERSONALITIES

load_dotenv()

# Constants and configuration
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (you can restrict this to specific origins)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


# Setup workflow for LangChain
workflow = StateGraph(state_schema=MessagesState)

nemo_nvidia_llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct", api_key=NVIDIA_API_KEY)
pinecone_vs = VectorStoreManager()

async def call_model(state: MessagesState, config):
    print("State: ", state)

    trimmed_state = trim_messages(state['messages'], strategy="last", token_counter=len, 
                                  max_tokens=21, start_on="human", end_on=("human"), include_system=False) # Gets context of last 21 messages
    user_input = trimmed_state[-1].content

    retrieved_docs = pinecone_vs.retrieve_from_vector_store(user_input, 1)
    retrieved_context = "\n".join([res.page_content for res in retrieved_docs])   

    personalityId = config["configurable"].get("personalityId")
    if personalityId == "custom_coach":
        background = config["configurable"].get("background")
        personalities = config["configurable"].get("personalities")
        gender = config["configurable"].get("gender")
    else:
        personality_data = PERSONALITIES.get(personalityId, PERSONALITIES["female_coach"])
        background = personality_data.get("Background", "")
        personalities = personality_data.get("Short Description", "")
        name = personality_data.get("Name", "")
        gender = personality_data.get("Gender", "")

    system_prompt = f"You are {name}({gender})\nThis is your background: {background}\nUse this as contextual information:\n"
    f"{retrieved_context}\nThese are your personalities: {personalities}\nWhen communicating with the user, remember to stay in character."

    messages = [SystemMessage(content=system_prompt)] + trimmed_state

    try:    
        response = await nemo_nvidia_llm.ainvoke(messages)
        return {"messages": response}
    except Exception as e:
        raise Exception(f"stream failed: {e}")


# Add the function to the workflow
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

# Simple memory management
memory = MemorySaver()
langchainApp = workflow.compile(checkpointer=memory)

@app.post("/chat")
async def chat_endpoint(request: Request):
    """
    FastAPI endpoint to handle user input and generate AI responses.
    """
    data = await request.json()
    user_input = data.get("message", "")
    userid = data.get("userid", "")
    personality = data.get("personality", "bubbly_coach")
    
    # Validate input
    if not user_input:
        return JSONResponse(content={"error": "Message is required"}, status_code=400)
    if not userid:
        return JSONResponse(content={"error": "No userid passed"}, status_code=400)

    try:
        # Stream the model response back to the client
        async def message_stream():
            config = {"configurable": {"thread_id": personality, "user_id": userid, "personality": personality}}
            
            messages = {"messages": [HumanMessage(content=user_input)]}
            
            try:
                async for msg, metadata in langchainApp.astream(messages, config=config, stream_mode="messages"):
                    yield msg.content
                
            except Exception as e:
                yield f"Error: {str(e)}"

        return StreamingResponse(message_stream(), media_type="text/plain")

    except Exception as e:
        # Handle any errors and return an appropriate error response
        return JSONResponse(content={"error": f"Internal server error: {str(e)}"}, status_code=500)

