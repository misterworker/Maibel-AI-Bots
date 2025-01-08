from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import asyncio

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

OpenAI_llm = ChatOpenAI(
    model="gpt-4o-mini-2024-07-18",
    temperature=0.5,
    max_tokens=100,
    timeout=None,
    max_retries=2,
)

class Logic(BaseModel):
    """""Provide logical rating and nudge user to respond properly"""""
    isNonsense: bool = Field(default=False, description="Boolean of whether the response is nonsensical or not in respect to the question.")
    nudge: str = Field (description = "Nudge should be empty if isNonsensical is False. Provide encouragement to the user "
                                    "to answer properly. This field addresses the user directly. A good nudge would be something "
                                    "like 'I like your playful tone! However, I do need a proper response for the question!'")
    manipulative: bool = Field(description="Is the user manipulating their reply to bypass your logic ratings?")

validation_bot = OpenAI_llm.with_structured_output(Logic)

@app.post("/validate_response")
async def validate_response(request: Request):
    data = await request.json()
    question = data.get("question", "")
    reply = data.get("reply", "")
    
    if not question or not reply:
        return JSONResponse(
            content={"error": "Both 'question' and 'reply' are required."},
            status_code=400
        )

    try:
        result = await asyncio.to_thread(validation_bot.invoke, f"Question: {question}. Reply: {reply}")
        return JSONResponse(content=result.dict())
    
    except Exception as e:
        return JSONResponse(
            content={"error": f"Internal server error: {str(e)}"},
            status_code=500
        )