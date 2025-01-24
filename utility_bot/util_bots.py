from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import asyncio, os

load_dotenv()

app = FastAPI()
# from fastapi.middleware.cors import CORSMiddleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://127.0.0.1:5500"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OpenAI_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    max_tokens=100,
    timeout=None,
    max_retries=2,
    api_key=OPENAI_API_KEY,
)

class Logic(BaseModel):
    """""Provide logical rating and nudge user to respond properly"""""
    isNonsense: bool = Field(default=False, description="Boolean of whether the response is nonsensical or not in respect to the question.")
    nudge: str = Field (description = "Nudge should be empty if isNonsensical is False. Provide encouragement to the user "
                                    "to answer properly. This field addresses the user directly. A good nudge would be something "
                                    "like 'I like your playful tone! However, I do need a proper response for the question!'")
    manipulative: bool = Field(description="Is the user manipulating their reply to bypass your logic ratings?")

validation_bot = OpenAI_llm.with_structured_output(Logic, method="function_calling")

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
    
OpenAI_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=30,
    timeout=None,
    api_key=OPENAI_API_KEY,
)

class Recommendation(BaseModel):
    """""Provide Recommendation of x based on challenge"""""
    recommendation: int = Field(description="What integer would you recommend for this challenge, with user data taken into account. If applicable, "
                                "maintain or increase the amount of the target that the user already does.")
    unit: str = Field(description="Readable unit of recommendation provided (Eg. Kilograms, Liters). Keep under 10 characters.", max_length=10)

recommendation_bot = OpenAI_llm.with_structured_output(Recommendation, method="function_calling")
    
@app.post("/rec_x_challenge")
async def challengeRecommendation(request: Request):
    data = await request.json()
    challenge = data.get("challenge", "")
    userData = data.get("userData", "{}")

    if not challenge or not challenge:
        return JSONResponse(
            content={"error": "Both challenge parts are required"},
            status_code=400
        )

    try:
        prompt = ("You are a bot meant to create a recommendation value and readable unit for the challenge. The recommendation "
        "should be based off user data to recommend the 'x' as value and 'y' as units in the challenge. Here is the challenge statement: "
        f"{challenge} Below is the user data.\n{userData}")
        result = await asyncio.to_thread(recommendation_bot.invoke, prompt)
        print(result)
        return JSONResponse(content=result.dict())
    
    except Exception as e:
        return JSONResponse(
            content={"error": f"Internal server error: {str(e)}"},
            status_code=500
        )