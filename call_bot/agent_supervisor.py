import os, asyncio
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from callbot import chat_endpoint
from challengebot import challengeBot

app = FastAPI()
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OpenAI_llm = ChatOpenAI(
    model="gpt-4o-mini-2024-07-18",
    temperature=0,
    max_tokens=100,
    timeout=None,
    max_retries=2,
    api_key=OPENAI_API_KEY,
)

class ChallengeIntent(BaseModel):
    """""Obtain Intent of Conversation"""""

    progressAmt: float = Field(default=0, description = "How much is the user progressing in the challenge by? Return the "
                             "absolute value in the same units as the challenge (If target is 2L of water and user enters "
                             " 500ml, enter 0.5), and default to 0 if not applicable. The progress can be negative.")
    isProgChallenge: bool = Field(default=False, description="True if user is trying to progress in their "
                                  "current challenge, based on whether you can identify the amount to progress "
                                  "and the context of the message itself. If the units mentioned in the input is completely"
                                  "unrelated to the actual units of the challenge, then its clearly not a challenge, so return false.")
    challengeTarget: int = Field(default=0, description = "What is the absolute target of the challenge? Give the answer in "
                                  "the same units as the challenge (If the challenge is 'Drink 2 liters of water', return 2. ")
    

validation_bot = OpenAI_llm.with_structured_output(ChallengeIntent, method="function_calling")


@app.post("/chat")
async def analyse_challenge_intent(request: Request):
    """Obtain Challenge Intent"""
    data = await request.json()
    user_input = data.get("message", "")
    userid = data.get("userid", "")

    if not user_input:
        return JSONResponse(content={"error": "Message is required"}, status_code=400)
    if not userid:
        return JSONResponse(content={"error": "No userid passed"}, status_code=400)

    # Other optional data
    challenge = data.get("challenge", "No Challenge")
    coachId = data.get("coachId", "female_coach")
    personality = data.get("personality", [])
    coachName = data.get("coachName", "")
    gender = data.get("gender", "")
    background = data.get("background", "")
    isComplete = bool(data.get("isComplete", False))
    challengeProgress = float(data.get("challengeProgress", 0.00))

    if coachId == "custom_coach":
        if not gender or not coachName or not background or not personality:
            return JSONResponse(content={"error": "Invalid custom coach"}, status_code=400)
    try:
        prompt = ("Your purpose is to help identify whether the user is sending a message to progress in the "
            f"current challenge or not.\n User Input: {user_input}\nCurrent Challenge: {challenge}")
        
        result = await asyncio.to_thread(validation_bot.invoke, prompt)

        if not result.isProgChallenge:
            try:
                cur_message = await chat_endpoint(user_input, userid, coachId, personality, coachName, gender, 
                                                  background, challenge, challengeProgress)
                return JSONResponse(content={"response": cur_message, "progressAmt": 0})
            except RuntimeError as e:
                return JSONResponse(content={"error": str(e)}, status_code=500)
            
        else:
            try:
                response = await challengeBot(challenge, challengeProgress, result.challengeTarget, result.progressAmt)
                return JSONResponse(content={"response": response, "progressAmt": result.progressAmt})
            except RuntimeError as e:
                return JSONResponse(content={"error": str(e)}, status_code=500)



    except Exception as e:
        return {"error": f"Internal server error: {str(e)}"}