import os, asyncio
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from callbot import chat_endpoint
from challengebot import challengeBot
from constants import PERSONALITIES

app = FastAPI()

# from fastapi.middleware.cors import CORSMiddleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://127.0.0.1:5500"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OpenAI_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=100,
    timeout=None,
    max_retries=2,
    api_key=OPENAI_API_KEY,
)

class ChallengeIntent(BaseModel):
    """""Obtain Challenge Intention of User Input"""""

    progressAmt: float = Field(default=0, description = "How much is the user progressing in the challenge by? Return the "
                             "value in the same units as the challenge (If target is 100 grams and user enters 500mg, "
                             "enter 0.5). Look out for keywords like 'deduct', 'regressed', or any indicators of regression as "
                             "it is possible for the user to deduct their progress from the challenge. If regression is present, "
                             "make the value negative. For example, the user could have said 'I've accidentally logged 3 servings "
                             "of vegetables instead of 2. If he says that, ensure that the value is -1 to adjust for the mistake. If "
                             "the user mentions that he has finished the challenge, just provide the full amount of the challenge as the value.")
    isProgChallenge: bool = Field(default=False, description="True if user is trying to update progress in their "
                                  "current challenge, based on whether you can identify the amount to progress "
                                  "and the context of the message itself. Increase your recall for this field. Look out for "
                                  "phrases like, 'Finished', or 'Im done', and any mention of logging or adjusting their progress in "
                                  "the challenge. These all mean that the user is looking to progress in the challenge, so output true for these.")
    

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
    personalities = data.get("personalities", [])
    coachName = data.get("coachName", "")
    gender = data.get("gender", "")
    background = data.get("background", "")
    isComplete = bool(data.get("isComplete", False))
    challengeProgress = float(data.get("challengeProgress", 0.0))
    recVal = float(data.get("recVal", 0.0))
    recUnit = data.get("recUnit", "")

    if coachId == "custom_coach":
        if not gender:
            return JSONResponse(content={"error": "Gender is required for custom coaches."}, status_code=400)
        if not coachName:
            return JSONResponse(content={"error": "Coach name is required for custom coaches."}, status_code=400)
        if not background:
            return JSONResponse(content={"error": "Background is required for custom coaches."}, status_code=400)
        if not personalities:
            return JSONResponse(content={"error": "Personalities is required for custom coaches."}, status_code=400)

    updatedChallenge = challenge.replace("{x}", str(recVal)).replace("{y}", recUnit)
    print("All data: ", data)
    print("Challenge: ", challenge)
    print("Updated Challenge: ", updatedChallenge)
    try:
        prompt = ("Your purpose is to identify whether the user is sending a message to update his or her progress in "
            f"current challenge or not.\n User Input: {user_input}\nCurrent Challenge: {updatedChallenge}")
        
        result = await asyncio.to_thread(validation_bot.invoke, prompt)

        if not result.isProgChallenge:
            try:
                cur_message = await chat_endpoint(user_input, userid, coachId, personalities, coachName, gender, 
                                                  background, challenge, challengeProgress)
                return JSONResponse(content={"response": cur_message, "finalProg": "NA"})
            except RuntimeError as e:
                return JSONResponse(content={"error": str(e)}, status_code=500)
            
        else:
            if recVal == 0.0:
                return JSONResponse(content={"error": "Invalid Challenge Value"}, status_code=400)
            try:
                if coachId != "custom_coach":
                    personality_data = PERSONALITIES.get(coachId, PERSONALITIES["female_coach"])
                    personalities = personality_data.get("Short Description", "")
                    background = personality_data.get("Background", "")
                    
                response = await challengeBot(challenge, challengeProgress, float(recVal), float(result.progressAmt), user_input,
                                              personalities, background)

                return JSONResponse(content={"response": response[0], "finalProg": response[1]})
            except RuntimeError as e:
                return JSONResponse(content={"error": str(e)}, status_code=500)



    except Exception as e:
        return {"error": f"Internal server error: {str(e)}"}