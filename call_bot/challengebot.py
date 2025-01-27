import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OpenAI_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.85,
    max_tokens=100,
    timeout=10,
    max_retries=2,
    api_key=OPENAI_API_KEY,
)

def calculate_absolute(prog, target, curIncrement):
    cur_absolute = prog*target
    return round(cur_absolute + curIncrement, 2)

def calculate_final_progress(prog, target, curIncrement):
    increment_percent = curIncrement/target
    finalChallengeProgress = round(increment_percent + prog, 2)
    return finalChallengeProgress

async def challengeBot(challenge, challengeProgress, target, increment, user_input, personalities, background):
    absolute_value = calculate_absolute(challengeProgress, target, increment)
    isNegative = increment < 0.0

    c_i = (f"The challenge is this: {challenge}. The user just progressed by: {increment}, "
    f"and the final challenge progress is now: {absolute_value}. Original user input: {user_input}")

    if isNegative:
        prompt = ("The user has voluntarily deducted progress from his challenge. Applaud his honesty "
                          f"and provide an update on his challenge progress. {c_i}")
           
    elif not isNegative and challengeProgress > 100:
        prompt = (f"The user is surpassing his challenge. Congratulate his initiative. {c_i}")

    else:
        prompt = (f"The user is progressing on his challenge. Encourage him!!!. {c_i}")
        
    response = await OpenAI_llm.ainvoke(prompt)
    finalprog = calculate_final_progress(challengeProgress, target, increment)
    return [response.content, finalprog]
    