import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OpenAI_llm = ChatOpenAI(
    model="gpt-4o-mini-2024-07-18",
    temperature=5,
    max_tokens=100,
    timeout=None,
    max_retries=2,
    api_key=OPENAI_API_KEY,
)

def calculate_absolute(prog, target, curIncrement):
    cur_absolute = prog*0.01*target
    return cur_absolute + curIncrement

async def challengeBot(challenge, challengeProgress, target, increment):
    absolute_value = calculate_absolute(challengeProgress, target, increment)
    isNegative = increment < 0.0

    c_i = (f"The challenge is this: {challenge}. The change in progress is this: {increment}, "
    f"and the challenge progress in absolute value is now: {absolute_value}")

    if isNegative:
        prompt = ("The user has voluntarily deducted progress from his challenge. Applaud his honesty "
                          f"and provide an update on his challenge progress. {c_i}")
           
    elif not isNegative and challengeProgress > 100:
        prompt = (f"The user is surpassing his challenge. Congratulate his initiative. {c_i}")

    else:
        prompt = (f"The user is progressing on his challenge. Encourage him!!!. {c_i}")
        
    response = await OpenAI_llm.ainvoke(prompt)
    return response.content
    