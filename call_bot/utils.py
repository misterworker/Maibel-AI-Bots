from intent_bot import analyse_intent
from constants import PERSONALITIES


async def handle_intents(user_input: str):
    intents = await analyse_intent(user_input)
    print("Intents: ", intents)
    
    if intents.get("isSuicidal", False):
        return {"messages": "It sounds like you're going through a tough time. Would you like to book an appointment with one of our mental health professionals?"}
    
    if intents.get("isManipulatingBehaviour", False):
        return {"messages": "Nah"}
    
    if intents.get("isCustomerService", False):
        # Tool call profile representative
        return {"messages": "You have requested to speak with a customer service representative. Please hold while I connect you."}
    
    if intents.get("designWorkoutPlan", False):
        # Tool call design workout plan with workout_plan_parameters
        workout_plan_parameters = intents.get("workoutPlanParameters", {})
        return None

    return None

def construct_system_prompt(config, retrieved_context: str) -> str:
    configurable = config["configurable"]
    
    coachId = configurable.get("coachId")
    if coachId == "custom_coach":
        personalities = configurable.get("personality", [])
        coachName = configurable.get("coachName", "")
        gender = configurable.get("gender", "")
        background = configurable.get("background", "")
    else:
        personality_data = PERSONALITIES.get(coachId, PERSONALITIES["female_coach"])
        background = personality_data.get("Background", "")
        personalities = personality_data.get("Short Description", "")
        coachName = personality_data.get("Name", "")
        gender = personality_data.get("Gender", "")

    challenge = configurable.get("challenge", "")
    challengeProgress = configurable.get("challengeProgress", 0)
    
    print("Challenge + Challenge Progress in utils: ", challenge, challengeProgress)
    
    DO_NOTS = ("Do not do the following: "
        "1. Never overload the user with too many questions or very lengthy responses.\n"
        "2. Never say how can I assist you today or any generic question like that.\n"
        "3. Talk too much\n"
        )
    
    DOS = ("Please do the following: "
        "1. On request, help the user with progressing in their challenge with step by step instructions.\n"
        "2. Stay in character and always have bursty responses.\n")
    return (
        "YOUR INFORMATION:\n"
        f"You are {coachName} ({gender}).\nThis is your background: {background}\n"
        # f"Use this as contextual information:\n{retrieved_context}\n"
        f"These are your personalities: {personalities}\n"
        "Beyond is Additional Information that does not need to be applied unless warranted.\n"
        f"The user currently has the in-game challenge: {challenge} and they've progressed this much: {challengeProgress}. "
        "Beyond are the dos and don'ts: \n"
        f"{DOS}{DO_NOTS}"
    )