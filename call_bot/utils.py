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

def construct_system_prompt(config, retrieved_context: str, summary: str, challenge: str) -> str:
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

    return (
        f"Summary of message so far: {summary}\n"
        f"You are {name} ({gender}).\nThis is your background: {background}\n"
        f"Use this as contextual information:\n{retrieved_context}\n"
        f"These are your personalities: {personalities}\n"
        "When communicating with the user, remember to stay in character. "
        f"The user currently has the in-game challenge: {challenge}. Just keep that in mind"
    )