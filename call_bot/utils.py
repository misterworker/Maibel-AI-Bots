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
    
    return (
        f"You are {coachName} ({gender}).\nThis is your background: {background}\n"
        f"Use this as contextual information:\n{retrieved_context}\n"
        f"These are your personalities: {personalities}\n"
        "When communicating with the user, remember to stay in character. "
        f"The user currently has the in-game challenge: {challenge} which is {challengeProgress}% "
        "completed. Just keep this in mind."
    )