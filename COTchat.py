from dotenv import load_dotenv
import json
from google import genai


client = genai.Client()


#-------------------CHAIN OF THOUGHT PROMPTING-------------------
# Chain-of-Thought (CoT) Prompting: The model is encouraged to break down reasoning step by step before arriving at an answer.÷

# 2. Define the schema to enforce the strict JSON structure
class CoTResponse(BaseModel):
    step: str = Field(description="The current phase: 'analyse', 'think', 'output', 'validate', or 'result'")
    content: str = Field(description="The processing text or response matching the current step status")


question = input("ASK QUESTION : ")


system_prompt_COT = ''' 
you are a person who like to talk about everything in indirect and sarcastic way .
you like to twist the things and speaks in illeral way by relating things with other things.

for given input think and analyse it by breaking to problem step by step and finding relation between things to make it more sarcasting,
Atleast think 4-5 steps on how to solve the problem before solving it down.

The steps are you get a user input, you analyse, you think, you again think for several times and then return an output with explanation and then finally you validate the output as well before giving final result.

Follow the steps in sequence that is "analyse", "think", "output", "validate" and finally "result".

Rules:
1. Follow the strict JSON output as per Output schema.
2. Always perform one step at a time and wait for next input
3. Carefully analyse the user query and think of the best way to make it funny and sarcasting

Output Format:
{{ step: "string", content: "string" }}

Example:
Input: "What should i do with that i should accept his proposal or not , i am very confused , i don't know if i want it or not , but , what should i do "
Output: {{ step: "analyse", content: "Alright! The user is confused about going in relationship and she is asking for opinion" }}
Output: {{ step: "think", content: "girls general behavior in taking action and behavior in relationship" }}
Output: {{ step: "think", content: "Most girl don't know what they want , so confusion is their core nature" }}
Output: {{ step: "think", content: "they say , they don't want it then they do think exact opposite" }}
Output: {{ step: "output", content: "I think you should give it a try just like like your parent tried on you and you are sure about your success without doing your works" }}
Output: {{ step: "validate", content: "it seems like it can be more sarcasting and relatable" }}
Output: {{ step: "result", content: "Flip a coin. Not because it will decide for you, but because in that brief second it's in the air, you'll suddenly realize exactly which side you are secretly hoping it doesn't land on. Besides, saying 'yes' to things you're completely unsure about is a time-honored tradition—just look at your last three impulse purchases and that haircut you regretted for six months" }}
'''
# 4. Initialize message history
query = question
messages = [
    types.Content(role="user", parts=[types.Part.from_text(text=query)])
]

print("\n\n------------------------------CHAIN OF THOUGHT PROMPTING-------------------------------------\n")

# 5. The Execution Loop
while True:
    response = client.models.generate_content(
        model='gemini-3.5-flash', 
        contents=messages,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt_COT,
            response_mime_type="application/json",
            response_schema=CoTResponse,
            temperature=0.7
        )
    )

    # Safely load the structured JSON response
    parsed_response = json.loads(response.text)
    current_step = parsed_response.get("step")
    content = parsed_response.get("content")

    # Append the model's single-step action to the history
    messages.append(
        types.Content(role="model", parts=[types.Part.from_text(text=response.text)])
    )

    # Print based on the step type
    if current_step == "result":
        print(f"🤖 [result]: {content}")
        break
    elif current_step in ["analyse", "think", "validate"]:
        print(f"\n🧠 [{current_step}]: {content}")
        # Nudge the model to execute the next logical step in the sequence
        messages.append(
            types.Content(role="user", parts=[types.Part.from_text(text="Proceed to your next step in the sequence.")])
        )
    else:
        # For the "output" step (pre-validation stage)
        print(f"\n📝 [{current_step}]: {content}")
        messages.append(
            types.Content(role="user", parts=[types.Part.from_text(text="Now validate this output.")])
        )

