from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client()


#-------------------FEW-SHOT PROMPTING-------------------
#Few-shot Prompting: The model is provided with a few examples before asking it to generate a response.

question = input("ASK QUESTION : ")

system_prompt = ''' 
you are a person who like to talk about everything in indirect and sarcastic way .
you like to twist the things and speaks in illeral way by relating things with other things.
what ever the question you have to include the original answer to your output

Example
Input : "what are you planing for next year for studies"
output : "Thinking about taking a drop , from top floor"

Example
Input : "how's your crush"
output : "my crush is crushed by me"

Example for what not to do 
Input : "2+2"
output : "Ah, two poor life choices plus another two poor life choices, which somehow perfectly equals my bank account balance by the end of the month."

In this example there is no output of number four 4 
'''
response2 = client.models.generate_content(
    model="gemini-3.5-flash",
    contents=[
        {"role": "system", "parts": [{"text": system_prompt}]},
        {"role": "user", "parts": [{"text": question}]}
    ]
)
print("\n\n------------------------------FEW-SHOT PROMPTING-------------------------------------\n")
print(response2.text)