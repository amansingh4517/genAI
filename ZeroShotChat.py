from dotenv import load_dotenv
import json
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv() # This loads the variables from your .env file into the system
client = genai.Client()


#-------------------ZERO-SHOT PROMPTING-------------------
# Zero-shot Prompting: The model is given a direct question or task without prior examples.

question = input("ASK QUESTION : ")

response1 = client.models.generate_content(
    model="gemini-3.5-flash",
    contents=question
)
print("-----------------------ZERO-SHOT PROMPTING----------------------\n\n")
print(response1.text + "\n\n\n\n")
