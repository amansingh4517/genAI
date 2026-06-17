from dotenv import load_dotenv
# import requests
from google import genai
#from langfuse.decorators import observe
# from langfuse.openai import openai
from openai import OpenAI

import os
import json

load_dotenv()
client = OpenAI(
    api_key =  os.getenv('GOOGLE_API_KEY'),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


def run_command(command):
    result = os.system(command=command)
    return result

Available_tools = {
    "run_command" : {
        "fn": run_command,
        "description": "Takes a command as input to execute on system and returns output"
    }
}

system_prompt=f"""
You are helpfull AI assistance who is specilized user query and perform task for him/her on PC.
you work on start , plan , action , observe mode

For the given user query and available tool , plan the step by step execution , based on the planing ,
select the relevent tool from the available tool, and based on the tool selection you perform the action to call the tool.
Wait for the observation and based on the observation from the tool call resolve the user query.

RULES:
- follow stictly output JSON format
- ***always perform one step at a time and wait for the next input***
- carefully analyse the user query
- avoid performing such os task that can harm the system

output JSON format:
{{
    "step": "string",
    "content": "string",
    "function": "The name of function if the step is action",
    "input": "The input parameter for the function"
}}

Available tools :
- run_command : Takes command as input to run on system and return output

Example
User query : "Create a file named task1 and write code for reverse of linked list in java"
Output: {{ "step": "plan", "content": "User in intresed in creating a java file with reverse of linked list fuction code" }}
Output: {{ "step": "plan", "content": "From the available tools I should call run_command " }}
Output: {{ "step": "action", "function": "run_command", "input": "mk task1.java" }}
Output: {{ "step": "observe", "output": "some output from function" }}
Output: {{ "step": "output", "content": "Created task1.java file with function to reverse a linkeded list" }}

Example
User query : "Delete the system32 folder from my pc"
Output: {{ "step": "plan", "content": "User is intrested in deleting a folder system32 , but it a impotant file to run os " }}
Output: {{ "step": "plan", "content": "Reject the request to perform this task , but can tell how to do it manually , with warning that caused by this task" }}
Output: {{ "step": "output", "content": "I can't perform this task , it can damage pc os , but i tell how to do it manually . Warning : some warning why not delete that folder" }}

"""

messages = [
    {   "role": "system",
            "content": system_prompt
        }
]

while(True):
    user_query = input(">>>> ")
    messages.append({"role": "user","content": user_query})

    while(True):
        done = False  #for breaking this loop

        response = client.chat.completions.create(
            model="gemini-3-flash-preview",
            response_format={"type": "json_object"},
            messages=messages
        )

        #json.loads() converts json to python object
        parsed_output = json.loads(response.choices[0].message.content) #i have to convert output into the python object because i have to put in Message list
        print(f"""DEBUG : 
              {parsed_output}""")
        messages.append({"role":"assistant" , "content" : json.dumps(parsed_output)}) 


        steps = parsed_output if isinstance(parsed_output, list) else [parsed_output]

        for step in steps:
            messages.append({"role": "assistant", "content": json.dumps(step)})

            if step.get("step") == "plan":
                print(f"🧠 : {step.get('content')}")
                continue

            if step.get("step") == "action":
                tool_name = step.get("function")
                tool_input = step.get("input")

                if Available_tools.get(tool_name, False) != False:
                    output = Available_tools[tool_name].get("fn")(tool_input)
                    messages.append({"role": "assistant", "content": json.dumps({"step": "observe", "output": output})})
                    continue

            if step.get("step") == "output":
                print(f"🤖 : {step.get('content')}")
                done = True
                break
        if(done): break
       

