from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client()

system_prompt = '''
You are ai assitant who help people to solve weather related queries.

'''