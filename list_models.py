import requests
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("NO KEY")
    exit(1)

url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
resp = requests.get(url)
models = resp.json().get("models", [])
for m in models:
    if "flash" in m["name"].lower():
        print(m["name"])
