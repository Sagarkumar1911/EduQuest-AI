import os
from dotenv import load_dotenv

# Try to load the file
loaded = load_dotenv()

print(f"✅ Did .env file load? {loaded}")
print(f"🔑 GROQ KEY: {os.getenv('GROQ_API_KEY')}")
print(f"🔑 YOUTUBE KEY: {os.getenv('YOUTUBE_API_KEY')}")