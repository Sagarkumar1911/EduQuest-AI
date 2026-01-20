import os
from dotenv import load_dotenv

print("🔍 SEARCHING FOR .ENV FILE...")

# 1. List all files to see the REAL names (reveals hidden .txt)
files = os.listdir()
env_found = False

for f in files:
    if f.startswith(".env"):
        print(f"   -> Found file named: '{f}'")
        if f == ".env":
            env_found = True
        elif f == ".env.txt":
            print("   ⚠️ RED ALERT: Your file is named .env.txt! Rename it to just .env")

# 2. Try to force load it
if env_found:
    print("\n✅ Loading .env file...")
    load_dotenv()
    
    key = os.getenv("GROQ_API_KEY")
    if key:
        print(f"   🎉 SUCCESS! Key found: {key[:10]}...")
    else:
        print("   ❌ File loaded, but KEY is missing. Check inside the file.")
        print("   Make sure it looks like: GROQ_API_KEY=gsk_...")
else:
    print("\n❌ ERROR: No exact '.env' file found. Check the name again.")