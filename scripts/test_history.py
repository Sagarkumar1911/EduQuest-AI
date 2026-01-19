# scripts/test_history.py
import sys
import os

# Add the current directory to Python's path
sys.path.append(os.path.dirname(__file__))

# Import your manager functions
from history_manager import init_history_db, log_activity_to_qdrant, get_qdrant_history

print("🚀 Starting History Manager Test...\n")

# 1. Initialize the Database
print("1️⃣  Initializing History Collection...")
init_history_db()

# 2. Save a Fake Interaction
print("\n2️⃣  Logging a test activity...")
test_query = "What is the function of the ribosome?"
test_answer = "Ribosomes are responsible for protein synthesis in the cell. They read RNA and translate it into amino acid chains."

log_activity_to_qdrant(test_query, test_answer)
print("   ✅ Activity Logged.")

# 3. Fetch History to Verify
print("\n3️⃣  Fetching Student History...")
history = get_qdrant_history()

if history:
    print(f"\n✅ SUCCESS! Found {len(history)} record(s):")
    for item in history:
        print("-" * 30)
        print(f"📅 Date:   {item['date']}")
        print(f"❓ Topic:  {item['topic']}")
        print(f"💡 Summary: {item['summary']}")
else:
    print("\n❌ FAILURE: No history found. Something is wrong.")

print("\n------------------------------------------------")