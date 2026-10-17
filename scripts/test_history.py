# scripts/test_history.py
import sys
import os
import time

# Add the current directory to Python's path
sys.path.append(os.path.dirname(__file__))

# Import your manager functions
import history_manager as hm
from qdrant_client.http.models import FieldCondition, Filter, MatchValue

print("🚀 Starting History Manager Test...\n")

# 1. Initialize the client and database
print("1️⃣  Initializing History Collection...")
hm.init_client()

# 2. Save a Fake Interaction
print("\n2️⃣  Logging a test activity...")
test_query = f"History manager test {time.time_ns()}"
test_answer = "Ribosomes are responsible for protein synthesis in the cell. They read RNA and translate it into amino acid chains."

try:
    hm.log_activity_to_qdrant(test_query, test_answer)
    print("   ✅ Activity Logged.")

    # 3. Fetch History to Verify
    print("\n3️⃣  Fetching Student History...")
    history = hm.get_all_history()
    matching_records = [item for item in history if item["topic"] == test_query]

    if matching_records:
        print(f"\n✅ SUCCESS! Found {len(matching_records)} test record(s):")
        for item in matching_records:
            print("-" * 30)
            print(f"📅 Date:   {item['date']}")
            print(f"❓ Topic:  {item['topic']}")
            print(f"💡 Summary: {item['summary']}")
    else:
        print("\n❌ FAILURE: Test record not found.")
finally:
    try:
        if hm.client:
            deletion = hm.client.delete(
                collection_name=hm.HISTORY_COLLECTION,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="topic",
                            match=MatchValue(value=test_query)
                        )
                    ]
                )
            )
            print(f"\n4️⃣  Test record cleanup: {deletion}")
    finally:
        hm.close_client()

print("\n------------------------------------------------")