import os
import json

from dotenv import load_dotenv
from pathlib import Path

def vector_store_call(data):
    from main import firestore_db as db
    
    agent_id = data['agent_id']
    
    doc = db.collection("agents").document(agent_id).get()
    agent_data = doc.to_dict()
    if agent_data:
        vector_store = agent_data['vector_store']
        return vector_store
    else:
        return (
            {
            "domain_kb_column_name": None,
            "domain_kb_name": None,
            "product_kb_column_name": None,
            "product_kb_name": None,
            "vector_store_id": None
        }
        )