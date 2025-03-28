import os
import json

from dotenv import load_dotenv
from pathlib import Path

from Orchestrator_Agent.orchestrator import orchestrator_call
from Response_Agents.no_context_agent import no_context_call
from chat_memory import ChatMemoryManager
from chat_session import add_message, to_langchain_messages

conversation_memory = ChatMemoryManager(max_messages=10)

def chat_call(data):
    session_id = data['session_id']
    user_query = data['query']
    chat_history = data['chat_history']
    
    # Step 1: Load chat history
    global conversation_memory
    
    if chat_history:
        chat_history = to_langchain_messages(chat_history)
    else:
        chat_history = conversation_memory.to_langchain_messages()

    # Step 2: Call orchestrator agent
    orchestrator_results = orchestrator_call(data, chat_history)
    
    # Step 3: Call response agent
    response_call_values = orchestrator_results.copy()
    response_call_values['user_query'] = user_query
    response_results = no_context_call(response_call_values)
    
    # Step 4: Save chat history
    print(response_results)
    response = response_results['response']
    conversation_memory.add("human", user_query)
    conversation_memory.add("ai", response)
    
    # Step 5: Save to session
    add_message(session_id, "human", user_query)
    add_message(session_id, "ai", response)
    
    # Step 6: Output results
    output_json = response_results
    return output_json