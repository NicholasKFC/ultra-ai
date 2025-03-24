import os
import json

from dotenv import load_dotenv
from pathlib import Path

from Orchestrator_Agent.orchestrator import orchestrator_call
from Response_Agents.no_context_agent import no_context_call
from chat_memory import ChatMemoryManager

conversation_memory = ChatMemoryManager(max_messages=10)

def chat_call(data):
    # Step 1: Load chat history
    global conversation_memory
    chat_history = conversation_memory.to_langchain_messages()
    
    # Step 2: Call orchestrator agent
    orchestrator_results = orchestrator_call(data, chat_history)
    
    # Step 3: Call response agent
    response_call_values = orchestrator_results.copy()
    user_query = data['query']
    response_call_values['user_query'] = user_query
    response_results = no_context_call(response_call_values)
    
    # Step 4: Save chat history
    print(response_results)
    response = response_results['response']
    conversation_memory.add("human", user_query)
    conversation_memory.add("ai", response)
    
    # Step 5: Output results
    output_json = response_results
    return output_json