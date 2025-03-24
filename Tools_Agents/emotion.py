import os
import json
from typing import Dict, List, Union

from dotenv import load_dotenv
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool

# Load environment variables
script_dir = Path(__file__).resolve().parent
env_path = script_dir.parent / '.env'
load_dotenv(dotenv_path=env_path)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
WEAVIATE_CLIENT_ID = os.environ.get("WEAVIATE_CLIENT_ID")
WEAVIATE_API_KEY = os.environ.get("WEAVIATE_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

def emotion_call(data: Dict[str, str]) -> str:  # Expecting Dict[str, str] and returning a JSON string
    """Analyzes user query for intent and emotion."""

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    template = """
    You are an intent and emotion analyzer. Based on the user query, identify the following:
**Emotions:**
Identify all emotions conveyed in the message and express them as percentages to represent their intensity. Use the following emotions:  
    - Happiness  
    - Surprise  
    - Contempt  
    - Sadness  
    - Fear  
    - Disgust  
    - Anger  
    - Amusement  
    - Contentment  
    - Embarrassment  
    - Excitement  
    - Guilt  
    - Pride in Achievement  
    - Relief  
    - Satisfaction  
    - Shame  

Ensure that the total percentage adds up to 100%.
Output your analysis of emotions where you sort the emotions by the highest pecentage.

**Intent:**
Determine whether it is one of the following:
- Buying Intent
- Greetings
- Objection
- Satisfication
- Others (The intent determinded)

**Output Format:** Return a JSON object with the following structure:
```
    {{
        "intent_analysis": "Output your analysis of the user query using the emotions and the intent and express it in a humanlike way.",
        "intent": 'intent_string',
        "emotion": {{"Amusement":0,"Anger":0,"Contempt":0,"Contentment":0,"Disgust":0,"Embarrassment":0,"Excitement":0,"Fear":0,"Guilt":0,"Happiness":0,"Pride in Achievement":0,"Relief":0,"Sadness":0,"Satisfaction":0,"Shame":0,"Surprise":0}}
    }}
```
(e.g., {{"intent_analysis": "The user intends to buy an item", "intent": "Buying Intent", "emotion": {{"Amusement":0,"Anger":0,"Contempt":0,"Contentment":0,"Disgust":0,"Embarrassment":0,"Excitement":0,"Fear":0,"Guilt":0,"Happiness":0,"Pride in Achievement":0,"Relief":0,"Sadness":0,"Satisfaction":0,"Shame":0,"Surprise":0}}}}).
    
    User Query: {input}
    """

    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"input": RunnablePassthrough()} # Changed "question" to "input"
        | prompt
        | llm
        | StrOutputParser()
    )
    response_message = rag_chain.invoke(data)
    return response_message


@tool
def emotion_tool(query: str) -> str:
    """This tool analyzes user queries to understand their underlying intent and emotional state. Use this tool when you need to determine why a user is asking a question and how they are feeling while asking it. This information can be crucial for tailoring responses, prioritizing requests, and improving user experience. The input to this tool is the user's query in text. The response from this tool is a JSON object containing the identified intent and emotion."""
    return emotion_call({"input": query})