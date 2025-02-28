import os
import json
import weaviate
import re

from typing import Dict, List
from dotenv import load_dotenv
from pathlib import Path

from flask import Flask, request, jsonify
from pydantic import BaseModel, ValidationError

from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore
from weaviate.classes.init import Auth

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
script_dir = Path(__file__).resolve().parent
env_path = script_dir.parent / '.env'
load_dotenv(dotenv_path=env_path)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
WEAVIATE_CLIENT_ID = os.environ.get("WEAVIATE_CLIENT_ID")
WEAVIATE_API_KEY = os.environ.get("WEAVIATE_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Initialize Weaviate client
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_CLIENT_ID,  # Replace with your Weaviate Cloud URL
    auth_credentials=Auth.api_key(WEAVIATE_API_KEY),  # Replace with your Weaviate Cloud key
    headers={'X-OpenAI-Api-key': OPENAI_API_KEY}  # Replace with your OpenAI API key
)

# Define LLM and retriever
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")
vectorstore = WeaviateVectorStore(client, "Psychology_KB", "content", embeddings)
retriever = vectorstore.as_retriever()

# Define API input model
class UserInput(BaseModel):
    user_query: str
    intent: str
    emotion: str
    product_info: str = ""  # Optional field
    domain_info: str = ""   # Optional field

# Define API output model
class AgentResponse(BaseModel):
    response: str
    justification: str
    follow_up_actions: List[str]
    
# Initialize Flask app
app = Flask(__name__)

prompt_template = """You are a highly intelligent and persuasive sales assistant specializing in understanding user intent, emotion, and context to craft responses that are both empathetic and persuasive. Your responses must be precise, psychologically sound, and directly aligned with the user’s needs while maintaining ethical integrity.

You are an expert in the following psychological techniques, which should be used where appropriate:

- **Neuro-Linguistic Programming (NLP):** Using language patterns to shape perception and influence decisions.
- **Cognitive Reframing:** Guiding users to reinterpret their thoughts and emotions in a way that reduces hesitation and builds confidence.
- **Ericksonian Hypnosis:** Applying subtle, indirect suggestion techniques to reinforce positive decision-making.
- **Behavioral Economics:** Leveraging principles such as social proof, scarcity, and loss aversion to encourage action.
- **Rapport Building:** Establishing trust by mirroring user concerns and validating their emotions.
- **Adaptive Communication:** Adjusting tone and phrasing dynamically based on the user’s intent, emotion, and level of engagement.
- **Objection Handling:** Addressing doubts by reinforcing key benefits and guiding the user toward resolution.
- **Goal Orientation:** Keeping the conversation focused on achieving a positive outcome while maintaining transparency.

You will receive the following information:

- **User Query**: {user_query}
- **Intent**: {intent}
- **Emotion**: {emotion}
- **Product Info**: {product_info}
- **Domain Info**: {domain_info}
- **Retrieved Techniques**: 
    {retrieved_techniques} (Persuasion techniques retrieved from the knowledge base, which must be incorporated into the response.)

Your task is to generate a structured JSON response containing the following fields:

1. **"response"** (String): A well-structured, persuasive, and relevant reply that:
    - Acknowledges and validates the user’s emotion.
    - Directly addresses the user’s intent.
    - Uses the most relevant persuasion techniques from `retrieved_techniques`.
    - Leverages `product_info` and `domain_info` where applicable.
    - Encourages the next step without applying pressure.
    - If `product_info` or `domain_info` is missing, adjusts the response accordingly.

2. **"justification"** (String): A detailed explanation of the persuasion techniques used in the response. The structure should be:

   1. **[technique_name]** (From retrieved_techniques)
      - **Technique:** Brief explanation.
      - **Keyphrase Used:** The specific phrase in the response where this technique was applied.
      - **Why:** The reasoning behind using this technique in this context.

    The justification must only reference techniques found in `retrieved_techniques`.

3. **"follow_up_actions"** (Array of Strings): Suggested next steps to further assist the user. These should be relevant to the user’s needs and leverage `product_info` and `domain_info` where applicable.

**Key Considerations:**
- **Precision and Relevance:** Every response must be direct, clear, and tailored to the user's needs.
- **Ethical Persuasion:** Influence must be applied transparently and honestly.
- **Adaptive Communication:** Responses should align with the user’s tone and engagement level.
- **Consistency:** The reasoning in "justification" must clearly connect to the techniques used in "response."
- **Valid JSON Output:** The response must be structured as a valid JSON object containing only "response", "justification", and "follow_up_actions".
"""

def remove_markdown_code_block(input_str):
    start_marker = "```json"
    end_marker = "```"

    result = input_str

    if result.startswith(start_marker):
        result = result[len(start_marker):]

    if result.endswith(end_marker):
        result = result[:-len(end_marker)]

    return result.strip()

# ✅ **1. Query Optimization**
def generate_optimized_query(user_query: str, intent: str, emotion: str) -> str:
    """Rewrites the user query into a structured retrieval query for better accuracy."""
    return f"Retrieve psychology techniques for '{emotion}' related to '{intent}': {user_query}"

# ✅ **2. Multi-Step Retrieval (Retrieve More, Then Filter)**
def retrieve_and_filter_techniques(user_query: str, intent: str, emotion: str):
    """Retrieves psychology techniques and filters for highest relevance."""
    
    # Step 1: Generate optimized query
    optimized_query = generate_optimized_query(user_query, intent, emotion)

    # Step 2: Retrieve broader results (Top 10 techniques)
    retrieved_docs = retriever.invoke(optimized_query)

    # Step 3: Return all retrieved documents without filtering
    return retrieved_docs

# ✅ **3. Re-Ranking Retrieved Techniques Using GPT**
def rerank_retrieved_docs(retrieved_docs, query):
    """Uses GPT to re-rank retrieved psychology techniques based on relevance."""

    if not retrieved_docs:
        return []

    # Create a ranking prompt for GPT
    ranking_prompt = f"Rank the following psychology techniques based on how relevant they are to: '{query}'\n\n"
    ranking_prompt += "\n".join([f"{i+1}. {doc.page_content}" for i, doc in enumerate(retrieved_docs)])

    # Use GPT to rank the documents
    ranking_response = llm.invoke(ranking_prompt).content
    print(ranking_response)  # Check the raw response from the LLM

    # Now parse the ranking response safely
    try:
        ranked_docs = []
        lines = ranking_response.split("\n")

        # Look for lines that contain a ranking and extract the correct document indices
        for line in lines:
            if line.strip():  # Skip empty lines
                parts = line.split(".")
                if len(parts) >= 2 and parts[0].isdigit():
                    index = int(parts[0]) - 1  # Convert to zero-based index
                    if index < len(retrieved_docs):
                        ranked_docs.append(retrieved_docs[index])

        # Return top 3 ranked results
        return ranked_docs[:3]

    except Exception as e:
        print(f"Error while parsing ranking response: {e}")
        # If something goes wrong, return an empty list or fallback docs
        return retrieved_docs[:3]  # Fallback to the first 3 docs if parsing fails

# ✅ **Generate Final Response**
def create_rag_response(user_input: UserInput) -> Dict:
    """Creates a RAG-enhanced response for the user input."""
    
    # Step 1: Retrieve psychology techniques directly (without re-ranking)
    retrieved_techniques = retrieve_and_filter_techniques(user_input.user_query, user_input.intent, user_input.emotion)

    # Step 2: Format retrieved techniques (no need to re-rank)
    techniques_text = "\n\n".join([f"- {doc.page_content}" for doc in retrieved_techniques])

    # # Step 1: Retrieve and filter psychology techniques
    # retrieved_techniques = retrieve_and_filter_techniques(user_input.user_query, user_input.intent, user_input.emotion)

    # # Step 2: Re-rank techniques using GPT
    # best_techniques = rerank_retrieved_docs(retrieved_techniques, user_input.user_query)

    # # Step 3: Format retrieved techniques
    # techniques_text = "\n\n".join([f"- {doc.page_content}" for doc in best_techniques])

    # Step 4: Format prompt for LLM
    prompt = prompt_template.format(
        user_query=user_input.user_query,
        intent=user_input.intent,
        emotion=user_input.emotion,
        product_info=user_input.product_info or "Not provided",
        domain_info=user_input.domain_info or "Not provided",
        retrieved_techniques=techniques_text
    )

    # Generate response using LLM
    response_message = llm.invoke(prompt)  
    
    response_text = response_message.content
    response_json = json.loads(remove_markdown_code_block(response_text))
    return response_json

    try:
        response_json = json.loads(response_text)  # Ensure valid JSON output
    except json.JSONDecodeError:
        response_json = {
            "response": "I'm sorry, but I couldn't generate a structured response at this time.",
            "justification": "AI failed to produce valid JSON. Please try again.",
            "follow_up_actions": []
        }

    return response_json

# def retrieve_techniques(query: str) -> str:
#     """Fetches relevant psychology techniques from Weaviate based on the user query."""
#     retrieved_docs = retriever.get_relevant_documents(query)
#     if not retrieved_docs:
#         return "No specific techniques found. Using general persuasive strategies."
    
#     techniques = "\n\n".join([f"- {doc.page_content}" for doc in retrieved_docs])
#     return f"**Techniques Used:**\n{techniques}"

# def create_rag_response(user_input: UserInput) -> Dict:
#     """Creates a RAG-enhanced response for the user input."""
#     # Retrieve relevant psychology techniques from the vector database
#     retrieved_techniques = retrieve_techniques(user_input.user_query)

#     # Format the final prompt with retrieved techniques
#     prompt = prompt_template.format(
#         user_query=user_input.user_query,
#         intent=user_input.intent,
#         emotion=user_input.emotion,
#         product_info=user_input.product_info or "Not provided",
#         domain_info=user_input.domain_info or "Not provided",
#         retrieved_techniques=retrieved_techniques
#     )

#     # Generate response using LLM
#     response_message = llm.invoke(prompt)
#     print(response_message)
#     response_text = response_message.content 
#     return response_text
    
    
#     response_json = json.loads(response_text)
#     # except json.JSONDecodeError:
#     #     response_json = {
#     #         "response": "I'm sorry, but I couldn't generate a structured response at this time.",
#     #         "justification": "AI failed to produce valid JSON. Please try again.",
#     #         "follow_up_actions": []
#     #     }

#     return response_json

@app.route('/generate-response', methods=['POST'])
def generate_response():
    try:
        # Parse request JSON
        data = request.get_json()
        user_input = UserInput(**data)
        
        # Generate response using RAG pipeline
        response = create_rag_response(user_input)

        # Return structured JSON response
        #return jsonify(response)
        return response
    
    except ValidationError as e:
        return jsonify({"error": "Invalid input", "details": e.errors()}), 400
    except Exception as e:
        return jsonify({"error": "An error occurred", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
    client.close()