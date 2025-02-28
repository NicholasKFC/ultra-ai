import os
import json

from dotenv import load_dotenv
from pathlib import Path
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredMarkdownLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
script_dir = Path(__file__).resolve().parent
env_path = script_dir.parent / '.env'
load_dotenv(dotenv_path=env_path)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Define API input model
class UserInput(BaseModel):
    user_query: str
    intent: str
    emotion: dict
    product_info: str = ""  # Optional field
    domain_info: str = ""   # Optional field
    irrelevant_query: bool = False  
    clarification_needed: bool = False

def load_documents_from_directory(directory_path):
    """Load and extract text from all supported document types in a specified directory, 
    adding 'Start of <document name>' and 'End of <document name>' markers."""
    
    docs = []
    supported_extensions = {".pdf", ".txt", ".docx", ".md"}

    # Iterate through all files in the given directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        # Check if it's a file and has a supported extension
        if os.path.isfile(file_path) and any(file_path.endswith(ext) for ext in supported_extensions):
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith(".txt"):
                loader = TextLoader(file_path)
            elif file_path.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            elif file_path.endswith(".md"):
                loader = UnstructuredMarkdownLoader(file_path)
            else:
                print(f"Unsupported file format: {filename}")
                continue
            
            # Load the document
            loaded_docs = loader.load()
            
            # Wrap text content with start and end markers
            for doc in loaded_docs:
                doc.page_content = f"#Start of {filename}#\n\n{doc.page_content}\n\n#End of {filename}#"
                docs.append(doc)
    
    return docs

def remove_markdown_code_block(input_str):
    start_marker = "```json"
    end_marker = "```"

    result = input_str

    if result.startswith(start_marker):
        result = result[len(start_marker):]

    if result.endswith(end_marker):
        result = result[:-len(end_marker)]

    return result.strip()

def large_context_call(data):
    from main import client
    
    user_input = UserInput(**data)
    
    # Define LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    
    directory_path = "./Response_Agents/Psy Techniques"
    documents = load_documents_from_directory(directory_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=16000, chunk_overlap=500)
    chunks = text_splitter.split_documents(documents)

    # Combine chunks into a single long context
    context = "\n\n".join([chunk.page_content for chunk in chunks])

    prompt_template = """You are a highly intelligent and persuasive sales assistant specializing in understanding user intent, emotion, and context to craft responses that are both empathetic and persuasive. Your responses must be precise, psychologically sound, and directly aligned with the user’s needs while maintaining ethical integrity.

    This forms a comprehensive knowledge base for training an AI agent in persuasive communication techniques, especially within text-based sales environments, the beginning of the knowledge base and the end of the knowledge base will both be marked with a '####' to indicate the beginning and ending:
    ####
    {context}
    ####

    You will receive the following information:

    - **User Query**: {user_query}
    - **Intent**: {intent}
    - **Emotion**: {emotion}
    - **Product Info**: {product_info} (This field may contain details of one or more products, including name, description, and a direct purchase link.  If no products are relevant, it will be empty.)
    - **Domain Info**: {domain_info}
    - **Irrelevant Query**: {irrelevant_query} (Boolean: true if the query is outside the scope of the sales assistant, false otherwise)
    - **Clarification Needed**: {clarification_needed} (Boolean: true if the agent needs more information to accurately address the user's query, false otherwise)

    Your task is to generate a structured JSON response containing the following fields:

    1.  **"response"** (String): A well-structured, persuasive, and relevant reply that:
        1.  Acknowledges and validates the user’s emotion.
        2.  Directly addresses the user’s intent.
        3.  Uses the most relevant persuasion techniques from the embedded knowledge base.
        4.  Leverages `product_info` and `domain_info` where applicable.
            *   **Crucially, if `product_info` contains product details, include a concise product description and the direct purchase link in the response.** Present multiple products in a clear, list format.
        5.  Offers recommendations, including links.
        6.  Encourages the next step without applying pressure.
        7.  If `clarification_needed` is true, the response should be a concise and relevant clarifying question.
        8.  If `irrelevant_query` is true, the response should politely decline to answer the question, explaining that the query is outside the scope of the sales assistant. For example: "I'm designed to assist with product information and sales inquiries, so I'm unable to answer general knowledge questions like that. However, I'd be happy to help with any questions about [mention relevant product categories or domain]."
        
    2.  **"justification" (String):** A detailed explanation of the persuasion techniques used in the response. **The justification MUST adhere strictly to the following format.  Each technique used should be separated by a new line. If no techniques are used, set "justification" to "No persuasion techniques used.":**

        ```
        [**technique_name**]
        **Technique:**
        Brief explanation of the technique.
        
        **Keyphrase Used:**
        The specific phrase in the response where this technique was applied.
        
        **Why:**
        The reasoning behind using this technique in this context. If the response is a clarifying question, explain why the agent needs the extra information and how it will be used to provide a better response. **If `irrelevant_query` is true, the justification should explain that the query was identified as outside the agent's designated area of expertise, and that the response followed instructions to politely decline.**

        [**technique_name**]
        **Technique:**
        ...
        
        **Keyphrase Used:**
        ...
        
        **Why:**
        ...

        ...(repeat for each technique)
        ```
        
        The justification must only reference techniques from the knowledge base.

    3.  **"follow_up_actions"** (Array of Strings): Suggested next steps to further assist the user. These should be relevant to the user’s needs and leverage `product_info` and `domain_info` where applicable.
        *    **If `clarification_needed` is true, the only follow-up action should be to wait for the user to respond to the question.**
        *   **If `irrelevant_query` is true, the follow-up actions should offer assistance with relevant topics and direct the user to appropriate resources. For example: ["Offer assistance with product information or sales inquiries.", "Direct user to relevant sections of the website or documentation."].**

    **Key Considerations:**

    *   **Precision and Relevance:** Every response must be direct, clear, and tailored to the user's needs.
    *   **Ethical Persuasion:** Influence must be applied transparently and honestly.
    *   **Adaptive Communication:** Responses should align with the user’s tone and engagement level.
    *   **Consistency:** The reasoning in "justification" must clearly connect to the techniques used in "response."
    *   **Valid JSON Output:** The response must be structured as a valid JSON object containing only "response", "justification", and "follow_up_actions".

    Now, generate the JSON response:
    """
    
    """Creates a RAG-enhanced response for the user input."""
    # Format prompt for LLM
    prompt = prompt_template.format(
        user_query=user_input.user_query,
        intent=user_input.intent,
        emotion=user_input.emotion,
        product_info=user_input.product_info or "Not provided",
        domain_info=user_input.domain_info or "Not provided",
        irrelevant_query=user_input.irrelevant_query,
        clarification_needed=user_input.clarification_needed,
        context=context
    )

    # Generate response using LLM
    response_message = llm.invoke(prompt)  
    response_text = response_message.content

    try:
        response_json = json.loads(remove_markdown_code_block(response_text))
    except json.JSONDecodeError:
        response_json = {
            "response": "I'm sorry, but I couldn't generate a structured response at this time.",
            "justification": "AI failed to produce valid JSON. Please try again.",
            "follow_up_actions": []
        }

    return response_json
