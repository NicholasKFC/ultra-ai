import os

from dotenv import load_dotenv
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
script_dir = Path(__file__).resolve().parent
env_path = script_dir.parent / '.env'
load_dotenv(dotenv_path=env_path)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
WEAVIATE_CLIENT_ID = os.environ.get("WEAVIATE_CLIENT_ID")
WEAVIATE_API_KEY = os.environ.get("WEAVIATE_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

def product_ask_demo_call(data):
    from main import client
    
    # Define LLM and retriever
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")
    vectorstore = WeaviateVectorStore(client, "Product_KB", "title", embeddings)
    retriever = vectorstore.as_retriever()
    
    query = data['query']

    template = """You are a helpful product recommendation agent. Your primary goal is to provide users with relevant product recommendations based on their query, including crucial product details with an emphasis on the URL and image. If no products match, return null.

    Instructions:

    1.  Understand the User Query: Analyze the user's query ({question}) to determine the specific product or type of product they are looking for.
    2.  Utilize Provided Context: Use the provided context ({context}) to identify relevant product information.  The context will contain information about various products, including titles, descriptions, normal prices, sale prices (if any), URLs, and image URLs.
    3.  Prioritize URLs and Images: The URL and image URL of the product are the MOST IMPORTANT pieces of information. Ensure they are always included in your response.
    4.  Provide Concise and Relevant Recommendations:  Recommend specific products that directly address the user's query.  Include the following information for EACH recommended product:
        *   **Product Name:** The full and accurate name of the product.
        *   **Sale Price (if any):** The current sale price. If there is no sale, indicate "N/A".
        *   **Normal Price:** The regular price of the product.
        *   **URL:** The direct URL link to the product page.
        *   **Image URL:** The URL of the product image.
        *   **Additional Details (if any):** Include any other pertinent information available in the context that might be useful to the user.
    5.  Format Your Response Clearly: Present your recommendations in a clear, easy-to-read format.  You can use bullet points, numbered lists, or a similar structure to organize the information.

    6.  Null for No Matches: If the context does not contain ANY information about products relevant to the query, OR if the available product information is insufficient to provide the required details (especially URL and image URL), you MUST output null instead.

    Example Response Format (Modify as needed for Context):

    Here are some t-shirt recommendations based on your query:

    *   **Product Name:**  Comfort Colors 1717 Adult Heavyweight RS T-Shirt
        *   **Sale Price:** N/A
        *   **Normal Price:** $15.00
        *   **URL:**  https://www.example.com/comfort-colors-1717
        *   **Image URL:** https://www.example.com/images/comfort-colors-1717.jpg
        *   **Additional Details:** Available in a wide range of colors.

    *   **Product Name:**  Nike Dri-FIT Training T-Shirt
        *   **Sale Price:** $20.00
        *   **Normal Price:** $25.00
        *   **URL:**  https://www.example.com/nike-dri-fit
        *   **Image URL:** https://www.example.com/images/nike-dri-fit.jpg
        *   **Additional Details:** Made with breathable, moisture-wicking fabric.

    If no suitable products are found return null.
    """

    prompt = ChatPromptTemplate.from_template(template)

    """Creates a RAG-enhanced response for the user input."""
    rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    )
    response_message = rag_chain.invoke(query)
    response_json = {'output': response_message}
    return response_json