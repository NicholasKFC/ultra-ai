import os

from dotenv import load_dotenv
from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore
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

def domain_call(data):
    from main import client
    
    # Define LLM and retriever
    #llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    llm = ChatOpenAI(model="gpt-4o-mini")
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")
    vectorstore = WeaviateVectorStore(client, "Domain_TM", "content", embeddings)
    retriever = vectorstore.as_retriever()
    
    query = data['query']

#     template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
# Question: {question}
# Context: {context}
# Answer:
#     """
    template = """
    You are a highly specialized information retrieval agent, focused solely on providing information directly from the provided context related to the company and its domain. Your only function is to extract and present relevant information; do NOT engage in conversation, ask clarifying questions, or provide any information outside of what is explicitly available in the context. If no information matches, return null.

    Instructions:

    1. Understand the User Inquiry: Analyze the user's question ({question}) to determine the specific information they are seeking about the company, its history, operations, values, products, or services.

    2. Utilize Provided Context: Use the provided context ({context}) to identify relevant information that directly answers the user's question. The context will contain internal documents, public statements, website content, and other relevant company data.

    3. Provide Direct and Concise Answers: Respond to the user's question with a direct and concise answer, drawing ONLY from the provided context. Your answer should:
        * Directly address the user's question.
        * Be factually accurate and consistent with the provided context.
        * Cite the source of the information whenever possible (e.g., "According to the 2023 Annual Report..."). This is important for transparency and verifiability. If no explicit source is provided in the context, omit the citation.

    4. Null for No Relevant Information: If the context contains absolutely NO information relevant to the user's question, or if the information is insufficient to answer the question directly, you MUST output null. This is crucial to prevent the agent from providing inaccurate or misleading information or attempting to "fill in the gaps."
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

@tool
def domain_tool(query: str) -> dict[str, str]:
    """This tool retrieves information about the company, its policies (e.g., return policy, shipping policy), general information about the industry, company values, and contact information. Use this tool when the user is asking about shipping, returns, company information, privacy, or other general policies and industry-related knowledge. The response from this tool is context about the company in text."""
    return domain_call({"query": query})