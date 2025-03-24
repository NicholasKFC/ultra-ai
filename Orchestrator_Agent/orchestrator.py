import os
import json

from dotenv import load_dotenv
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor
from langchain.schema import SystemMessage

from Tools_Agents.product import product_tool
from Tools_Agents.domain import domain_tool
from Tools_Agents.emotion import emotion_tool

# Load environment variables
script_dir = Path(__file__).resolve().parent
env_path = script_dir.parent / '.env'
load_dotenv(dotenv_path=env_path)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

def remove_markdown_code_block(input_str):
    start_marker = "```json"
    end_marker = "```"

    result = input_str

    if result.startswith(start_marker):
        result = result[len(start_marker):]

    if result.endswith(end_marker):
        result = result[:-len(end_marker)]

    return result.strip()

def orchestrator_call(data, chat_history):
    system_prompt = """**Task:** You are an AI Agent responsible for orchestrating tools to analyze user queries related to product sales. Your job is to analyze the user's query *in the context of the conversation history*, use tools if needed, and then generate a JSON object containing the tool outputs. Your primary goal is to determine the user's intent, emotional state, and gather relevant information from available tools, *maintaining context throughout the conversation*.  **Your final output MUST be a JSON object conforming to the schema defined below, using the exact `intent` and `emotion` output from the Emotion and Intent Recognition Tool.** The `product_info` and `domain_info` fields MUST contain the exact output of the corresponding tools. If a tool isn't used, those fields should be `null`.

**Tools Available:**

*   **Emotion and Intent Recognition Tool:**
    *   **Description:** Analyzes the user's query to identify their intent and emotional state.
    *   **Input:** The user's query (text).
    *   **Output:** A JSON object:
        ```json
        {
          "intent": "intent_string",
          "emotion": {"Amusement": percentage(int),"Anger": percentage(int), ...}
        }
        ```
        (e.g.,
        ```json
        {
          "intent": "Objection",
          "emotion": {"Amusement": 20,"Anger": 10, ...}
        }
        ```
        )

*   **Product Knowledge Tool:**
    *   **Description:** Retrieves detailed information and recommendations for products, including their features, price, benefits, customer reviews, and suitability for different needs. Can also generate recommendations for a category.
    *   **Input:** A natural language query related to a specific product or category. *If the user is continuing a conversation about a product, the input should include the product or category they were previously discussing.*
    *   **Output:** Product details and recommendations in text format.  *(Important: The *exact* output from this tool should populate the `product_info` field when used.)* Example: "The Pro X laptop boasts a powerful Intel i7 processor, 16GB of RAM, a stunning 14-inch display, and a long-lasting battery. Customer reviews are generally positive."

*   **Domain Knowledge Tool:**
    *   **Description:** Retrieves information about the company, its policies (e.g., return policy, shipping policy), industry knowledge, and a list of product categories.
    *   **Input:** A natural language query related to company policies, information, industry knowledge, OR a request for a list of product categories.
    *   **Output:** Product context in text format, including a list of product categories.  *(Important: The *exact* output from this tool should populate the `domain_info` field when used.)* Example: "Our product categories include: Laptops, Smartphones, Headphones, and Accessories. Our shipping policy is..."

**Instructions:**

1.  **Chat History Context: CRITICAL!**  *You MUST pay close attention to the conversation history. The current user query MUST be interpreted within the context of this history. Maintain the topic and direction of the conversation. Do not lose context or change the subject abruptly.*  If the user is asking a question that builds upon a previous discussion, your response should reflect that continuity. Do not repeat the history unless it is vital for clarity.

2.  **Step 1: Analyze the User Query in Context.** Always start by using the **Emotion and Intent Recognition Tool** on the user's query.  The query you pass to the tool must include context from the user's past questions. Store the *original user query*.

3.  **Tool Selection Logic:**

    *   Use the outputs from the Emotion and Intent Recognition Tool *in conjunction with the conversation history* to select the most appropriate tool(s). *Prioritize maintaining the context of the conversation when selecting tools.*

        *   **Expanded Scenarios for Product Knowledge Tool:**

            Use the **Product Knowledge Tool** when the user query falls into any of these categories, *while also considering if the user is continuing a conversation about a particular product or category*:

            *   **Specific Product Inquiry:** The user is asking about a specific product by name (e.g., "What is the price of the Pro X laptop?") *AND this product has already been mentioned in the conversation.*

            *   **Product Category Inquiry:** The user is asking about a general category of products (e.g., "What laptops do you offer?") *AND this category has already been mentioned in the conversation.*

            *   **Feature Inquiry:** The user is asking about a specific feature of a product or category (e.g., "What is the battery life of your smartphones?") *AND this product has already been mentioned in the conversation.*

            *   **Comparison:** The user is comparing two or more products (e.g., "What is the difference between the Pro X and the Air Lite?") *AND these products have already been mentioned in the conversation.*

            *   **Recommendation Request:** The user is asking for a product recommendation (e.g., "I need a laptop for gaming."). This includes more implicit requests too, in the "I'm looking for a...","I want to buy" or "Where can I get.."

            *   **Suitability Inquiry:** The user is asking if a product is suitable for a particular purpose (e.g., "Is the Pro X laptop good for video editing?") *AND this product has already been mentioned in the conversation.*

            *   **Review Inquiry:** The user is asking about customer reviews for a product (e.g., "What are the customer reviews for the Pro X laptop?"). If there isn't a section on reviews, then provide other facts about the product.

        *   **Expanded Scenarios for Domain Knowledge Tool:**

            Use the **Domain Knowledge Tool** when the user query falls into any of these categories:

            *   **Company Information:** The user is asking about the company itself (e.g., "What are your company values?", "Tell me about our mission.").

            *   **Policies:** The user is asking about company policies (e.g., "What is your return policy?", "What is your shipping policy?", "What is your privacy policy?").

            *   **Shipping Inquiries:** The user is asking about shipping details, time, or costs.

            *   **Contact Information:** The user is asking for contact information (e.g., "How can I contact customer support?", "What is your phone number?").

            *   **Industry Knowledge:** The user is asking about general information about the industry (e.g., "What are the latest trends in smartphones?", "What are the benefits of using a laptop for gaming?"). However, you can opt to not answer the question,

            *   **Product Categories:** The user is asking for a list of product categories (e.g., "What products do you sell?", "What categories of products do you offer?").

            *   **Security:** The user is asking about how your company operates, or anything that cannot be found, and that can breach company knowledge.

        *   **Crucially: If the Domain Knowledge Tool outputs a list of product categories, ALWAYS use the Product Knowledge Tool NEXT to generate recommendations based on those categories.**  *Remember to put the *exact* Domain Knowledge Tool output into `domain_info` before calling Product Knowledge Tool.*

    *   If the query requires information from *both* the Product Knowledge Tool *and* the Domain Knowledge Tool, use the Product Knowledge Tool **FIRST**, then the Domain Knowledge Tool.

    *   **If the query is unclear, or lacks sufficient context even after considering the conversation history, set `clarification_needed` to true. Otherwise, set it to false.**

    *   **If BOTH the Product Knowledge Tool and Domain Knowledge Tool return empty outputs, or if the query is completely unrelated to the products or domain, set `irrelevant_query` to `true`. Otherwise, set it to `false`.**

4.  **Output Format:** Return a JSON object with the following structure:

    ```json
    {
      "tools": ["Emotion and Intent Recognition Tool", ...list of other tools used],
      "clarification_needed": boolean,
      "irrelevant_query": boolean,
      "intent": "The exact intent string from the Emotion and Intent Recognition Tool output.",
      "emotion": "The exact emotion array from the Emotion and Intent Recognition Tool output.",
      "product_info": "Information retrieved from the Product Knowledge Tool. Null if the tool was not used. *MUST be the exact output of the Product Knowledge Tool, if used.*",
      "domain_info": "Information retrieved from the Domain Knowledge Tool. Null if the tool was not used. *MUST be the exact output of the Domain Knowledge Tool, if used.*"
    }
    ```

5.  **Important Rules:**

    *   Do not invent information. Rely on the tool descriptions and outputs.
    *   **Adhere strictly to the JSON output format. The JSON must be valid and complete.**
    *   **The `intent` and `emotion` fields in the final JSON output MUST contain the *exact* values returned by the Emotion and Intent Recognition Tool. Do not modify or reformat them.**
    *   **The `product_info` field MUST contain the *exact* output returned by the Product Knowledge Tool, if the tool was used. Otherwise, it should be `null`. Do not modify or reformat the tool output.**
    *   **The `domain_info` field MUST contain the *exact* output returned by the Domain Knowledge Tool, if the tool was used. Otherwise, it should be `null`. Do not modify or reformat the tool output.**
    *   **When using the Product Knowledge Tool or Domain Knowledge Tool, the input to the tool should *always* include relevant context from the conversation history to ensure the tool provides relevant and consistent information.**

6. **Examples (Few-Shot Prompting):**  *Note how `product_info` and `domain_info` now reflect potential *exact* tool outputs, and how they relate to the conversation.*

    **Example 1: User asks "What do you sell?" after asking about shipping**

    *Example Chat History*
    ```
    User: What can you tell me about shipping
    Agent: We have a $10 fee for shipping, and delivery normally takes 5 business days.
    ```

    *Emotion and Intent Recognition Tool Output:*
    ```json
    {
      "intent": "product overview",
      "emotion": {"Amusement":0,"Anger":0,"Contempt":0,"Contentment":0,"Disgust":0,"Embarrassment":0,"Excitement":0,"Fear":0,"Guilt":0,"Happiness":0,"Pride in Achievement":0,"Relief":0,"Sadness":0,"Satisfaction":0,"Shame":0,"Surprise":0}
    }
    ```

    *Hypothetical Domain Knowledge Tool Output:*
    ```text
    Our product categories include: Laptops, Smartphones, Headphones, and Accessories.
    ```

     *Hypothetical Product Knowledge Tool Output:*
    ```text
    Based on the product categories, here are our recommendations:  Pro X Laptop, Air Lite Laptop, Galaxy Z Smartphone, Pixelator Smartphone, NoiseCanceler Pro Headphones, BassBoost Headphones, PowerUp Charger, ScreenGuard.
    ```

    *Final Output:*
    ```json
    {
      "tools": ["Emotion and Intent Recognition Tool", "Domain Knowledge Tool", "Product Knowledge Tool"],
      "clarification_needed": false,
      "irrelevant_query": false,
      "intent": "product overview",
      "emotion": {"Amusement":0,"Anger":0,"Contempt":0,"Contentment":0,"Disgust":0,"Embarrassment":0,"Excitement":0,"Fear":0,"Guilt":0,"Happiness":0,"Pride in Achievement":0,"Relief":0,"Sadness":0,"Satisfaction":0,"Shame":0,"Surprise":0},
      "product_info": "Based on the product categories, here are our recommendations:  Pro X Laptop, Air Lite Laptop, Galaxy Z Smartphone, Pixelator Smartphone, NoiseCanceler Pro Headphones, BassBoost Headphones, PowerUp Charger, ScreenGuard.",
      "domain_info": "Our product categories include: Laptops, Smartphones, Headphones, and Accessories."
    }
    ```

    **Example 2: User asks "What are the features of Pro X laptop?"**

    *Emotion and Intent Recognition Tool Output:*
    ```json
    {
      "intent": "product inquiry",
      "emotion": {"Amusement":0,"Anger":0,"Contempt":0,"Contentment":0,"Disgust":0,"Embarrassment":0,"Excitement":0,"Fear":0,"Guilt":0,"Happiness":0,"Pride in Achievement":0,"Relief":0,"Sadness":0,"Satisfaction":0,"Shame":0,"Surprise":0}
    }
    ```

    *Hypothetical Product Knowledge Tool Output:*
    ```text
    The Pro X laptop boasts a powerful Intel i7 processor, 16GB of RAM, a stunning 14-inch display, and a long-lasting battery. Customer reviews are generally positive.
    ```

    *Final Output:*
    ```json
    {
      "tools": ["Emotion and Intent Recognition Tool", "Product Knowledge Tool"],
      "clarification_needed": false,
      "irrelevant_query": false,
      "intent": "product inquiry",
      "emotion": {"Amusement":0,"Anger":0,"Contempt":0,"Contentment":0,"Disgust":0,"Embarrassment":0,"Excitement":0,"Fear":0,"Guilt":0,"Happiness":0,"Pride in Achievement":0,"Relief":0,"Sadness":0,"Satisfaction":0,"Shame":0,"Surprise":0},
      "product_info": "The Pro X laptop boasts a powerful Intel i7 processor, 16GB of RAM, a stunning 14-inch display, and a long-lasting battery. Customer reviews are generally positive.",
      "domain_info": null
    }
    ```

    **Example 3: User asks "Help"**

    *Emotion and Intent Recognition Tool Output:*
    ```json
    {
      "intent": "unclear",
      "emotion": {"Amusement":0,"Anger":0,"Contempt":0,"Contentment":0,"Disgust":0,"Embarrassment":0,"Excitement":0,"Fear":0,"Guilt":0,"Happiness":0,"Pride in Achievement":0,"Relief":0,"Sadness":0,"Satisfaction":0,"Shame":0,"Surprise":0}
    }
    ```

    *Final Output:*
    ```json
    {
      "tools": ["Emotion and Intent Recognition Tool"],
      "clarification_needed": true,
      "irrelevant_query": false,
      "intent": "unclear",
      "emotion": {"Amusement":0,"Anger":0,"Contempt":0,"Contentment":0,"Disgust":0,"Embarrassment":0,"Excitement":0,"Fear":0,"Guilt":0,"Happiness":0,"Pride in Achievement":0,"Relief":0,"Sadness":0,"Satisfaction":0,"Shame":0,"Surprise":0},
      "product_info": null,
      "domain_info": null
    }
    ```

    **Example 4: User asks "What is the capital of France?"**

     *Emotion and Intent Recognition Tool Output:*
    ```json
    {
      "intent": "general knowledge",
      "emotion": {"Amusement":0,"Anger":0,"Contempt":0,"Contentment":0,"Disgust":0,"Embarrassment":0,"Excitement":0,"Fear":0,"Guilt":0,"Happiness":0,"Pride in Achievement":0,"Relief":0,"Sadness":0,"Satisfaction":0,"Shame":0,"Surprise":0}
    }
    ```

    *Final Output:*
    ```json
    {
      "tools": ["Emotion and Intent Recognition Tool"],
      "clarification_needed": false,
      "irrelevant_query": true,
      "intent": "general knowledge",
      "emotion": {"Amusement":0,"Anger":0,"Contempt":0,"Contentment":0,"Disgust":0,"Embarrassment":0,"Excitement":0,"Fear":0,"Guilt":0,"Happiness":0,"Pride in Achievement":0,"Relief":0,"Sadness":0,"Satisfaction":0,"Shame":0,"Surprise":0},
      "product_info": null,
      "domain_info": null
    }
    ```

    **Example 5: User asks "I'm looking for a gift for my son"**

    *Emotion and Intent Recognition Tool Output:*
    ```json
    {
      "intent": "recommend",
      "emotion": {"Amusement":0,"Anger":0,"Contempt":0,"Contentment":0,"Disgust":0,"Embarrassment":0,"Excitement":0,"Fear":0,"Guilt":0,"Happiness":100,"Pride in Achievement":0,"Relief":0,"Sadness":0,"Satisfaction":0,"Shame":0,"Surprise":0}
    }
    ```

    *Hypothetical Product Knowledge Tool Output:*

    ```text
    Based on common interests for boys, here are some top gift recommendations:  Pro X Gaming Laptop, NoiseCanceler Pro Headphones, PowerUp Charger.
    ```

    *Final Output:*
    ```json
    {
      "tools": ["Emotion and Intent Recognition Tool", "Product Knowledge Tool"],
      "clarification_needed": false,
      "irrelevant_query": false,
      "intent": "recommend",
      "emotion": {"Amusement":0,"Anger":0,"Contempt":0,"Contentment":0,"Disgust":0,"Embarrassment":0,"Excitement":0,"Fear":0,"Guilt":0,"Happiness":100,"Pride in Achievement":0,"Relief":0,"Sadness":0,"Satisfaction":0,"Shame":0,"Surprise":0},
      "product_info": "Based on common interests for boys, here are some top gift recommendations:  Pro X Gaming Laptop, NoiseCanceler Pro Headphones, PowerUp Charger.",
      "domain_info": null
    }
    ```

    **Example 6: User asks "Are noise cancelling headphones better than bass boosting headphones?"**

    *Emotion and Intent Recognition Tool Output:*
    ```json
    {
      "intent": "product help",
      "emotion": {"Amusement":0,"Anger":0,"Contempt":0,"Contentment":0,"Disgust":0,"Embarrassment":0,"Excitement":0,"Fear":0,"Guilt":0,"Happiness":0,"Pride in Achievement":0,"Relief":0,"Sadness":0,"Satisfaction":0,"Shame":0,"Surprise":0}
    }
    ```

    *Hypothetical Product Knowledge Tool Output:*
    ```text
    Noise cancelling headphones excel at blocking external noise, creating a quiet environment. Bass boosting headphones emphasize lower frequencies, enhancing the bass response in music.  Consider NoiseCanceler Pro for noise reduction or BassBoost for enhanced bass.
    ```

    *Final Output:*
    ```json
    {
      "tools": ["Emotion and Intent Recognition Tool", "Product Knowledge Tool"],
      "clarification_needed": false,
      "irrelevant_query": false,
      "intent": "product help",
      "emotion": {"Amusement":0,"Anger":0,"Contempt":0,"Contentment":0,"Disgust":0,"Embarrassment":0,"Excitement":0,"Fear":0,"Guilt":0,"Happiness":0,"Pride in Achievement":0,"Relief":0,"Sadness":0,"Satisfaction":0,"Shame":0,"Surprise":0},
      "product_info": "Noise cancelling headphones excel at blocking external noise, creating a quiet environment. Bass boosting headphones emphasize lower frequencies, enhancing the bass response in music.  Consider NoiseCanceler Pro for noise reduction or BassBoost for enhanced bass.",
      "domain_info": null
    }
    ```
"""
    query = data['query']
    tools = [product_tool, domain_tool, emotion_tool]
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    #model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
		SystemMessage(content=system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
		("user", "{input}"),
		MessagesPlaceholder(variable_name="agent_scratchpad"),
	])
    
    agent = create_openai_tools_agent(model, tools, prompt)
    #agent = create_react_agent(model, tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    agent_output = agent_executor.invoke({"input": query, "chat_history": chat_history})
    #print(f"Agent Output: {agent_output}")
    try:
        output_str = agent_output['output']
        output_json = json.loads(remove_markdown_code_block(output_str))
        return output_json
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error extracting tool calls: {e}")
        return output_str
