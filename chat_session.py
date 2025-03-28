import boto3
import datetime
import json
import os
import uuid

from dotenv import load_dotenv
from pathlib import Path
from botocore.exceptions import ClientError
from langchain.schema import HumanMessage, AIMessage

# Load environment variables
script_dir = Path(__file__).resolve().parent
env_path = script_dir.parent / '.env'
load_dotenv(dotenv_path=env_path)

aws_access_key = os.environ.get("aws_access_key")
aws_secret_key = os.environ.get("aws_secret_key")

# Initialize the S3 client
s3_client = boto3.client('s3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key)

# Define the S3 bucket name
bucket_name = 'ultraai-chat-sessions'

# Function to get or create a chat session
def get_session(session_id=None):
    # If no session_id is provided, create a new session
    if session_id is None:
        session_id = str(uuid.uuid4())  # Generate a unique session ID
        session_file_key = f"chat_sessions/{session_id}.json"
        chat_history = []  # Initialize an empty chat history

        # Upload an empty chat history file to S3
        s3_client.put_object(
            Bucket=bucket_name,
            Key=session_file_key,
            Body=json.dumps(chat_history),
            ContentType='application/json'
        )
        print(f"New chat session created with Session ID: {session_id}")
        return {"session_id": session_id, "chat_history": chat_history}

    # If a session_id is provided, check if it exists
    session_file_key = f"chat_sessions/{session_id}.json"

    try:
        # Check if the session file exists in S3
        response = s3_client.get_object(Bucket=bucket_name, Key=session_file_key)
        # If the object exists, fetch the chat history
        chat_history = json.loads(response['Body'].read().decode('utf-8'))
        print(f"Chat session with Session ID: {session_id} found.")
        return {"session_id": session_id, "chat_history": chat_history}
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            # If the session file does not exist, create a new session
            print(f"Chat session with Session ID: {session_id} not found. Creating new session.")
            return get_session()  # Recursively create a new session
        else:
            raise e  # Reraise the exception if it's a different error

# Function to add a new message to the chat session
def add_message(session_id, role, content):
    # If a session_id is provided, check if it exists
    session_file_key = f"chat_sessions/{session_id}.json"
    
    # Get the current chat history
    response = s3_client.get_object(Bucket=bucket_name, Key=session_file_key)
    chat_history = json.loads(response['Body'].read().decode('utf-8'))

    # Add the new message to the chat history
    chat_history.append({
        'role': role,
        'content': content,
        'timestamp': datetime.datetime.now().isoformat()
    })

    # Upload the updated chat history back to S3
    s3_client.put_object(
        Bucket=bucket_name,
        Key=session_file_key,
        Body=json.dumps(chat_history),
        ContentType='application/json'
    )

    print(f"Message added to session {session_id}. Role: {role}, Content: {content}")
    
def to_langchain_messages(chat_history):
    messages = []
    for msg in chat_history:
        if msg["role"] == "human":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "ai":
            messages.append(AIMessage(content=msg["content"]))
    return messages