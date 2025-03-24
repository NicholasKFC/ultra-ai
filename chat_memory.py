from langchain.schema import HumanMessage, AIMessage

class ChatMemoryManager:
    def __init__(self, max_messages=10):
        self.history = []
        self.max_messages = max_messages

    def add(self, role, content):
        self.history.append({"role": role, "content": content})
        if self.max_messages:
            self.history = self.history[-self.max_messages:]

    def get_context(self):
        return self.history
    
    def get_formatted_history(self):
        return "\n".join(f"{msg['role']}: {msg['content']}" for msg in self.history)
    
    def to_langchain_messages(self):
        messages = []
        for msg in self.history:
            if msg["role"] == "human":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "ai":
                messages.append(AIMessage(content=msg["content"]))
        return messages