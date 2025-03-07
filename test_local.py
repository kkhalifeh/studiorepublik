# test_local_fixed.py

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import json
import time
import os

load_dotenv()

# Initialize embeddings and vector DB
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Chroma(persist_directory="./studio_db", embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# Define function that will be used by model
tools = [
    {
        "type": "function",
        "function": {
            "name": "split_response",
            "description": "Split a response into multiple messages for a more natural conversation flow",
            "parameters": {
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Array of message strings to be sent as separate messages"
                    }
                },
                "required": ["messages"]
            }
        }
    }
]

# Initialize LLM with function calling
llm = ChatOpenAI(
    model_name="gpt-4o",
    temperature=0.7,
)

# Define the function calling in model_kwargs instead
llm_with_tools = llm.bind(
    model_kwargs={"tools": tools}
)

# Correctly define prompt using PromptTemplate
prompt_template = PromptTemplate.from_template("""
You are Zayn, a friendly and professional AI sales qualifier at StudioRepublik Dubai. Your primary goal is to qualify potential clients, encourage scheduling a facility tour, and collect useful profiling information to help the sales team.

Your conversational priorities are:
1. Politely and naturally encourage clients to schedule a facility visit.
2. If the client is hesitant or not immediately ready to schedule a visit, shift the conversation to gently gather more profiling information, including:
   - Fitness goals and interests
   - Preferred types of workouts or classes
   - Current fitness routine
   - Place of residence or neighborhood (to confirm proximity to StudioRepublik)

Guidelines:
- Maintain a warm, conversational, empathetic tone.
- NEVER be pushy. If they hesitate to schedule, smoothly transition into engaging conversation to collect profiling data.
- Respond ONLY from the provided context. If uncertain, politely say: "The sales team can provide more details during your visit or through direct follow-up."
- ALWAYS use the split_response function to break your response into 2-3 separate messages for a more natural conversation flow. Each message should be a separate thought or point.

Context:
{context}

Conversation:
Client Question: {question}
Zayn's Response:
""")

# Define the RAG chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_template
    | llm_with_tools
)

# Helper function to parse function arguments
def parse_function_args(tool_call):
    function_args = tool_call.get("arguments", "{}")
    if isinstance(function_args, str):
        try:
            return json.loads(function_args)
        except json.JSONDecodeError:
            print(f"Error parsing function arguments: {function_args}")
            return {}
    return function_args

# Function to handle the split_response tool
def handle_split_response(response):
    print("\nDebug - Full response structure:")
    print(f"Response type: {type(response)}")
    print(f"Response has 'tool_calls' attribute: {hasattr(response, 'tool_calls')}")
    
    # First check if there are tool calls
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"Found {len(response.tool_calls)} tool calls")
        
        for i, tool_call in enumerate(response.tool_calls):
            print(f"Tool call {i+1}: {tool_call.get('name', 'unknown')}")
            
            if tool_call.get("name") == "split_response":
                function_args = parse_function_args(tool_call)
                messages = function_args.get("messages", [])
                
                print(f"Split response messages: {len(messages)}")
                
                if messages:
                    return messages
    else:
        print("No tool calls found in response")
    
    # If no split_response tool was called or it failed, use the content
    content = response.content if hasattr(response, 'content') else str(response)
    print(f"Falling back to content: {content[:50]}...")
    
    # If we have content, return it as a single message
    if content:
        return [content]
    
    # Last resort fallback
    return ["I'm sorry, I'm having trouble with my response system. Let me try again. How can I help you with StudioRepublik today?"]

# Manual message splitting as a fallback
def split_message(content, max_length=1000):
    """Split a long message into multiple smaller messages"""
    # If message is short enough, return as is
    if len(content) <= max_length:
        return [content]
    
    # Try to split by paragraphs
    paragraphs = content.split('\n\n')
    
    if len(paragraphs) > 1:
        # Group paragraphs to stay below max_length
        messages = []
        current_message = ""
        
        for paragraph in paragraphs:
            if len(current_message) + len(paragraph) + 2 <= max_length:
                if current_message:
                    current_message += "\n\n" + paragraph
                else:
                    current_message = paragraph
            else:
                messages.append(current_message)
                current_message = paragraph
        
        if current_message:
            messages.append(current_message)
        
        return messages
    
    # If no paragraphs, split by sentences
    import re
    sentences = re.split(r'(?<=[.!?])\s+', content)
    
    messages = []
    current_message = ""
    
    for sentence in sentences:
        if len(current_message) + len(sentence) + 1 <= max_length:
            if current_message:
                current_message += " " + sentence
            else:
                current_message = sentence
        else:
            messages.append(current_message)
            current_message = sentence
    
    if current_message:
        messages.append(current_message)
    
    return messages

# Local testing function
def test_local_chat():
    print("Hey there! I'm Zayn from StudioRepublik. How can I help you today? (Type 'quit' to exit)\n")
    
    # Chat loop
    while True:
        question = input("You: ")
        if question.lower() in ["quit", "exit"]:
            print("Zayn: Thanks for chatting! Have a great day 😊")
            break
        
        print("\nProcessing your question...")
        
        try:
            # Process with RAG chain
            response = rag_chain.invoke(question)
            
            print(f"Response received. Processing messages...")
            
            # Get messages using the split_response tool or fallback method
            messages = handle_split_response(response)
            
            # If for some reason we have no messages or empty messages, use fallback
            if not messages or all(not msg for msg in messages):
                print("No valid messages found, using manual splitting...")
                text_content = response.content if hasattr(response, 'content') else str(response)
                messages = split_message(text_content)
            
            # Print messages with a delay between them
            print("\nZayn is typing...\n")
            for i, message in enumerate(messages):
                time.sleep(1)  # Simulate typing delay
                print(f"Zayn: {message}")
                if i < len(messages) - 1:
                    time.sleep(0.5)  # Delay between messages
                    print("\nZayn is typing...\n")
            
            print()  # Extra line for readability
            
        except Exception as e:
            print(f"\nError in processing: {str(e)}")
            print("Zayn: I apologize, but I'm experiencing a technical issue. Let me try again!")

if __name__ == "__main__":
    test_local_chat()