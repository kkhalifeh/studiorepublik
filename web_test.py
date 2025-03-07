# web_test.py

from flask import Flask, render_template, request, jsonify, session
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
import time
import uuid

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Initialize embeddings and vector DB
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Chroma(persist_directory="./studio_db", embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# Dictionary to store conversation histories
conversations = {}

# Initialize LLM
llm = ChatOpenAI(
    model_name="gpt-4o",
    temperature=0.7,
)

# Define the system message
system_message = """You are Zayn, a friendly and professional AI sales qualifier at StudioRepublik Dubai. Your primary goal is to qualify potential clients, encourage scheduling a facility tour, and collect useful profiling information to help the sales team.

Your conversational priorities are:
1. SUGGEST A TOUR within the first 2-3 exchanges in the conversation.
2. If the client shows ANY resistance to scheduling (says "not now", "maybe later", etc.) or ignores your tour suggestion twice, **PAUSE suggesting tours and focus on gathering profiling information instead**:
   - Fitness goals and interests
   - Preferred types of workouts or classes
   - Current fitness routine
   - Place of residence or neighborhood (to confirm proximity to StudioRepublik)
3. **Reintroduce a tour suggestion if the client shows renewed interest—like asking about membership details, expressing intent to join, or planning class attendance. Keep it casual and natural.**
   
Guidelines:
- Be EXTREMELY conversational and casual - as if texting a friend.
- Keep messages VERY SHORT (1-2 sentences max per message).
- Use different emojis naturally but not too frequently.
- Always break your responses into 2-3 separate messages maximum.
- Be brief and to-the-point. Avoid long explanations.
- Sound like a real person chatting on WhatsApp, not a formal representative.
- IMPORTANT: Only use greetings like "Hey" or "Hello" at the very beginning of the conversation. For all follow-up messages, respond directly without any greetings.
- NEVER BE PUSHY. If they say "not interested" or ignore your tour suggestion twice, focus on building rapport through conversation instead.
- After a client says "not interested" in a tour, ask about their fitness routines or goals instead.
- Respond ONLY from the provided context. If uncertain, simply say: "Our sales team can fill you in when you visit."
- Do not format your response with paragraph breaks for me to split - I will automatically split your response by sentences.

Here's information about StudioRepublik that you can refer to:
"""

# Format docs function
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Function to manually split a response into multiple messages
def split_into_messages(text, max_messages=3):
    import re
    
    # Remove any extra spaces or newlines
    text = text.strip()
    
    # First try to split by double newlines (paragraphs)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    if len(paragraphs) >= 2 and len(paragraphs) <= max_messages:
        return paragraphs
    
    # Next try to split by single newlines
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    
    if len(lines) >= 2 and len(lines) <= max_messages:
        return lines
    
    # Finally, split by sentences and group them
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # If we have very few sentences, just return them
    if len(sentences) <= max_messages:
        return sentences
    
    # Group sentences into 2-3 messages
    messages = []
    message_count = min(max_messages, 3)  # Max 3 messages
    sentences_per_message = len(sentences) // message_count
    
    for i in range(message_count):
        start_idx = i * sentences_per_message
        end_idx = start_idx + sentences_per_message if i < message_count - 1 else len(sentences)
        message = " ".join(sentences[start_idx:end_idx])
        messages.append(message)
    
    return messages

@app.route('/')
def index():
    # Generate a unique session ID if not exists
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    # Initialize conversation history if needed
    session_id = session['session_id']
    if session_id not in conversations:
        conversations[session_id] = [
            SystemMessage(content=system_message + format_docs(retriever.get_relevant_documents("")))
        ]
    
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    
    # Get session ID
    session_id = session.get('session_id')
    if not session_id or session_id not in conversations:
        session['session_id'] = str(uuid.uuid4())
        session_id = session['session_id']
        conversations[session_id] = [
            SystemMessage(content=system_message + format_docs(retriever.get_relevant_documents("")))
        ]
    
    # Get conversation history
    conversation = conversations[session_id]
    
    # Add user message to history
    conversation.append(HumanMessage(content=user_message))
    
    try:
        # Get relevant documents
        docs = retriever.get_relevant_documents(user_message)
        context = format_docs(docs)
        
        # Inform the model about the context
        context_message = f"Here's relevant information for the current question: {context}"
        
        # Process with LLM
        conversation.append(SystemMessage(content=context_message))
        response = llm.invoke(conversation)
        conversation.append(AIMessage(content=response.content))
        
        # Remove the context message to keep the history clean
        conversation.pop(-2)
        
        # Split the content into multiple messages
        messages = split_into_messages(response.content)
        
        return jsonify({'messages': messages})
    
    except Exception as e:
        print(f"Error processing message: {str(e)}")
        return jsonify({'messages': ["I'm having trouble processing your request right now. Let me get that fixed!"]})

if __name__ == '__main__':
    # Make sure templates directory exists
    os.makedirs('templates', exist_ok=True)
    
    # Create HTML template
    with open('templates/index.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Zayn WhatsApp Simulator</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f0f0f0;
        }
        .chat-container {
            background-color: #dcf8c6;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 20px;
            height: 400px;
            overflow-y: auto;
        }
        .message {
            padding: 10px;
            margin: 5px 0;
            border-radius: 10px;
            max-width: 70%;
            word-wrap: break-word;
        }
        .user-message {
            background-color: #ffffff;
            margin-left: auto;
            margin-right: 10px;
        }
        .bot-message {
            background-color: #e1ffc7;
            margin-right: auto;
            margin-left: 10px;
        }
        .input-container {
            display: flex;
            margin-top: 10px;
        }
        input {
            flex-grow: 1;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 20px;
            font-size: 16px;
        }
        button {
            margin-left: 10px;
            padding: 10px 20px;
            background-color: #25d366;
            color: white;
            border: none;
            border-radius: 20px;
            cursor: pointer;
        }
        .typing-indicator {
            padding: 10px;
            margin: 5px 0;
            border-radius: 10px;
            background-color: #e1ffc7;
            max-width: 70px;
            margin-right: auto;
            margin-left: 10px;
            display: none;
        }
        .dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #888;
            margin-right: 3px;
            animation: wave 1.3s linear infinite;
        }
        .dot:nth-child(2) {
            animation-delay: -1.1s;
        }
        .dot:nth-child(3) {
            animation-delay: -0.9s;
        }
        @keyframes wave {
            0%, 60%, 100% {
                transform: initial;
            }
            30% {
                transform: translateY(-5px);
            }
        }
    </style>
</head>
<body>
    <h1>Zayn WhatsApp Simulator</h1>
    <div class="chat-container" id="chatContainer">
        <div class="message bot-message">
            Hey there! I'm Zayn from StudioRepublik. How can I help you today? 😊
        </div>
    </div>
    <div class="typing-indicator" id="typingIndicator">
        <span class="dot"></span>
        <span class="dot"></span>
        <span class="dot"></span>
    </div>
    <div class="input-container">
        <input type="text" id="userInput" placeholder="Type your message..." onkeydown="if(event.key==='Enter')sendMessage()">
        <button onclick="sendMessage()">Send</button>
    </div>

    <script>
        function appendMessage(text, isUser) {
            const chatContainer = document.getElementById('chatContainer');
            const messageElement = document.createElement('div');
            messageElement.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
            messageElement.textContent = text;
            chatContainer.appendChild(messageElement);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        function showTypingIndicator() {
            document.getElementById('typingIndicator').style.display = 'block';
            document.getElementById('chatContainer').scrollTop = document.getElementById('chatContainer').scrollHeight;
        }

        function hideTypingIndicator() {
            document.getElementById('typingIndicator').style.display = 'none';
        }

        async function sendMessage() {
            const userInput = document.getElementById('userInput');
            const userMessage = userInput.value.trim();
            
            if (!userMessage) return;
            
            // Add user message to chat
            appendMessage(userMessage, true);
            userInput.value = '';
            
            // Show typing indicator
            showTypingIndicator();
            
            try {
                // Send to server
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: userMessage }),
                });
                
                const data = await response.json();
                
                // Handle multiple messages with typing indicators between them
                hideTypingIndicator();
                
                if (data.messages && data.messages.length > 0) {
                    for (let i = 0; i < data.messages.length; i++) {
                        if (i > 0) {
                            // Show typing indicator between messages
                            showTypingIndicator();
                            await new Promise(resolve => setTimeout(resolve, 1000));
                            hideTypingIndicator();
                        }
                        
                        appendMessage(data.messages[i], false);
                        
                        // Small delay after message
                        await new Promise(resolve => setTimeout(resolve, 300));
                    }
                } else {
                    appendMessage("Sorry, I couldn't process your request.", false);
                }
            } catch (error) {
                hideTypingIndicator();
                appendMessage("I'm having trouble connecting right now. Please try again later.", false);
                console.error('Error:', error);
            }
        }
    </script>
</body>
</html>
        ''')
    
    app.run(debug=True, port=5000)