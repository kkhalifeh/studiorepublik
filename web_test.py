# web_test.py

from flask import Flask, render_template, request, jsonify, session
import re
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os
import time
import uuid
from datetime import datetime
import asyncio  # New import for async handling
from threading import Lock  # New import for thread-safe message buffering

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Chroma(persist_directory="./studio_db",
                  embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 5})

conversations = {}
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

# Dynamic system message with current date
current_date = datetime.now().strftime("%B %d, %Y")
system_message = f"""You are Zayn, a friendly and professional AI sales qualifier at StudioRepublik Dubai, located at Exit 41 - Umm Al Sheif, Eiffel Building 1, Sheikh Zayed Road, 8 16th Street, Dubai (Google Maps: https://maps.app.goo.gl/6Tm26dSG17bo4vHS9). Your primary goal is to qualify potential clients, encourage scheduling a facility tour, and collect useful profiling information to help the sales team. Today is {current_date}.

Your conversational priorities are:
1. SUGGEST A TOUR within the first 2-3 exchanges in the conversation—when they agree, offer available tour slots and confirm their pick with a message like 'Great! You’re booked—see you soon! 😊'.
2. If the person shows ANY resistance to scheduling OR ignores your tour suggestion ONCE, PAUSE suggesting tours and switch to gathering profiling information for 4 exchanges. and switch to gathering profiling information:
   - Fitness goals and interests
   - Preferred types of workouts or classes
   - Current fitness routine
   - Place of residence or neighborhood (to confirm proximity to StudioRepublik)
3. Reintroduce a tour suggestion after those 4 exchanges (including at least 2 exchanges of profiling) OR if the client shows renewed interest—like asking about membership details, facility features, or class schedules—and offer available slots to lock it in..
4. PROACTIVELY GATHER PROFILING INFORMATION by asking at least one profiling question within the first 3 exchanges only if they haven’t committed to a tour yet—like ‘What’s your go-to workout these days?’ or ‘What classes catch your eye?

Guidelines:
- Be EXTREMELY conversational and casual - as if texting a friend.
- Keep messages VERY SHORT (1-2 sentences max per message).
- Use emojis naturally but sparingly 😊
- ALWAYS break your responses into EXACTLY 2-3 SEPARATE messages—do not combine multiple ideas (e.g., membership engagement, tour suggestion, profiling) into one message.
- Be brief and to-the-point. Avoid long explanations.
- Sound like a real person chatting on WhatsApp, not a formal representative.
- IMPORTANT: Only use greetings like "Hey" or "Hello" at the very beginning. For follow-ups, respond directly without greetings.
- NEVER BE PUSHY. If they ignore or resist a tour suggestion, focus on building rapport through profiling—do not suggest tours until the PAUSE counter is exhausted.conversation instead—ask about their fitness vibe or goals.
- ALWAYS CHECK THE PROVIDED CONTEXT FIRST—use details like location, services, or pricing (e.g., AED 400/month for adults, AED 1,250/term for juniors aged 6-16) if they’re there! If the context doesn’t have a clear answer (e.g., ClassPass, sauna, unlisted features), transfer with “Let me pass you to the team—they’ll sort it!” and keep the chat flowing.
- IF THE USER’S LOCATION IS FAR AWAY (e.g., outside Dubai like Abu Dhabi), DO NOT SUGGEST A TOUR OR ASK ABOUT THEIR FITNESS ROUTINE. Instead, say: "Gotcha! Since you’re in [location], it might be a bit far. Keep us in mind if you’re ever in Dubai—we’d love to welcome you! 😊 I’m here if you have any questions."
- IF ASKED TO SCHEDULE A JUNIOR ASSESSMENT, transfer with “Let me pass you to the team—they’ll handle your junior assessment!”
- IF ASKED TO BOOK ANYTHING OTHER THAN A VISIT/APPOINTMENT FOR ADULTS (e.g., classes, programs, activities), transfer with “Let me pass you to the team—they’ll book that for you!”
- IF ASKED ABOUT TRIALS, transfer with “Let me pass you to the team—they’ll sort out your trial details!”
- FOR ANYTHING ELSE NOT COVERED ABOVE, transfer with “Let me pass you to the team—they’ll sort it out!”
- NEVER INVENT DETAILS LIKE DISCOUNTS, FAMILY PACKAGES, OR UNLISTED FEATURES—pricing and perks are sensitive, so only use explicit prices (AED 400/month for Basic, AED 1,250/term for juniors) and pass anything unclear to the team.
- ALWAYS SHARE THE LOCATION (Exit 41 - Umm Al Sheif, Eiffel Building 1, Sheikh Zayed Road, 8 16th Street, Dubai) when asked—it’s critical!
- For junior term questions, use today’s date ({current_date}) to determine the current term by comparing it to the term dates in the context—stick to the exact term start and end dates! If the date falls between a term’s start and end, that’s the current term!
- Do not format your response with paragraph breaks—I’ll split it by sentences.
Here's information about StudioRepublik that you can refer to:
"""

# Message buffer and lock for bundling messages
message_buffers = {}
buffer_locks = {}


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def split_into_messages(text, max_messages=3):
    import re
    text = text.strip()
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) >= 2 and len(paragraphs) <= max_messages:
        return paragraphs
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) >= 2 and len(lines) <= max_messages:
        return lines
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) <= max_messages:
        return sentences
    messages = []
    message_count = min(max_messages, 3)
    sentences_per_message = len(sentences) // message_count
    for i in range(message_count):
        start_idx = i * sentences_per_message
        end_idx = start_idx + sentences_per_message if i < message_count - \
            1 else len(sentences)
        message = " ".join(sentences[start_idx:end_idx])
        messages.append(message)
    return messages


@app.route('/')
def index():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    session_id = session['session_id']
    if session_id not in conversations:
        conversations[session_id] = [
            SystemMessage(content=system_message +
                          format_docs(retriever.get_relevant_documents("")))
        ]
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
async def chat():
    user_message = request.json.get('message', '').strip()
    if not user_message:
        return jsonify({'messages': []})

    session_id = session.get('session_id')
    if not session_id or session_id not in conversations:
        session['session_id'] = str(uuid.uuid4())
        session_id = session['session_id']
        conversations[session_id] = [
            SystemMessage(content=system_message +
                          format_docs(retriever.get_relevant_documents("")))
        ]

    # Initialize buffer and lock if not present
    if session_id not in message_buffers:
        message_buffers[session_id] = []
        buffer_locks[session_id] = Lock()

    # Add message to buffer
    with buffer_locks[session_id]:
        message_buffers[session_id].append(user_message)
        last_message_time = time.time()

    # Wait 5 seconds, checking for new messages
    await asyncio.sleep(10)

    # Process all buffered messages
    with buffer_locks[session_id]:
        if time.time() - last_message_time >= 10:  # Ensure 5 seconds have passed since last message
            bundled_messages = "\n".join(message_buffers[session_id])
            message_buffers[session_id] = []  # Clear buffer after processing
        else:
            # Another message came in, wait again
            return jsonify({'messages': []})

    conversation = conversations[session_id]
    conversation.append(HumanMessage(content=bundled_messages))

    try:
        docs = retriever.get_relevant_documents(bundled_messages)
        context = format_docs(docs)
        context_message = f"Here's relevant information for the current question: {context}"
        conversation.append(SystemMessage(content=context_message))
        response = llm.invoke(conversation)
        conversation.append(AIMessage(content=response.content))
        conversation.pop(-2)

        messages = split_into_messages(response.content)
        return jsonify({'messages': messages})

    except Exception as e:
        print(f"Error processing message: {str(e)}")
        return jsonify({'messages': ["I'm having trouble processing your request right now. Let me get that fixed!"]})

# @app.route('/chat', methods=['POST'])
# def chat():
#     user_message = request.json.get('message', '')

#     # Get session ID
#     session_id = session.get('session_id')
#     if not session_id or session_id not in conversations:
#         session['session_id'] = str(uuid.uuid4())
#         session_id = session['session_id']
#         conversations[session_id] = [
#             SystemMessage(content=system_message + format_docs(retriever.get_relevant_documents("")))
#         ]

#     # Get conversation history
#     conversation = conversations[session_id]

#     # Add user message to history
#     conversation.append(HumanMessage(content=user_message))

#     try:
#         # Get relevant documents
#         docs = retriever.get_relevant_documents(user_message)
#         context = format_docs(docs)

#         # Instead of adding a new system message, create a temporary conversation copy with updated context
#         temp_conversation = conversation.copy()

#         # Update the first system message with new context
#         if temp_conversation and isinstance(temp_conversation[0], SystemMessage):
#             temp_conversation[0] = SystemMessage(content=system_message + context)

#         # Process with LLM using the temporary conversation
#         response = llm.invoke(temp_conversation)

#         # Add AI response to the original conversation history
#         conversation.append(AIMessage(content=response.content))

#         # Split the content into multiple messages
#         messages = split_into_messages(response.content)

#         return jsonify({'messages': messages})

#     except Exception as e:
#         print(f"Error processing message: {str(e)}")
#         return jsonify({'messages': ["I'm having trouble processing your request right now. Let me get that fixed!"]})


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
