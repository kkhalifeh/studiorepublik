from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Initialize embeddings and vector DB
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Chroma(persist_directory="./studio_db", embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# Initialize LLM
llm = ChatOpenAI(model_name="gpt-4o")

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
- Break your responses into multiple separate messages (2-3) for a more natural conversation flow on WhatsApp.

Context:
{context}

Conversation:
Client Question: {question}
Zayn's Response:
""")



# Define QA chain correctly
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=False,
)

print("Hey there! I'm Zayn from StudioRepublik. How can I help you today? (Type 'quit' to exit)\n")

# Chat loop
while True:
    question = input("You: ")
    if question.lower() in ["quit", "exit"]:
        print("Zayn: Thanks for chatting! Have a great day 😊")
        break

    response = qa_chain({"query": question})
    print(f"Zayn: {response['result']}\n")
