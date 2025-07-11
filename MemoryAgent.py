import json
import os
import uuid
from typing import List, Literal
from dotenv import load_dotenv
import tiktoken
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import get_buffer_string, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_chroma import Chroma
from langchain.tools.retriever import create_retriever_tool
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from pydantic import BaseModel
from typing import List

load_dotenv()



# --- Long-Term Memory Setup (Same as before) ---
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
persist_path_memory = "./memory_db"  # Separate path for clarity
os.makedirs(persist_path_memory, exist_ok=True)

recall_vector_store = Chroma(
    embedding_function=embedding_model,
    persist_directory=persist_path_memory,
    collection_name="memory"
)

# --- Agentic RAG Setup (Your Original Code) ---
db_path = r"Knowledge_Base"
embeddings = GoogleGenerativeAIEmbeddings( model="models/embedding-001")
retriever = Chroma(persist_directory=db_path, embedding_function=embeddings).as_retriever()

info_knowledge_base_tool = create_retriever_tool(
    retriever,
"Cognitive Behavioral Knowledge",
    "This tool contains Cognitive Behavioral Therapy (CBT) knowledge to analyze user input and provide appropriate therapeutic responses. It identifies cognitive distortions, suggests coping strategies, and guides users toward healthier thought patterns.",

)


def get_user_id(config: RunnableConfig) -> str:
    user_id = config["configurable"].get("user_id")
    if user_id is None:
        raise ValueError("User ID needs to be provided to save a memory.")
    return user_id

class KnowledgeTriple(TypedDict):
    subject: str
    predicate: str
    object: str


@tool
def save_recall_memory(memories: List[KnowledgeTriple], config: RunnableConfig) -> str:
    """Save all the memory to vectorstore for later semantic retrieval. Always call this tool"""
    user_id = get_user_id(config)
    for memory in memories:
        # Skip null/meaningless memory
        if (
            memory["subject"].strip().upper() == "NULL" and
            memory["predicate"].strip().upper() == "NULL" and
            memory["object"].strip().upper() == "NULL"
        ):
            continue

        serialized = " ".join(memory.values())
        document = Document(
            serialized,
            id=str(uuid.uuid4()),
            metadata={
                "user_id": user_id,
                **memory,
            },
        )
        recall_vector_store.add_documents([document])
    return "Memories saved successfully."


@tool
def search_recall_memories(query: str, config: RunnableConfig) -> List[str]:
    """Search for relevant memories."""
    user_id = get_user_id(config)
    documents = recall_vector_store.similarity_search(
        query, k=5, filter={"user_id": user_id}
    )
    return [document.page_content for document in documents]



tools = [info_knowledge_base_tool, save_recall_memory] # Include the memory saving tool here


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    emotion  :  Annotated[Sequence[BaseMessage], add_messages]
    recall_memories: List[str] # Add recall memories to the state

# Define the data model
class grade(BaseModel):
    """A binary score for relevance checks"""
    binary_score: str = Field(
        description="Response 'yes' if the document is relevant to the question or 'no' if it is not."
    )


def grade_documents(state) -> Literal["generate", "rewrite"]:
    model = ChatGroq(temperature=0, model="llama3-70b-8192", streaming=True)
    llm_with_tool = model.with_structured_output(grade)
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )
    chain = prompt | llm_with_tool
    messages = state["messages"]
    last_message = messages[-1]
    question = messages[0].content
    retrieved_docs = last_message.content
    scored_result = chain.invoke({"question": question, "context": retrieved_docs})
    score = scored_result.binary_score
    print("==",score,"++")
    if score == "yes" or "no":
        print("==== [DECISION: DOCS RELEVANT] ====")
        return "generate"
    else:
        print("==== [DECISION: DOCS NOT RELEVANT] ====")
        print(score)
        return "rewrite"
    

def agent(state):
    print("==[AGENT]==")
    messages = state["messages"]
    emotion = state["emotion"]
    user_emotion = emotion[-1].content
    recall_memories_str = "\n".join(state.get("recall_memories", []))

    model = ChatGroq(temperature=0, streaming=True, model="llama3-70b-8192")
    prompt = f"""
    You are **SERENE**, an AI voice assistant specializing in **Cognitive Behavioral Therapy (CBT)**.
    You also have access to long-term memory. Relevant memories from past conversations are:
    ```
    {recall_memories_str}
    ```
   
    YOU ANSWER FOR SIMPLE QUESTION THAT DOES NOT REQUIRE ANY KNOWLEDGE SUCH AS `GREETING`(eg : hi,bye,hello,welcome,thanks) , GENERAL QUESTION (eg : who are you , how are you , your name ) AND -> END
    Use these memories to provide more context-aware and personalized responses.

    Your goal is to provide **empathetic, solution-focused, and concise** therapeutic responses through voice interaction.

    ### **Key Instructions:**
    - **Recognize and adapt to the user's emotional state: "{user_emotion}".
    - Recognize **cognitive, emotional, and behavioral** aspects relevant to the user's concerns.
    - Deliver responses that are **practical, clear, and aligned with CBT techniques**.
    - Keep answers **conversational, engaging, and optimized for spoken interaction**.

    ### **Response Guidelines:**
    - **Be warm, supportive, and emotionally attuned to the user.**
    - **Use CBT principles** like cognitive restructuring, coping strategies, and reframing techniques.
    - **Maintain a natural, fluid, and engaging tone suitable for voice interaction.**
    - **Ensure responses are between 30-50 words for clarity and effectiveness.**

     **SERENE's Response (empathetic, solution-focused, 20-50 words):**
     
---

**User's Current Emotional State:** {user_emotion}





    """
# YOU ANSWER FOR SIMPLE QUESTION THAT DOES NOT REQUIRE ANY KNOWLEDGE SUCH AS GREETING , GENERAL QUESTION AND -> END
    final_messages = [("user",prompt)] + list(messages)

    # Bind the retriever tool and the memory saving tool
    model_with_tools = model.bind_tools(tools)

    # Generate agent response
    print("++ From NON RAG ++")
    response = model_with_tools.invoke(final_messages)

    # Returns as a list since it is appended to the existing list
    return {"messages": [response]}


def rewrite(state):
    print("==== [QUERY REWRITE] ====")
    messages = state["messages"]

    question = next((msg.content for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)

    msg = [
    ("user" ,
        f"""
Only provide the improved version of the question as output. Do not include any reasoning or explanation. 
Rewrite the question by maintaining the context using different semantics while preserving the original intent.

Original Question:
-------
{question}
-------

Improved Question:"""
    )
] + list(messages)

    model = ChatGroq(temperature=0, model="llama3-70b-8192", streaming=True)
    response = model.invoke(msg)
    print(response.content , "<------ rewritten query")
    return {"messages": [response.content]}


def generate(state):
    print("==[GENERATEOR]==")
    messages = state["messages"]
    question = next((msg.content for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)
    emotion = state["emotion"]
    user_emotion = emotion[-1].content
    print(user_emotion,"<-----")
    print(question , "<------")
    docs = messages[-1].content
    messages = state["messages"]
    recall_memories_str = "\n".join(state.get("recall_memories", []))
    # Load the RAG prompt template
    prompt = PromptTemplate(
    template="""You are **SERENE**, an AI voice assistant specializing in **Cognitive Behavioral Therapy (CBT)** within a Retrieval-Augmented Generation (RAG) system.
Your goal is to provide **concise, empathetic, and solution-focused** responses tailored for voice interaction.
You  have access to long-term memory. Relevant memories from past conversations are:
    ```
    {recall_memories_str}
    ```



Use these memories to provide more context-aware and personalized responses.
### Key Instructions:
- **Recognize and adapt to the user's emotional state: "{user_emotion}".**
*** ang - angry , sad - sad , neu - neutral , hap - happy ***
- **Analyze the cognitive, emotional, and behavioral aspects of their concern.**
- **Deliver a response that is clear, supportive, and directly addresses their needs.**
- **Keep responses between 30-50 words for clarity and effectiveness.**

### Response Guidelines:
- **Be empathetic and conversational.**
- **Use CBT techniques like cognitive restructuring, coping strategies, and reframing.**
- **Ensure the response is engaging and natural for voice interaction.**

---

**User's Current Emotional State:** {user_emotion}

<current question>
**User’s Concern:**
{question}
<\current question>

**Context for the Response:**
{context}

---

 **SERENE’s Answer (concise, empathetic, and solution-focused, 30-50 words):**""",
    input_variables=["context", "question", "voice_emotion" , "recall_memories_str" ]
)

    llm = ChatGroq(model_name="llama3-70b-8192", temperature=0, streaming=True)
    rag_chain = prompt | llm | StrOutputParser()
    print("++ From RAG ++")
    response = rag_chain.invoke({"context": docs, "question": question , "user_emotion" : user_emotion , "recall_memories_str" :recall_memories_str })
    return {"messages": [response]}


# Updated memory prompt with stricter formatting
memory_prompt = PromptTemplate(
    template="""
You are a knowledge extraction system for long-term memory in LangGraph. Your task is to extract only **meaningful personal facts** about the user or people they mention, using subject-predicate-object triples in **strictly valid JSON array format**.

You are ONLY allowed to extract facts related to the following categories, which are intended for long-term memory storage:

1. Preferences (Likes/Dislikes)  
2. Goals and Aspirations  
3. Hobbies and Interests  
4. Skills and Strengths  
5. Weaknesses or Limitations  
6. Beliefs and Values  
7. Location or Environment Preferences  
8. Tasks and Reminders (Long-Term Tasks)  
9. Relationships and Roles  
10. Fears or Avoidances

---

### CRITICAL JSON FORMAT REQUIREMENTS:
**EVERY JSON object MUST have EXACTLY these THREE keys (no more, no less):**
- "subject" (exactly this spelling)
- "predicate" (exactly this spelling) 
- "object" (exactly this spelling)

**DO NOT:**
- Add extra characters to key names (NO "subjectsubject", "predicatepredicate", "objectobject")
- Use different key names
- Skip any of the three required keys
- Add additional keys

**CORRECT FORMAT:**
{{"subject": "value", "predicate": "value", "object": "value"}}

**INCORRECT FORMATS:**
{{"subjectsubject": "value", "predicate": "value", "object": "value"}} ❌
{{"subject": "value", "predicatepredicate": "value", "object": "value"}} ❌
{{"subject": "value", "predicate": "value", "objectobject": "value"}} ❌
{{"subject": "value", "predicate": "value"}} ❌ (missing object)

---

### GUIDELINES:
- Always start subject names with a capital letter.
- If the user's name is known (from memory), use it instead of "User".
- Do not infer or hallucinate facts.
- If the message is a greeting or not meaningful, return:  
  [{{"subject": "NULL", "predicate": "NULL", "object": "NULL"}}]
- Only output a valid JSON array and nothing else.
- If a fact does not fall under the above 10 categories, ignore it.
- If the user is mentioned (e.g., "Hi, I am Albert"), extract:  
  {{"subject": "Albert", "predicate": "is the", "object": "user"}}
- Use that name for all future facts instead of "User".

---

### IDENTIFY USER NAME:
Look at the `existing_memory` below. If a triple like  
{{"subject": "Sathya", "predicate": "is the", "object": "user"}}  
is found, then use "Sathya" as the subject **instead of "User"** in new facts.

---

### DUPLICATE PREVENTION:
- Existing memory is provided as a list of sentences (e.g., "Sathya likes energetic tamil songs").
- For each extracted triple, convert it to a sentence:  
  "<subject> <predicate> <object>"
- Do not output the triple if that sentence (case-insensitive) already exists or is semantically equivalent.

---

### EXAMPLES:

Input: "Hey, I'm Sathya. Actually I feel very low so I am going to drink coffee which is my favourite with my best friend Harish but he prefers tea. So sad."
Output:
[
  {{"subject": "Sathya", "predicate": "is the", "object": "user"}},
  {{"subject": "Sathya", "predicate": "likes", "object": "coffee"}},
  {{"subject": "Sathya", "predicate": "has best friend", "object": "Harish"}},
  {{"subject": "Harish", "predicate": "prefers", "object": "tea"}}
]

Input: "I have a physics exam coming soon."
Output:
[
  {{"subject": "Sathya", "predicate": "has", "object": "physics exam"}}
]

Input: "Hi I am Albert"
Output:
[
  {{"subject": "Albert", "predicate": "is the", "object": "user"}}
]

Input: "Hello!"
Output:
[
  {{"subject": "NULL", "predicate": "NULL", "object": "NULL"}}
]

---

### Existing memory:
{existing_memory}

---

**REMEMBER: Use EXACTLY these key names: "subject", "predicate", "object". No duplicates, no extra characters.**

Now extract triples from this:
"{text}"

**VALID JSON ARRAY OUTPUT:**""",
    input_variables=["text", "existing_memory"],
)

# Add validation function to clean malformed JSON
def clean_memory_output(raw_memories):
    """Clean and validate memory output from LLM"""
    cleaned_memories = []
    
    if isinstance(raw_memories, list):
        for memory in raw_memories:
            if isinstance(memory, dict):
                # Clean malformed keys
                cleaned_memory = {}
                
                # Extract subject
                subject_key = next((k for k in memory.keys() if 'subject' in k.lower()), None)
                if subject_key:
                    cleaned_memory['subject'] = memory[subject_key]
                
                # Extract predicate  
                predicate_key = next((k for k in memory.keys() if 'predicate' in k.lower()), None)
                if predicate_key:
                    cleaned_memory['predicate'] = memory[predicate_key]
                
                # Extract object
                object_key = next((k for k in memory.keys() if 'object' in k.lower()), None)
                if object_key:
                    cleaned_memory['object'] = memory[object_key]
                
                # Only add if all three keys are present
                if len(cleaned_memory) == 3:
                    cleaned_memories.append(cleaned_memory)
                else:
                    print(f"Skipping malformed memory: {memory}")
    
    return cleaned_memories

# Updated memory creation chain with error handling
parser = JsonOutputParser()
memory_model = ChatGroq(temperature=0, model="llama3-70b-8192", streaming=True)
def create_memory_chain_with_validation():
    def memory_chain_with_cleanup(inputs):
        try:
            # Get raw output from LLM
            raw_output = (memory_prompt | memory_model | parser).invoke(inputs)
            # Clean and validate
            cleaned_output = clean_memory_output(raw_output)
            return cleaned_output
        except Exception as e:
            print(f"Memory extraction error: {e}")
            return [{"subject": "NULL", "predicate": "NULL", "object": "NULL"}]
    
    return memory_chain_with_cleanup

# Replace your existing memory_creator_chain with this:
memory_creator_chain = create_memory_chain_with_validation()

# Updated load_memories function with better error handling
def load_memories(state: AgentState, config: RunnableConfig) -> AgentState:
    """Loads relevant memories based on the latest user message."""
    messages = state["messages"]
    last_user_message = next((msg.content for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)
    if last_user_message:
        try:
            recall_memories = search_recall_memories.invoke(last_user_message, config)
            new_memories = memory_creator_chain({"text": last_user_message, "existing_memory": recall_memories})
            print(f"Cleaned memories: {new_memories}")
            
            # Only save if we have valid memories
            if new_memories and not (len(new_memories) == 1 and 
                                  new_memories[0].get('subject') == 'NULL'):
                save_recall_memory.invoke({'memories': new_memories}, config)
            
            return {"recall_memories": recall_memories}
        except Exception as e:
            print(f"Error in load_memories: {e}")
            return {"recall_memories": []}
    return {"recall_memories": []}





workflow = StateGraph(AgentState)

# Add the load_memories node at the beginning
workflow.add_node("load_memories", load_memories)

# Define the rest of the nodes as before
workflow.add_node("agent", agent)
retrieve = ToolNode(tools=tools)
workflow.add_node("retrieve", retrieve)
workflow.add_node("rewrite", rewrite)
workflow.add_node(
    "generate",
    generate,
)

# Connect edges, starting with loading memories
workflow.add_edge(START, "load_memories")
workflow.add_edge("load_memories", "agent") # Agent now receives recall_memories in its state

workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "retrieve",
        END: END,
    },
)

workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")


graph = workflow.compile(checkpointer=MemorySaver())

# config = RunnableConfig(recursion_limit=10, configurable={"user_id": "1", "thread_id": "1"}) #{"configurable": {"user_id": "1", "thread_id": "1"}}

# while True :
    
#     query = input("Enter the query : ")
#     if 'stop' in query :
#         break
#     emo = input("Enter the emo : ")
#     response = graph.invoke({"messages": [("user",query)] , "emotion": [("user", emo)] }, config=config)
#     print("Bot:",response["messages"][-1].content)