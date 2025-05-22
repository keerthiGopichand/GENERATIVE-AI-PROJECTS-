from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from ecommbot.ingest import ingestdata
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from ecommbot.model import initialize_chat_ollama

### Statefully manage chat history ###
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def generation(vstore):
    # Start Ollama server in a separate thread if it’s not already running
    initialize_chat_ollama()
    
    llm = ChatOllama(model="llama3.1:8b")
    
    retriever = vstore.as_retriever(search_kwargs={"k": 3})

    contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    PRODUCT_BOT_TEMPLATE = """
    Your ecommercebot bot is an expert in product recommendations and customer queries.
    It analyzes product titles and reviews to provide accurate and helpful responses.
    Ensure your answers are relevant to the product context and refrain from straying off-topic.
    Your responses should be concise and informative.

    CONTEXT:
    {context}

    CHAT HISTORY:
    {chat_history}
    """

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PRODUCT_BOT_TEMPLATE),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )


    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
                                rag_chain,
                                get_session_history,
                                input_messages_key="input",
                                history_messages_key="chat_history",
                                output_messages_key="answer",
                                )

    return conversational_rag_chain

if __name__=='__main__':
    vstore = ingestdata()
    chain  = generation(vstore)
    print(chain.invoke("can you tell me the best bluetooth buds?"))