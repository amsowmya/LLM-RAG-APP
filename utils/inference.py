from utils.llm import LLM
# from utils.build_rag_pinecone import RAG
from utils.build_rag import RAG
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import RetrievalQA 

def predict_rag(qns:str, history=None) -> str:
    llm = LLM().get_azure_openai_llm()
    retriver = RAG().get_retriever()
    template = """
    Answer the question based only on the following context:
    {context}
    
    Question: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    retrieval_chain = (
        {"context": retriver, "question": RunnablePassthrough()}
        | prompt
        | llm 
        | StrOutputParser()
    )
    
    result = retrieval_chain.invoke(qns)
    return result