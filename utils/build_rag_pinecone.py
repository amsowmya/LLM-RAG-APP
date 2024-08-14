from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from dotenv import load_dotenv
import os 

load_dotenv()

class RAG:
    def __init__(self) -> None:
        self.pdf_folder_path = os.getenv("SOURCE_DATA")
        self.emb_model_path = os.getenv("EMBEDED_MODEL")
        # self.emb_model = self.get_hf_embedding_model(self.emb_model_path)
        self.emb_model = self.get_embedding_model()
        self.vector_store_path = os.getenv('VECTOR_STORE')
        self.pinecone_api_key = os.getenv('PINECONE_API')
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.index = 'llmrag'
        
    def load_docs(self, path:str) -> PyPDFDirectoryLoader:
        loader = PyPDFDirectoryLoader(path)
        docs = loader.load()
        return docs 
    
    # def get_hf_embedding_model(self, emb_model) -> HuggingFaceBgeEmbeddings:
    #     model_kwargs = {'device': 'cpu'}
    #     encode_kwargs = {'normalize_embeddings': True}
    #     embeddings_model = HuggingFaceBgeEmbeddings(
    #         model_name=emb_model,
    #         model_kwargs=model_kwargs,
    #         encode_kwargs=encode_kwargs
    #     )
    #     return embeddings_model
    
    def get_embedding_model(self) -> AzureOpenAIEmbeddings:
        embeddings = AzureOpenAIEmbeddings(
            deployment=os.getenv('AZURE_EMBEDDING_DEPLOYMENT'),
            model='gpt4omodel',
            openai_api_type='azure',
            azure_endpoint=os.getenv('AZURE_EMBEDDING_ENDPOINT'),
            api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        )
        return embeddings
    
    def split_docs(self, docs) -> RecursiveCharacterTextSplitter:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        documents = text_splitter.split_documents(docs)
        return documents
    
    def populate_pinecone_db(self) -> None:
        try:
            self.doc = self.load_docs(self.pdf_folder_path)
            self.documents = self.split_docs(self.doc)
            
            index = self.pc.Index(self.index)
            
            vector_store = PineconeVectorStore(index=index, embedding=self.emb_model)
            vector_store.add_documents(self.documents)
            
        except Exception as e:
            print(e)
        
    def load_pinecone_db(self):
        try:
            
            index = self.pc.Index(self.index)
            vector_store = PineconeVectorStore(index=index, embedding=self.emb_model)
            
            return vector_store
        except Exception as e:
            print(e)
    
    def get_retriever(self):
        return self.load_pinecone_db().as_retriever()
    
    
# rag = RAG()
# rag.populate_pinecone_db()