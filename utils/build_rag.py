from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from dotenv import load_dotenv
import os 

class RAG:
    def __init__(self) -> None:
        self.pdf_folder_path = os.getenv("SOURCE_DATA")
        self.emb_model_path = os.getenv("EMBEDED_MODEL")
        # self.emb_model = self.get_hf_embedding_model(self.emb_model_path)
        self.emb_model = self.get_embedding_model()
        self.vector_store_path = os.getenv('VECTOR_STORE')
        
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
    
    def populate_vector_db(self) -> None:
        try:
            self.doc = self.load_docs(self.pdf_folder_path)
            self.documents = self.split_docs(self.doc)
            
            db = Chroma.from_documents(self.documents,
                                    embedding = self.emb_model,
                                    persist_directory=self.vector_store_path)
            
            db.persist()
        except Exception as e:
            print(e)
        
    def load_vector_db(self) -> Chroma:
        try:
            db = Chroma(persist_directory=self.vector_store_path, 
                        embedding_function=self.emb_model
                        )
            return db
        except Exception as e:
            print(e)
    
    def get_retriever(self) -> Chroma:
        return self.load_vector_db().as_retriever()
    
    
# rag = RAG()
# rag.populate_vector_db()