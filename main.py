"""for Splitting text into chunks then embeddings"""
import os
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings

##loading embeddings to Pinecone
from langchain.vectorstores import Pinecone


##1.Loading and Embedding
loader = TextLoader('./sample.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap=4)
docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings()


##2.Loading the embeddings to our PineCone Client
load_dotenv()
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
INDEX_NAME = "chatbot_demo"

#3.checking if index exists
if not pc.list_indexes(INDEX_NAME):
    pc.create_index_for_model(
        name = INDEX_NAME,
        cloud = "aws",
        region = "us-east-1",
        embed = {
            "model" : "llama-text-embed-v2",
            "field_map" : {"text":"chunk_text"}
        }
    )
##Because your index is integrated with an embedding model, 
#you provide the textual statements and Pinecone converts them to dense vectors automatically.
dense_index = pc.Index(INDEX_NAME)
dense_index.upsert_records("chatbot-namespace",docs)
