from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

import getpass
import os


if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API key: ")

# -------------------------------------------------------PHASE 1-----------------------------------------------------------------

                                    # extracting external data to vectorDB / RETRIEVAL


file_paths = [
    "/Users/singhaman4545/Downloads/Grades for Aman Singh_ AWS Academy Cloud Foundations [160598].pdf",
]

# Step 1 : Load source
loader = UnstructuredLoader(file_paths)

# Load all documents
docs = loader.load()

# print(docs)

# Step 2 : Chunking
# function that decide how to spit the doc
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=100
    )
chunks = text_splitter.split_documents(docs)


# print(len(chunks))               # number of chunks
# print(chunks[0].page_content)    # the actual text
# print(chunks[0].metadata)        # source info

# STEP 3 : Embedding
print( type(chunks))
embedder = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
vectors = embedder.embed_documents(
     [chunk.page_content for chunk in chunks]
)

# STEP 3 : storing embedding to vectorDB

vectorDB = QdrantVectorStore.from_documents(
    documents=[],
    embedding=embedder,
    collection_name="demo",
    url="http://localhost:6333",
)

vectorDB.add_documents(documents = docs)

# -------------------------------------------------------PHASE 2-----------------------------------------------------------------

                                            # user intraction / AUGMENTATION

# user input
query = input("Enter Query : ")

# embedding of query and searching for relevent chunk are done by this fuction
relevent_content = vectorDB.similarity_search(
    query = "what abstract represent"
)


for res in relevent_content:
    print()
    print(f"* {res.page_content} [{res.metadata}]")



# -------------------------------------------------------PHASE 3-----------------------------------------------------------------

                                                 #Prompting phase and GENERATION

system_prompt = f"""
You are a helpfull ai assistant who helps to answer user querys related to the context provided to you ,
you should always analyse query , plan , search in context , result , verify result , final output .

context : {relevent_content}


"""