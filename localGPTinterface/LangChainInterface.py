from gpt4all import GPT4All
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2
from langchain.document_loaders import PyPDFLoader
from bs4 import BeautifulSoup
import chromadb
from sentence_transformers import SentenceTransformer, util
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page_number in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_number]
            text += page.extract_text()
    return text

def split_text_into_sentences(text):
    sentences = text.split('.')
    return [s.strip() for s in sentences if s.strip()]

def embed_sentences(sentences):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(sentences)
    return embeddings

pdf_path = '/Users/damensavvasavvi/Desktop/localGPTinterface/output.pdf'
text = extract_text_from_pdf(pdf_path)
sentences = split_text_into_sentences(text)
embeddingsResult = embed_sentences(sentences)

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="***", metadata={"hnsw:space": "cosine"})

def id_generator():
    idArray = []
    for i in range(len(embeddingsResult)):
        idArray.append(str(i))
    return idArray


collection.add(embeddings = embeddingsResult, ids = id_generator(), documents= sentences)

question1 = ""
issue = ""


results = collection.query(
    query_texts = [issue],
    n_results=3,
)

def Myconcat():
    x = ''
    for z in results['documents']:
     for i in z:
        x += i
    return x
context = Myconcat()
print(f"text: {Myconcat()}" ) 


template = """
"""
prompt = PromptTemplate(template=template, input_variables=["question"])

local_path = (
    "/Users/damensavvasavvi/Desktop/localGPTinterface/gpt4all/orca-mini-3b-gguf2-q4_0.gguf"
)
callbacks = [StreamingStdOutCallbackHandler()]
llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True, allow_download=False)

llm_chain = LLMChain(prompt=prompt, llm=llm)
question = f"")
llm_chain.run(question)
