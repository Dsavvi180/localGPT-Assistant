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
collection = chroma_client.create_collection(name="Wiki_context_data", metadata={"hnsw:space": "cosine"})

def id_generator():
    idArray = []
    for i in range(len(embeddingsResult)):
        idArray.append(str(i))
    return idArray


collection.add(embeddings = embeddingsResult, ids = id_generator(), documents= sentences)

question1 = "What is the ObjectID of EventsAir"
issue = "I cannot create bookings with Events Air. "


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

# modelPath = (Path.home()/'desktop'/'localGPTinterface'/'gpt4all')  
# model = GPT4All(model_name='orca-mini-3b-gguf2-q4_0.gguf',
#                 model_path=modelPath,
#                 allow_download=False)
# output = model.generate(f"You are now a personalised assistant. Answer any questions from the Context Text provided."
#                         f" Provide just the term for the Object ID. QUESTION: "
#                         f"What is the ObjectId for EventsAir? Text Context: {Myconcat()}", max_tokens=100)
# print("Output:",output,'\n modelPath: ',  modelPath)   


template = """You are now a personalised IT assistant. Answer all questions based on the text provided. Your job is to provide an Object ID from the context provided based on the 
issue you have at hand. In general, the issues are IT related and we are organizing each issues into categories based on object IDs. You need to provide an Object ID that you think 
best fits the issue. The subject of an issue often gives the best indication of the relevant Object ID. Choose an object ID from the following list ONLY:
Only reply with an ObjectID.
Object IDs: Email & Diary, Zoom, Events Air, Managed Mobile Requests, Office 365, Power BI, SAPS Business suite,Jira, No assignable object.

HINT: The objectID can be found in the subject
Question: {question}
"""
prompt = PromptTemplate(template=template, input_variables=["question"])

local_path = (
    "/Users/damensavvasavvi/Desktop/localGPTinterface/gpt4all/orca-mini-3b-gguf2-q4_0.gguf"
)
callbacks = [StreamingStdOutCallbackHandler()]
llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True, allow_download=False)

llm_chain = LLMChain(prompt=prompt, llm=llm)
question = f"What is the ObjectID of this {issue} ? Use the data in the following text to aid your answer. {context}."
print(f"here is output of llm: \n")
llm_chain.run(question)