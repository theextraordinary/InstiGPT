from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
import torch
import pickle
def train_model():
# Load model and tokenizer
    model_id = "google/flan-t5-xxl"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=100
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)

    from langchain.embeddings import HuggingFaceEmbeddings
    model_name = "sentence-transformers/all-mpnet-base-v2"
    hf = HuggingFaceEmbeddings(model_name=model_name)
    from langchain.document_loaders import PyPDFLoader
    loader = PyPDFLoader("data/ugrulebook.pdf")
    pages=loader.load()
    from langchain.text_splitter import CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    docs = text_splitter.split_documents(pages)
    from langchain.vectorstores import FAISS
    vectorStore = FAISS.from_documents(docs, hf)
    from langchain.chains import RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(
        local_llm,
        retriever=vectorStore.as_retriever()
    )
    return qa_chain

def question_answer(question,model):
    return model({'query':question})


model=train_model()
print('Ask a question from ugrulebook')
ques=input()
while(ques!=''):
    result=model({'query':ques})
    print(result['result'])
    print('Ask again or press enter to exit')
    ques=input()
