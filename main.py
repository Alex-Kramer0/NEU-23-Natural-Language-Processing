from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader

load_dotenv()
embeddings = OpenAIEmbeddings()


def create_custom_embeddings(path: str) -> FAISS:
    loader = TextLoader(path)
    documents= loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    db = FAISS.from_documents(docs, embeddings)
    return db

db=create_custom_embeddings('/Users/harisha/Desktop/data.txt')
#print(db)



def get_response_from_query(db, query, k=4):

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = OpenAI(model_name="text-davinci-003")

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a children story telling assistant that that can generate stories based of a word or sentence
        
        Generate stories based on the following user input: {question}
        By searching the following documents: {docs}
                
        Your answers should be creative and detailed.
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response

res=get_response_from_query(db,"Generate a story about firetrucks")

print(res)