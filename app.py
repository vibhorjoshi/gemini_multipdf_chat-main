import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings 
from PyPDF2 import PdfReader 
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Ensure PyPDF2 is installed for PDF reading

load_dotenv()  # Load environment variables from .env file
google_api_key = os.getenv('GOOGLE_API_KEY')  # Use a variable name

if not google_api_key:
    raise ValueError("Google API key not found. Please set it in the .env file.")

# Read all PDF files and return text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
            else:
                print("Warning: No text found on a page.")
    if not text:
        print("Warning: No text found in any of the uploaded PDFs.")
    return text

# Split text into chunks
def get_text_chunks(text):
    if not text:
        print("Warning: Received empty text for chunking.")
        return []  # Return empty list if text is empty

    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    print(f"Number of chunks created: {len(chunks)}")  # Log the number of chunks
    return chunks  # List of strings
# Get embeddings for each chunk
 # Example of an alternative embedding class

# Get embeddings for each chunk
def get_vector_store(chunks):
    if not chunks:
        raise ValueError("Text chunks are empty!")

    print(f"Number of chunks: {len(chunks)}")

    # Ensure embeddings are generated properly
    embeddings = OpenAIEmbeddings()  # Change to a different embedding class if necessary
    test_embedding = embeddings.embed_query("Test query")
    print(f"Sample embedding: {test_embedding}")

    # Create FAISS index from the embeddings
    try:
        chunk_embeddings = embeddings.embed_documents(chunks)
        vector_store = FAISS.from_embeddings(chunk_embeddings)
        vector_store.save_local("faiss_index")
        print("FAISS index successfully created.")
    except Exception as e:
        raise RuntimeError(f"Failed to create FAISS index: {e}")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "Answer is not available in the context."
    
    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", client=genai, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Upload some PDFs and ask me a question."}]

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # type: ignore

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) 
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    print(response)
    return response

def main():
    st.set_page_config(page_title="Gemini PDF Chatbot", page_icon="🤖")

    # Sidebar for uploading PDF files
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                
                if not text_chunks:
                    st.error("No text chunks were created from the PDF files.")
                    return  # Exit if no chunks are available

                get_vector_store(text_chunks)
                st.success("Done")


    # Initialize chat messages
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Upload some PDFs and ask me a question."}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User input for questions
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Display chat messages and bot response
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = user_input(prompt)
                    placeholder = st.empty()
                    full_response = ''
                    for item in response['output_text']:
                        full_response += item
                        placeholder.markdown(full_response)
                    placeholder.markdown(full_response)

            if response is not None:
                message = {"role": "assistant", "content": full_response}
                st.session_state.messages.append(message)

if __name__ == "__main__":
    main()

