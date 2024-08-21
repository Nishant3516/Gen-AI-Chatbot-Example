import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

# Set your OpenAI API key
OPENAI_API_KEY = "sk-XXX"

# Create the main header for the Streamlit application
st.header("My Chatbot")

# Sidebar for document upload
while st.sidebar:
    st.title("Your Documents")
    # Allow users to upload a PDF file
    file = st.file_uploader("Upload a PDF file", type="pdf")

# Process the uploaded file if it exists
if file is not None:
    # Initialize PDF reader
    pdf_reader = PdfReader(file)
    text = ""

    # Extract text from each page of the PDF
    for page in pdf_reader.pages:
        text += page.extract_text()
        # Uncomment the following line to display the extracted text
        # st.write(text)

    # Split the extracted text into chunks for better processing
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",           # Split text by newlines
        chunk_size=1000,           # Size of each chunk in characters
        chunk_overlap=150,         # Overlap between chunks to maintain context
        length_function=len        # Function to measure chunk length
    )

    # Create chunks of text
    chunks = text_splitter.split_text(text)

    # Generate embeddings for the text chunks using OpenAI
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Store the text chunks in a vector store (FAISS) for similarity search
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Input field for users to ask a question
    ques = st.text_input("Add a question")

    # Initialize the OpenAI language model for answering questions
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,   # Your OpenAI API key
        # Temperature for text generation (0 for deterministic output)
        temperature=0,
        max_tokens=1000,                 # Maximum tokens to generate
        model_name="gpt-3.5-turbo",      # Model name
    )

    # Process the question if provided
    if ques:
        # Perform a similarity search on the text chunks using the question
        match = vector_store.similarity_search(ques)

        # Load a pre-built question-answering chain
        chain = load_qa_chain(llm, chain_type="stuff")

        # Run the chain with the matched documents and question
        response = chain.run(input_documents=match, question=ques)

        # Display the response
        st.write(response)
