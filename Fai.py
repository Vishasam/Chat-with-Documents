#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Application Name : Chat with Documents
    
# Requirements : 
#     --- openai
#     --- langchain
#     ---- docx2txt
#     --- streamlit                  
#     --- tiktoken
#     --- pypdf
#     --- textloader
#     --- pdfplumber
    

# Run this app with the command in the terminal of your downloaded directory : "streamlit run your_app_name.py"


# In[2]:


# Steps to run this application:
#     Step -1 : Uncomment all the packages to install
#     Step -2 : Run all the cells
#     Step -3 : At the top letf most corner, click
#         File -> Download as -> Python (.py)
#     Step -4 : Open the terminal and change the directory where the dowloaded python file located.
#                In my case, The default path is 

#                "C:\Users\Shala>" and i need to change like this

#                 " C:\Users\Shala> cd Downloads",

#                 and run the file using streamlit command

#             " C:\Users\Shala\Downloads>streamlit run FAI_11.py"


# In[3]:


# --- This application works based on the OpenAI model named "gpt-3.5-turbo" and  used Langchain framework 
#     for Retrieval Question and Answering.To provide a seamless user experience, I have integrated 
#     Streamlit as the frontend interface, allowing for a user-friendly and intuitive interaction.

# --- Once the app runs on the local webpage,the user needs to upload a document which may be in the following
#     any formats of .pdf or .docx or .txt only,making t versatile and accommodating to different user needs.
 
# ---Upon document uploaded,pressing the Proceed button,triggers an instant execution of chunking,parsing 
#    and indexing.The embeddings of the document stored in the vectorbase callede Chroma for 
#    retrieval processes.A message will be displayed, indicating the completion of
#    the Chunking and Embedding steps.


# -- User may make use of the chatbox to ask the relevant question about the
#   uploaded document and the system will provide insightful and accurate answers 
#   based on the extracted information.

#--- It also extracts data about the images and  user may ask questions about the 
#   images by mentioning their names.
#   The chat history is displayed below the chatbox, separated by a horizontal line, 
#   acting as a separator for the current question and existing questions. This design allows users to view
#  their ongoing conversation and past interactions.


# In[5]:


# pip install openai langchain docx2txt tiktoken pypdf textloader pdfplumber streamlit


# In[1]:


# Importing the necessary packages

from langchain.embeddings.openai import OpenAIEmbeddings #Used for embedding the data
from langchain.vectorstores import Chroma #Stores the embedded data 
import streamlit as st
import platform
import openai


# In[2]:


# Setting environmental variable for openai api key
import os
os.environ["OPENAI_API_KEY"] = "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"


# In[3]:


st.set_page_config(
    page_title='Chat With Documents',
    page_icon=":books:",
    menu_items=None
)


# In[4]:


# '''Document Format Handling:
# --- Loading PDF, DOCX and TXT files
# --- os.path.splitext() function used to split the file name and extension
# --- Also extract the file extension from the provided file path'''


# In[5]:


def load_document(file):
    import os
    name, extension = os.path.splitext(file) 
    

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return None
    
    # Reads the document and extracts its content,returning it as a text data
    data = loader.load()
    return data


# In[6]:


# '''
# --- This function is useful for splitting data in manageable chunks
#     which can be helpful when processing large documents or feed data
#     to language models.

# --- RecursiveCharacterTextSplitter splits the  text recursively into smaller chunks 
#     based on a list of characters. It tries to split the text on these characters in order 
#    until the chunks are small enough.This splits based on characters (by default "\n\n") 
#    and measure chunk length by number of characters.
# --- chunk_size :defines the maximum size of each text chunk.Default size of 256
# --- chunk_overlap : determines the number of characters by which consecutive chunks overlap
# --- THe specified chunk size and overlap can be adjusted as needed
# --- The actual text splitting happen in the "text_splitter.split_documents(data)"
# --- Returns a list of smaller text chunks obtained from the splitting process.
# '''


# In[7]:


def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks


# In[8]:


# '''
# --- The create_embeddings() function takes a list of text chunks and create 
#     embeddings using OpenAIEmbeddings() and stores all the embeddings in a 
#    Chroma vector store.

# --- from_documents() method calculates the embedings for each text chunk and the
#    resulting embeddings are stored in a "Chroma" vector store, represented by 
#    the 'vector_store' variable
# '''


# In[9]:


def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store


# In[10]:


# '''
# --- Retrieve and Answering:

# --- Takes three parameters as input:
# --- vector_store : Chroma vectore contains Embeddings of text chunk
# --- q : Users query
# --- k : No.of similar documents to consider during retrieval process

# --- Model will retrieve the top k most similar chunks
# --- Imports two classes from the Langchain library
# --- RetriecalQA : Responsible for retrieving question- answering mechanism,
#    Answer responded based on their similarity to the users query
# --- Chat OpenAI : represents language model for chatbased interation
# --- Provides gpt-3.5-turbo model for language generation capabilities

# --- Temperature determines the randomness of the model's o/p
# --- Higher temperature above 1 gives output more diverse and creative
# --- lower temperature close to 0 gives more focused & deterministic
# --- Creates retriever object for searching and retrieving similar chunks 
# based on ther embeddings similarity to the users query

# --- '''


# In[11]:


def ask_and_get_answer(vector_store, q, k=3):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    answer = chain.run(q)
    return answer


# In[12]:


# '''
# --- calculate the no.of tokens present in a list of text documents and estimating 
#    embedding  cost using tiktoken

#  --- The cost is calculated based on a predefined pricing model,wher 18.76 is the cost per
#     1000 tokens and it can be used  to assess the expenses of using language models that 

#     charge based on token usage.
#  --- Calculates the tokens using the model called text-embedding-ada-002


# In[13]:


def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])

    return total_tokens, (total_tokens / 1000 * 0.0001 * 18.76)


# In[14]:


# '''
# --- Clear the chat history from streamlit session state:
# --- Using clear_history() function, the app can remove the existing chat history
#    and start with a clean slate,allowing user to have a fresh chat session.

# --- Inside the function,it first checks if the key 'history' exists in the
#    st.session state using the 'in' keyword

# --- If the chat history is stored,the function uses the 'del' statement to the 
#    delete the history from the session state.
# '''


# In[15]:


def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']


# In[16]:


############# Main Run Code ##############


# In[17]:


if __name__ == "__main__":
    import os

    # loading the OpenAI api key from .env
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    css_styles = '''
    .css-uf99v8 {
    background-color: #ddb892;
}

    .animate {
        background-image: linear-gradient(
            -225deg,
            #231557 0%,
            #44107a 29%,
            #ff1361 67%,
            #fff800 100%
        );
        background-size: auto auto;
        background-clip: border-box;
        background-size: 200% auto;
        color: #fff;
        background-clip: text;
        text-fill-color: transparent;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: textclip 2s linear infinite;
        display: inline-block;
        font-size: 70px;
    }
    @keyframes textclip {
        to {
            background-position: 200% center;
        }
    }
    '''
    st.markdown(f"<style>{css_styles}</style>", unsafe_allow_html=True)
    st.markdown("<span class='animate'>Chat with Documents</span>", unsafe_allow_html=True)

    if 'mobile' in platform.platform().lower():
        print('Click here to enter file ')

    custom_css = """
    <style>
    /* Apply dark background to the sidebar */
    [data-testid="stHeader"]{
      background-color: #ddb892 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #b08968 !important;
    }

    [data-testid="stDecoration"] {
background-image: linear-gradient(90deg, #b08968, #b08968);    }
    </style>
    """
    

    with st.sidebar:
        st.sidebar.markdown(custom_css, unsafe_allow_html=True)

        st.subheader("Load files")
        # file uploader widget to allow multiple files
        uploaded_files = st.file_uploader('Upload PDF files:', type=['pdf'], accept_multiple_files=True)

        # chunk size number widget
        chunk_size = 512
        k = 3
        # add data button widget
        add_data = st.button('Proceed', on_click=clear_history)
        expanded=True

        if uploaded_files and add_data:  # if the user browsed files
            vector_stores = []
            for file_num, uploaded_file in enumerate(uploaded_files):
                with st.spinner(f'Reading, chunking and embedding file {file_num + 1} ...'):

                    # writing the file from RAM to the current directory on disk
                    bytes_data = uploaded_file.read()
                    file_name = os.path.join('./', uploaded_file.name)
                    with open(file_name, 'wb') as f:
                        f.write(bytes_data)

                    data = load_document(file_name)
                    chunks = chunk_data(data, chunk_size=chunk_size)
                    tokens, embedding_cost = calculate_embedding_cost(chunks)
                    vector_store = create_embeddings(chunks)

                    vector_stores.append(vector_store)
                    st.success(f'File {file_num + 1} uploaded, chunked and embedded successfully.')

            # saving the vector stores in the session state
            st.session_state.vs = vector_stores

    # user's question text input widget
    q = st.text_input('Ask a question about the content of your file:')

    if q:  # if the user entered a question and hit enter
        if 'vs' in st.session_state:  # if there are vector stores (user uploaded, split, and embedded files)
            vector_stores = st.session_state.vs
            answers = []
            for idx, vector_store in enumerate(vector_stores):
                answer = ask_and_get_answer(vector_store, q, k)
                answers.append(answer)

                # text area widget for the LLM answer
                st.text_area(f'Answer {idx + 1}:', value=answer)

                # st.divider()
                st.markdown('<hr>', unsafe_allow_html=True)

            # if there's no chat history in the session state, create it
            if 'history' not in st.session_state:
                st.session_state.history = ''

            # the current questions and answers
            question_answers = '\n'.join([f'YOU : {q} \nBOT {idx + 1} : {answer}' for idx, answer in enumerate(answers)])
            # '''
            # --- Responsible for updating the chat history stored in the st session state.
            # --- Formatted string which concates the current questions and answers with existing
            # --- {"-" * 100}: creates a horizontal line of 100 dashes act as a seperator

            # '''
            st.session_state.history = f'{question_answers} \n {"-" * 100} \n {st.session_state.history}'
            h = st.session_state.history

            # text area widget for the chat history
            st.text_area(label='Chat History', value=h, key='history', height=400)

