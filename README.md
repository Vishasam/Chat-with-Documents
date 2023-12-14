# Chat-with-Documents

Task :
To build a chat application that allows users to have interactive conversations with the content of the uploaded documents

This application works based on the OpenAI model named "gpt-3.5-turbo" and  used Langchain framework  for Retrieval Question and Answering.To provide a seamless user experience, I have integrated Streamlit as the frontend interface, allowing for a user-friendly and intuitive interaction.

Once the app runs on the local webpage,the user needs to upload a document which may be in the following formats of .pdf or .docx or .txt only,making it versatile and accommodating to different user needs.
 
Upon document uploaded,pressing the Proceed button,triggers an instant execution of chunking,parsing  and indexing.The embeddings of the document stored in the vectorbase callede Chroma for retrieval processes.A message will be displayed, indicating the completion of  the Chunking and Embedding steps.

Users may make use of the chatbox to ask the relevant question about the  uploaded document and the system will provide insightful and accurate answers based on the extracted information.

It also extracts data about the images and  users may ask questions about the images by mentioning their names.The chat history is displayed below the chatbox, separated by a horizontal line, acting as a separator for the current question and existing questions. This design allows users to view  their ongoing conversation and past interactions.


Sample Result : 



<img width="575" alt="image" src="https://github.com/ShaliniMuthukumar/Chat-with-Documents/assets/106624891/23e6070a-396d-42df-827c-a62320654d76">



<img width="569" alt="image" src="https://github.com/ShaliniMuthukumar/Chat-with-Documents/assets/106624891/5aaae660-e1ac-498e-b03d-39a5fd6b7b52">

<img width="569" alt="image" src="https://github.com/ShaliniMuthukumar/Chat-with-Documents/assets/106624891/0e35b3cc-180c-4f09-89d8-70a09a15979b">




Here the user query was indicated as YOU and the RAG result was indicated as BOT.

<img width="566" alt="image" src="https://github.com/ShaliniMuthukumar/Chat-with-Documents/assets/106624891/51e3dfb4-e1b1-451e-b82b-2f80e9e95287">



You can see that the app provides chat history for the current session state.

