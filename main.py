from fastapi import FastAPI, HTTPException,UploadFile,File
import os
from dotenv import load_dotenv
import google.generativeai as genai
import io
import fitz
from PIL import Image
from io import BytesIO
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
# import faiss
# import numpy as np
# from transformers import AutoModel, AutoTokenizer


# Load environment variables
load_dotenv()

# FastAPI instance
app = FastAPI()


origins = [
    "http://localhost",          
    "http://localhost:3000",     
    "http://localhost:5173/",       
    "*"                        
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # List of allowed origins
    allow_credentials=True,           # Allow cookies and credentials
    allow_methods=["*"],              # Allow all HTTP methods
    allow_headers=["*"],              # Allow all headers
)

# Set the API key for google.generativeai explicitly
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    raise RuntimeError("Google API Key is not set in environment variables.")


model = genai.GenerativeModel('gemini-1.5-flash')

# In-memory store for extracted document text
uploaded_docs = {}
chat_history = []

def extract_text_from_pdf(file_io):
    text = ""
    pdf_document = fitz.open(stream=file_io, filetype="pdf")
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        text += page.get_text()
    pdf_document.close()
    return text

def generate_gemini_response(prompt):
    chat_history.append(prompt)
    context = "\n".join(chat_history)
    formatted_prompt = (
        f"{context}\n\n"
        "Please respond to the following prompt in Markdown format, using bullet points, lists, tables, or headers as needed. "
        "Ensure that the response is concise, relevant, and structured for easy readability.\n\n"
        f"**Prompt:** {prompt}\n\n"
        "**Instructions:**\n"
        "- Provide direct, well-structured answers to the question.\n"
        "- Use Markdown elements like headers (`#`), bullet points (`-`), tables, or numbered lists to organize information.\n"
        "- Focus on accuracy and clarity, answering based on the context of previous messages.\n\n"
        "Return only the Markdown response text."
    )
    response = model.generate_content(
        formatted_prompt,
        generation_config=genai.types.GenerationConfig(
            candidate_count=1,
            stop_sequences=[],
            temperature=0.5,
            # max_output_tokens=1000,
        )
    )
    chat_history.append(response.text.strip())

    return response.text.strip()



def generate_document_response(prompt, document_text):
    prompt = (
    f"## Document Text\n"
    f"{document_text}\n\n"
    f"## Question\n"
    f"{prompt}\n\n"
    f"---\n\n"
    f"**Response:**\n"
    f"Based on the document, provide an answer to the above question as accurately and concisely as possible. "
    f"Use only information that directly addresses the question without extra details."
   )


    chat_history.append(prompt)
    context = "\n".join(chat_history)  
     # Combine the document content with the question for context
    response = model.generate_content(
    context,
    generation_config=genai.types.GenerationConfig(
        candidate_count=1,
        stop_sequences=[], 
        # max_output_tokens=10000,  
        temperature=0.5,
    )
)
    chat_history.append(response.text)
   
    suggestive_prompt = ( 
    f"Based on the document text below:\n\n{document_text}\n\n"
    f"Generate the three most relevant follow-up questions for the following question: '{prompt}'."
    f"These questions should help the user explore important details, clarify key points, or deepen understanding."
    f"Only output the questions in the following format:\n\n"
    f"1. [First question]\n"
    f"2. [Second question]\n"
    f"3. [Third question]\n\n"
    f"Do not include any additional text or explanations."
)
    suggestions = model.generate_content(suggestive_prompt, generation_config=genai.types.GenerationConfig(
        candidate_count=1,
        stop_sequences=[],
        # max_output_tokens=200,  
        temperature=0.5,
    ))
    suggested_questions = suggestions.text.split("\n") if suggestions.text else []
    return {
        "response": response.text,
        "suggested_questions": suggested_questions
    }



     


def upload_file_to_gemini(sourceFile):
    response = genai.upload_file(sourceFile, mime_type="application/pdf")
    # Check for the presence of an ID or other unique identifier
    if hasattr(response, 'id'):
        return response.id
    elif hasattr(response, 'name'):
        return response.name  
    else:
        raise ValueError("File upload response does not contain a file ID or unique identifier.")

def multimodal_search(image_data, prompt):
    image_r = Image.open(BytesIO(image_data)) 
    prompt__ = (
        f"Analyze the provided image in detail. Identify key visual elements, context, and relevant patterns "
        f"or information that could answer the question below as accurately as possible:\n\n"
        f"Image Analysis:\n\n"
        f"Based on the image content, please answer the following question:\n\n"
        f"**Question:** {prompt}\n\n"
        f"**Instructions:** Provide a concise, well-structured answer in Markdown format, including "
        f"bullet points, lists, or tables if they help to present the information clearly. Respond directly "
        f"to the question, focusing on visual evidence from the image."
    )
    response =  model.generate_content([prompt__, image_r])  
    return response.text


class PromptRequest(BaseModel):
    prompt: str
class DocumentSearchRequest(BaseModel):
    file_id: str
    question: str
    



# API ROUTES

@app.post("/text-search/")
async def generate_response(request: PromptRequest):
    try:
        response = generate_gemini_response(request.prompt)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post('/upload-source-file/')
async def upload_source_file(sourceFile: UploadFile = File(...)):
    try:
        file_bytes = await sourceFile.read()
        # Extract text from the PDF
        file_io = io.BytesIO(file_bytes)
        document_text = extract_text_from_pdf(file_io)
        
        # Store the document text with a unique ID
        file_id = sourceFile.filename
        uploaded_docs[file_id] = document_text
        
        return {"message": "File uploaded successfully", "file_id": file_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


@app.post("/document-search/")
async def ask_question(request: DocumentSearchRequest):
    try:
        # Retrieve the document text from in-memory storage
        document_text = uploaded_docs.get(request.file_id)
        if not document_text:
            raise HTTPException(status_code=404, detail="Document not found.")
        
        # Generate a response based on the document text and question
        response = generate_document_response(request.question, document_text)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get('/list-documents/')
async def list_documents():
    try:
        # List document IDs and count the total number
        document_ids = list(uploaded_docs.keys())
        total_documents = len(document_ids)
        
        return {"total_documents": total_documents, "document_ids": document_ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.delete('/delete-document/{file_id}')
async def delete_document(file_id: str):
    try:
        # Check if the document exists in the dictionary
        if file_id in uploaded_docs:
            # Delete the document from the dictionary
            del uploaded_docs[file_id]
            return {"message": f"Document with ID '{file_id}' has been deleted successfully."}
        else:
            # Document not found
            raise HTTPException(status_code=404, detail="Document not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post('/multimodal-search/')
async def multimodal_search_endpoint(image: UploadFile = File(...), prompt: str = None):
    try:
        image_data = await image.read()
        response =  multimodal_search(image_data, prompt)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/')
async def root():
    return {"message": "Welcome to the Gemini API!"}



    
   


    




