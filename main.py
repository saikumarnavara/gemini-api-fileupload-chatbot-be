from fastapi import FastAPI, HTTPException,UploadFile,File
import os
from dotenv import load_dotenv
import google.generativeai as genai
import io
import fitz
from PIL import Image
from io import BytesIO




# Load environment variables
load_dotenv()

# FastAPI instance
app = FastAPI()

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
    response = model.generate_content(context,generation_config=genai.types.GenerationConfig(
        candidate_count=1,
        stop_sequences=["x"],
        max_output_tokens=1000,
        temperature=0.1,
    ))
    chat_history.append(response.text)
    return response.text


def generate_document_response(prompt, document_text):
    prompt = (
    f"Given the following document text:\n\n{document_text}\n\n"
    f"Please answer the following question based on the content of the document as accurately and concisely as possible:\n\n"
    f"Question: {prompt}\n\n"
    f"Provide only the information that directly addresses the question without extra details."
    )
    chat_history.append(prompt)
    context = "\n".join(chat_history)  
     # Combine the document content with the question for context
    response = model.generate_content(
    context,
    generation_config=genai.types.GenerationConfig(
        candidate_count=1,
        stop_sequences=["x"], 
        max_output_tokens=1500,  
        temperature=0.1,
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
        stop_sequences=["x"],
        max_output_tokens=100,  
        temperature=0.1,
    ))
    
    # Process suggestions into a list of questions
    suggested_questions = suggestions.text.split("\n") if suggestions.text else []
    
    # Return the main response and cleaned list of suggested questions
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
    response =  model.generate_content([prompt, image_r])  
    return response.text



    



# API ROUTES

@app.post("/text-search/")
async def generate_response(prompt: str):
    try:
        response = generate_gemini_response(prompt)
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
    
@app.post('/document-search/')
async def ask_question(file_id: str, question: str):
    try:
        # Retrieve the document text from in-memory storage
        document_text = uploaded_docs.get(file_id)
        if not document_text:
            raise HTTPException(status_code=404, detail="Document not found.")
        response = generate_document_response(question, document_text)
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



    
   


    




