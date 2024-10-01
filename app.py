# from fastapi import FastAPI, Form, Request
# from fastapi.responses import FileResponse, HTMLResponse
# from fastapi.templating import Jinja2Templates
# import numpy as np
# import pickle
# from fpdf import FPDF
# from pydantic import BaseModel
# from sklearn.preprocessing import StandardScaler
# import os
# from transformers import pipeline
# import pandas as pd

# # Initialize FastAPI app
# app = FastAPI()

# # Load the text generation model from Hugging Face
# generator = pipeline('text-generation', model='facebook/bart-large-cnn')

# # Set up Jinja2 templates
# templates = Jinja2Templates(directory="templates")

# # Load the pre-trained model and scaler
# with open("model.pkl", "rb") as f:
#     model = pickle.load(f)

# with open("scaler.pkl", "rb") as f:
#     scaler = pickle.load(f)

# # Define HealthData class for input validation
# class HealthData(BaseModel):
#     age: float
#     weight: float
#     ap_hi: float
#     ap_lo: float
#     cholesterol: int
#     gluc: int

# # Predict cardiovascular risk using the pre-trained model
# def predict_cardiovascular_risk(user_data):
#     # Prepare the data for prediction
#     input_data = pd.DataFrame({
#         'age': [user_data.age],
#         'weight': [user_data.weight],
#         'ap_hi': [user_data.ap_hi],
#         'ap_lo': [user_data.ap_lo],
#         'cholesterol': [user_data.cholesterol],
#         'gluc': [user_data.gluc]
#     })
    
#     # Scale the data using the preloaded scaler
#     scaled_data = scaler.transform(input_data)
    
#     # Predict using the preloaded model
#     prediction = model.predict(scaled_data)
    
#     return int(prediction[0])

# # Generate personalized health report using Hugging Face model
# def generate_health_report(user_data, prediction):
#     prompt = f"""
#     A user has the following health details:
#     - Age: {user_data.age}
#     - Weight: {user_data.weight} kg
#     - Systolic Blood Pressure: {user_data.ap_hi}
#     - Diastolic Blood Pressure: {user_data.ap_lo}
#     - Cholesterol level: {user_data.cholesterol}
#     - Glucose level: {user_data.gluc}
    
#     The user's cardiovascular risk prediction is: {prediction} (0 means low risk, 1 means high risk).

#     Based on this information, please provide a detailed health report, including:
#     - An assessment of the user's cardiovascular health
#     - Recommended lifestyle changes
#     - Dietary suggestions
#     - Any additional medical advice
    
#     Please ensure the report is well-structured and easy to understand.
#     """
    
#     # Generate the health report from the model (make sure to pass just the response)
#     response = generator(prompt, max_length=300, truncation=True, clean_up_tokenization_spaces = True)[0]['generated_text'].strip()
    
#     # Return only the generated report, exclude the prompt
#     return response


# # Create PDF report from the generated text
# def create_pdf_report(report_text, filename="health_report.pdf"):
#     pdf = FPDF()
#     pdf.set_auto_page_break(auto=True, margin=15)
#     pdf.add_page()

#     # Title
#     pdf.set_font("Arial", 'B', 16)
#     pdf.cell(200, 10, "Personalized Health Report", ln=True, align="C")

#     # Line Break
#     pdf.ln(10)

#     # Body
#     pdf.set_font("Arial", '', 12)
#     pdf.multi_cell(0, 10, report_text)

#     # Save the PDF
#     pdf.output(filename)

# # Route to serve the HTML form
# @app.get("/", response_class=HTMLResponse)
# async def serve_form(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# # API route to accept user health data, predict, and generate report
# @app.post("/predict")
# async def predict_health_report(age: float = Form(...), weight: float = Form(...),
#                                 ap_hi: float = Form(...), ap_lo: float = Form(...), 
#                                 cholesterol: int = Form(...), gluc: int = Form(...)):
    
#     # Create HealthData object
#     user_data = HealthData(age=age, weight=weight, ap_hi=ap_hi, ap_lo=ap_lo, cholesterol=cholesterol, gluc=gluc)
    
#     # Make prediction
#     prediction = predict_cardiovascular_risk(user_data)
    
#     # Generate health report using Hugging Face model
#     report_text = generate_health_report(user_data, prediction)
    
#     # Create PDF of the report
#     pdf_filename = "health_report.pdf"
#     create_pdf_report(report_text, filename=pdf_filename)

#     # Return the PDF file
#     return FileResponse(pdf_filename, media_type='application/pdf', filename=pdf_filename)

# # Start the FastAPI server if running directly
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)


from fastapi import FastAPI, Form, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import pickle
from fpdf import FPDF
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
import os
from transformers import pipeline
import pandas as pd
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize FastAP app
app = FastAPI()

# Set up Hugging Face pipeline (using BART or a more powerful model like GPT-2 for now)
generator_pipeline = pipeline('text-generation', model='gpt2')

# Create Hugging Face LLM using LangChain's pipeline interface
llm = HuggingFacePipeline(pipeline=generator_pipeline)

# Define a prompt template using LangChain's PromptTemplate class
prompt_template = PromptTemplate(
    input_variables=["user_data", "prediction"],
    template="""
    A user has the following health details:
    - Age: {user_data.age}
    - Weight: {user_data.weight} kg
    - Systolic Blood Pressure: {user_data.ap_hi}
    - Diastolic Blood Pressure: {user_data.ap_lo}
    - Cholesterol level: {user_data.cholesterol}
    - Glucose level: {user_data.gluc}

    The user's cardiovascular risk prediction is: {prediction} (0 means low risk, 1 means high risk).

    Based on this information, please provide a detailed, human-readable health report, including:
    - An assessment of the user's cardiovascular health
    - Recommended lifestyle changes
    - Dietary suggestions
    - Additional medical advice

    Make sure the report is informative, concise, and easy to understand.
    """
)

# Create LangChain LLM chain to handle the report generation
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Load the pre-trained model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Define HealthData class for input validation
class HealthData(BaseModel):
    age: float
    weight: float
    ap_hi: float
    ap_lo: float
    cholesterol: int
    gluc: int

# Predict cardiovascular risk using the pre-trained model
def predict_cardiovascular_risk(user_data):
    # Prepare the data for prediction
    input_data = pd.DataFrame({
        'age': [user_data.age],
        'weight': [user_data.weight],
        'ap_hi': [user_data.ap_hi],
        'ap_lo': [user_data.ap_lo],
        'cholesterol': [user_data.cholesterol],
        'gluc': [user_data.gluc]
    })
    
    # Scale the data using the preloaded scaler
    scaled_data = scaler.transform(input_data)
    
    # Predict using the preloaded model
    prediction = model.predict(scaled_data)
    
    return int(prediction[0])

# Generate personalized health report using the LangChain agent
def generate_health_report(user_data, prediction):
    # Run the LangChain pipeline with the structured prompt
    report_text = llm_chain.run(user_data=user_data, prediction=prediction)
    return report_text

# Create PDF report from the generated text
def create_pdf_report(report_text, filename="health_report.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "Personalized Health Report", ln=True, align="C")

    # Line Break
    pdf.ln(10)

    # Body
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 10, report_text)

    # Save the PDF
    pdf.output(filename)

# Route to serve the HTML form
@app.get("/", response_class=HTMLResponse)
async def serve_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# API route to accept user health data, predict, and generate report
@app.post("/predict")
async def predict_health_report(age: float = Form(...), weight: float = Form(...),
                                ap_hi: float = Form(...), ap_lo: float = Form(...), 
                                cholesterol: int = Form(...), gluc: int = Form(...)):
    
    # Create HealthData object
    user_data = HealthData(age=age, weight=weight, ap_hi=ap_hi, ap_lo=ap_lo, cholesterol=cholesterol, gluc=gluc)
    
    # Make prediction
    prediction = predict_cardiovascular_risk(user_data)
    
    # Generate health report using the LangChain agent
    report_text = generate_health_report(user_data, prediction)
    
    # Create PDF of the report
    pdf_filename = "health_report.pdf"
    create_pdf_report(report_text, filename=pdf_filename)

    # Return the PDF file
    return FileResponse(pdf_filename, media_type='application/pdf', filename=pdf_filename)

# Start the FastAPI server if running directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
