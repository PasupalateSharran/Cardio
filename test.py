import numpy as np
import pickle
from transformers import pipeline
import pandas as pd

# Load the text generation model from Hugging Face
generator = pipeline('text-generation', model='facebook/bart-large-cnn')

# Load the pre-trained model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Predict cardiovascular risk using the pre-trained model
def predict_cardiovascular_risk(age, weight, ap_hi, ap_lo, cholesterol, gluc):
    # Prepare the data for prediction
    input_data = pd.DataFrame({
        'age': [age],
        'weight': [weight],
        'ap_hi': [ap_hi],
        'ap_lo': [ap_lo],
        'cholesterol': [cholesterol],
        'gluc': [gluc]
    })
    
    # Scale the data using the preloaded scaler
    scaled_data = scaler.transform(input_data)
    
    # Predict using the preloaded model
    prediction = model.predict(scaled_data)
    
    return int(prediction[0])

# Generate personalized health report using Hugging Face model
def generate_health_report(age, weight, ap_hi, ap_lo, cholesterol, gluc, prediction):
    prompt = f"""
    A user has the following health details:
    - Age: {age}
    - Weight: {weight} kg
    - Systolic Blood Pressure: {ap_hi}
    - Diastolic Blood Pressure: {ap_lo}
    - Cholesterol level: {cholesterol}
    - Glucose level: {gluc}

    The user's cardiovascular risk prediction is: {prediction} (0 means low risk, 1 means high risk).

    Based on this information, please provide a detailed health report, including:
    - An assessment of the user's cardiovascular health
    - Recommended lifestyle changes
    - Dietary suggestions
    - Any additional medical advice
    
    Please ensure the report is well-structured and easy to understand.
    """
    
    # Generate the health report from the model
    response = generator(prompt, max_length=300, truncation=True, clean_up_tokenization_spaces=True)[0]['generated_text'].strip()

    # Return the generated report, exclude the prompt
    return response

# Function to take user input and run the prediction
def main():
    print("Enter user health data:")

    # Get user input for health data
    age = float(input("Age: "))
    weight = float(input("Weight (kg): "))
    ap_hi = float(input("Systolic Blood Pressure: "))
    ap_lo = float(input("Diastolic Blood Pressure: "))
    cholesterol = int(input("Cholesterol level (1, 2, 3): "))
    gluc = int(input("Glucose level (1, 2, 3): "))

    # Make prediction
    prediction = predict_cardiovascular_risk(age, weight, ap_hi, ap_lo, cholesterol, gluc)

    # Generate health report using Hugging Face model
    report_text = generate_health_report(age, weight, ap_hi, ap_lo, cholesterol, gluc, prediction)
    
    # Print the report in the terminal
    print("\n--- Generated Health Report ---\n")
    print(report_text)

if __name__ == "__main__":
    main()
