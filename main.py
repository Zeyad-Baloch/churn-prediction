from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib

model = joblib.load('churn_predictor_model.pkl')

app = FastAPI(title="Churn Prediction Predictor")

class CustomerData(BaseModel):
    SeniorCitizen: int
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    gender_Male: bool = False
    Partner_Yes: bool = False
    Dependents_Yes: bool = False
    PhoneService_Yes: bool = False
    MultipleLines_No_phone_service: bool = False
    MultipleLines_Yes: bool = False
    InternetService_Fiber_optic: bool = False
    InternetService_No: bool = False
    OnlineSecurity_No_internet_service: bool = False
    OnlineSecurity_Yes: bool = False
    OnlineBackup_No_internet_service: bool = False
    OnlineBackup_Yes: bool = False
    DeviceProtection_No_internet_service: bool = False
    DeviceProtection_Yes: bool = False
    TechSupport_No_internet_service: bool = False
    TechSupport_Yes: bool = False
    StreamingTV_No_internet_service: bool = False
    StreamingTV_Yes: bool = False
    StreamingMovies_No_internet_service: bool = False
    StreamingMovies_Yes: bool = False
    Contract_One_year: bool = False
    Contract_Two_year: bool = False
    PaperlessBilling_Yes: bool = False
    PaymentMethod_Credit_card_automatic: bool = False
    PaymentMethod_Electronic_check: bool = False
    PaymentMethod_Mailed_check: bool = False

@app.get("/")
def root():
    return {"message": "Churn Predictor API is running "}


@app.post("/predict")
def predict_churn(customer: CustomerData):
    try:
  
        data = customer.dict()

     
        data['MultipleLines_No phone service'] = data.pop('MultipleLines_No_phone_service')
        data['InternetService_Fiber optic'] = data.pop('InternetService_Fiber_optic')
        data['OnlineSecurity_No internet service'] = data.pop('OnlineSecurity_No_internet_service')
        data['OnlineBackup_No internet service'] = data.pop('OnlineBackup_No_internet_service')
        data['DeviceProtection_No internet service'] = data.pop('DeviceProtection_No_internet_service')
        data['TechSupport_No internet service'] = data.pop('TechSupport_No_internet_service')
        data['StreamingTV_No internet service'] = data.pop('StreamingTV_No_internet_service')
        data['StreamingMovies_No internet service'] = data.pop('StreamingMovies_No_internet_service')
        data['Contract_One year'] = data.pop('Contract_One_year')
        data['Contract_Two year'] = data.pop('Contract_Two_year')
        data['PaymentMethod_Credit card (automatic)'] = data.pop('PaymentMethod_Credit_card_automatic')
        data['PaymentMethod_Electronic check'] = data.pop('PaymentMethod_Electronic_check')
        data['PaymentMethod_Mailed check'] = data.pop('PaymentMethod_Mailed_check')

        df = pd.DataFrame([data])

        
        expected_columns = model.feature_names_in_  # Get the column order from the model
        df = df.reindex(columns=expected_columns, fill_value=0)  # Reorder and fill missing with 0

        
        probability = model.predict_proba(df)[0][1]
        prediction = model.predict(df)[0]

        return {
            "churn_probability": round(probability, 3),
            "prediction": "Will Churn" if prediction == 1 else "Will Not Churn"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
