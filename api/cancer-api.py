from fastapi import FastAPI, Query 
import uvicorn
import pandas as pd
import pickle
import base64

app = FastAPI(title="Cancer Prediction")

@app.get("/")
def homepage(): 
    return {"message": "Cancer Prediction"}


@app.get("/cancer")
def cancer_classifier(age:int = Query(description="Age",
                                               default=51,
                                               ge=20, le=80),
                      gender:int = Query(description="Gender",
                                               default=0,
                                               ge=0, le=1),
                      bmi:float = Query(description="BMI",
                                               default=27.5,
                                               ge=15.0, le=40.0),
                      smoking:int = Query(description="Smoking",
                                               default=0,
                                               ge=0, le=1),
                      genetic_risk:int = Query(description="GeneticRisk",
                                               default=0,
                                               ge=0, le=2),
                      physical_activity:float = Query(description="PhysicalActivity",
                                               default=0.0,
                                               ge=0.0, le=10.0),
                      alcohol_intake:float = Query(description="AlcoholIntake",
                                               default=0,
                                               ge=0.0, le=5.0),
                      cancer_history:int = Query(description="CancerHistory",
                                               default=0,
                                               ge=0, le=1),
                      model_type: str = Query(description="Model type", default= "Neural Network")
                   ):            
    
    
    # Creating the dataframe
    column_names = ["Age", "Gender", "BMI", "Smoking", "GeneticRisk", 
                    "PhysicalActivity", "AlcoholIntake", "CancerHistory"]

    data = [[age, gender, bmi, smoking, genetic_risk, 
             physical_activity, alcohol_intake, cancer_history]]

    df = pd.DataFrame(data=data, columns=column_names)


    # Loading the pipelines and selecting the model based chosen
    if model_type == "Neural Network":
        pipe = pickle.load(open("models/cancer-pipe-nn.pkl", "rb"))
    elif model_type == "XGBoost":
        pipe = pickle.load(open("models/cancer-pipe-ml.pkl", "rb"))
    
    predictions = pipe.predict(df)
    predictions = predictions[0].tolist() # tolist is used to convert a series to list

    if predictions == 0:
        with open("images/0.jpg", "rb") as f:
            binary_img = f.read()
        
    elif predictions == 1:
        with open("images/1.jpg", "rb") as f:
            binary_img = f.read()

    encoded_img = base64.b64encode(binary_img)
    return {"image": encoded_img}


if __name__ == "__main__":
    uvicorn.run("cancer-api:app",
                port=8000,
                host="0.0.0.0",
                reload=True)