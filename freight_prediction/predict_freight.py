import pandas as pd
import joblib

model_path = 'models/predict_freight_model.pkl'

def load_model(model_path: str = model_path):
    """
    load trained freight cost prediction model .
    """
    model = joblib.load(model_path)
    return model

def predict_freight_cost(input_data):
    """
    predict freight cost for new vendor invoices.
    """
    model = load_model()
    input_df = pd.DataFrame(input_data)
    input_df['Predict_Freight'] = model.predict(input_df)
    return input_df

if __name__ == "__main__":

    #just an example
    sample_data = {
        'Dollars': [1900, 10000, 13000]
    }

    prediction = predict_freight_cost(sample_data)
    print(prediction)
