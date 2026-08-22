import pandas as pd
import joblib

model_path = 'models/predict_flag_invoice.pkl'
scaler_path = 'models/scaler.pkl'

def load_model(model_path: str = model_path):
    """
    load trained classification model .
    """
    model = joblib.load(model_path)
    return model

def predict_invoice_flag(input_data):
    """
    predict invoice flag for new vendor invoices.
    """
    model = load_model()
    scaler = joblib.load(scaler_path)

    input_df = pd.DataFrame(input_data)
    input_scaled = scaler.transform(input_df)
    input_df['Predict_Flag'] = model.predict(input_scaled)
    return input_df

if __name__ == '__main__':

    #just an example - columns must match training feature order:
    #['total_item_dollars','total_item_quantity','invoice_dollars','freight','invoice_quantity']
    sample_data = {
        'total_item_dollars': [15000, 8000],
        'total_item_quantity': [500, 300],
        'invoice_dollars': [15000, 8200],
        'freight': [200, 150],
        'invoice_quantity': [500, 300]
    }

    prediction = predict_invoice_flag(sample_data)
    print(prediction)
