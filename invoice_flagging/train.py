from pathlib import Path
import joblib

from datapreprocessing import load_invoice_data, apply_labels, split_data, scale_feature
from model_evaluation import train_random_forest, evaluate_classifier

DB_PATH = "data/inventory.db"

features = ['total_item_dollars', 'total_item_quantity', 'invoice_dollars', 'freight', 'invoice_quantity']
target = 'flag_invoice'

def main():
    Path("models").mkdir(exist_ok=True)

    #load data
    df = load_invoice_data(DB_PATH)
    df = apply_labels(df)

    x_train, x_test, y_train, y_test = split_data(df, features, target)
    x_train_scale, x_test_scale = scale_feature(
        x_train, x_test, 'models/scaler.pkl'
    )

    #using grid model
    grid_model = train_random_forest(x_train_scale, y_train)
    evaluate_classifier(
        grid_model.best_estimator_,
        x_test_scale,
        y_test,
        "Random Forest Classifier"
    )
    #saving the model
    joblib.dump(grid_model.best_estimator_, 'models/predict_flag_invoice.pkl')

if __name__ == "__main__":
    main()
