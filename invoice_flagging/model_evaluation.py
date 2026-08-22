from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, make_scorer

def train_random_forest(x_train, y_train):
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    )

    params = {
        'n_estimators': [51, 101, 151],
        'max_depth': [None, 4, 5, 6],
        'min_samples_split': [2, 3, 4],
        'criterion': ['gini', 'entropy']
    }
    score_ = make_scorer(accuracy_score)
    grid_model = GridSearchCV(
        estimator=rf,
        param_grid=params,
        scoring=score_,
        cv=3,
        n_jobs=-1,
    )

    grid_model.fit(x_train, y_train)
    return grid_model

def evaluate_classifier(model, x_test, y_test, model_name):
    preds = model.predict(x_test)

    accuracy = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)

    print(f"\n {model_name} Performance:")
    print(f" accuracy:{accuracy:.2f}")
    print(report)
