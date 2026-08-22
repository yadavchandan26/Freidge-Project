from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error

def train_linear_regression(x_train, y_train):
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model

def train_decision_tree(x_train, y_train, max_depth=None):
    model = DecisionTreeRegressor(
        max_depth=max_depth, random_state=42
    )
    model.fit(x_train, y_train)
    return model

def train_random_forest(x_train, y_train, max_depth=None):
    model = RandomForestRegressor(
        max_depth=max_depth, random_state=42
    )
    model.fit(x_train, y_train)
    return model

def evaluate_model(model, x_test, y_test, model_name: str) -> dict:
    """
    Evaluate regression model and return metrics.
    """

    preds = model.predict(x_test)
    mse = mean_squared_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"\n {model_name} Performance:")
    print(f"MSE : {mse:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2 : {r2:.2f}")

    return {
        "model_name": model_name,
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    }
