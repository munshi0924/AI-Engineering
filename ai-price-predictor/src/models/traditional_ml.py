
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def create_models():
    """
    Return a dictionary of regression models.
    """

    return {
        "baseline": DummyRegressor(
            strategy="median"
        ),

        "linear_regression": Ridge(
            alpha=10.0
        ),

        "random_forest": RandomForestRegressor(
            n_estimators=150,
            max_depth=18,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),

        "xgboost": XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        ),
    }
