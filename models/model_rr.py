import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def run_ridge_regression(df_train, df_test, numerical_features, categorical_features, target_col='N_y'):
    Xtr = df_train[numerical_features + categorical_features]
    Xte = df_test[numerical_features + categorical_features]
    y_tr, y_te = df_train[target_col], df_test[target_col]

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ])

    pipe = Pipeline([('prep', preprocessor), ('ridge', Ridge(alpha=1.0, random_state=42))])
    pipe.fit(Xtr, y_tr)
    y_pred = pipe.predict(Xte)

    mae = mean_absolute_error(y_te, y_pred)
    rmse = mean_squared_error(y_te, y_pred, squared=False)
    r2 = r2_score(y_te, y_pred)

    results = pd.DataFrame({'predicted': y_pred, 'observed': y_te})
    return results, {'MAE': mae, 'RMSE': rmse, 'R2': r2}
