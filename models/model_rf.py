import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def run_random_forest(df_train, df_test, numerical_features, categorical_features, target_col='N_y'):
    Xtr_num, Xte_num = df_train[numerical_features], df_test[numerical_features]
    Xtr_cat, Xte_cat = df_train[categorical_features], df_test[categorical_features]
    y_tr, y_te = df_train[target_col], df_test[target_col]
    
    scaler = StandardScaler().fit(Xtr_num)
    Xtr_num_s, Xte_num_s = scaler.transform(Xtr_num), scaler.transform(Xte_num)
    
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(pd.concat([Xtr_cat, Xte_cat]))
    Xtr_cat_e, Xte_cat_e = ohe.transform(Xtr_cat), ohe.transform(Xte_cat)
    
    Xtr, Xte = np.hstack((Xtr_num_s, Xtr_cat_e)), np.hstack((Xte_num_s, Xte_cat_e))
    
    model = RandomForestRegressor(
        n_estimators=500, max_depth=20, min_samples_split=10, min_samples_leaf=5,
        random_state=42, n_jobs=-1
    )
    model.fit(Xtr, y_tr)
    
    y_pred = model.predict(Xte)
    
    mae = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    r2 = r2_score(y_te, y_pred)
    
    results = pd.DataFrame({'predicted': y_pred, 'observed': y_te.values})
    
    return results, {'MAE': mae, 'RMSE': rmse, 'R2': r2}
