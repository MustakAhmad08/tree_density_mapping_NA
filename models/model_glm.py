import numpy as np, pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm

def run_glm(df_train, df_test, numerical_features, categorical_features, target_col='N_y'):
    enc = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    enc.fit(pd.concat([df_train[categorical_features], df_test[categorical_features]], axis=0))
    scaler = StandardScaler().fit(df_train[numerical_features])

    def prep(df):
        X_num = scaler.transform(df[numerical_features])
        X_cat = enc.transform(df[categorical_features])
        X = np.hstack([X_num, X_cat])
        y = df[target_col].values
        return X, y

    Xtr, y_tr = prep(df_train)
    Xte, y_te = prep(df_test)

    pca = PCA(n_components=0.95, svd_solver='full')
    Xtr_p, Xte_p = pca.fit_transform(Xtr), pca.transform(Xte)
    Xtr_c, Xte_c = sm.add_constant(Xtr_p), sm.add_constant(Xte_p)

    glm = sm.OLS(y_tr, Xtr_c).fit()
    y_pred = glm.predict(Xte_c)

    mae = mean_absolute_error(y_te, y_pred)
    rmse = mean_squared_error(y_te, y_pred, squared=False)
    r2 = r2_score(y_te, y_pred)

    results = pd.DataFrame({'predicted': y_pred, 'observed': y_te})
    return results, {'MAE': mae, 'RMSE': rmse, 'R2': r2}
