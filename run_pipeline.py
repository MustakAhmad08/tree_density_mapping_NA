import argparse, pandas as pd
from models.model_ffnn import run_ffnn
from models.model_rf import run_random_forest
from models.model_rr import run_ridge_regression
from models.model_glm import run_glm

CAT_FEATURES = ['bio6_binned']
NUM_FEATURES = [
    'livestock','rh100','GEDIBiomass','Evapot','HumanFP','corr_annual_precip','rmf',
    'bio1','bio2','bio3','bio4','bio6','bio7','bio8','bio9','bio10','bio11',
    'bio14','bio15','bio17','bio18','bio19','maxEVI','maxLAI','meanFpar','maxNDVI',
    'sumET','sumPET','BD','CN','pH','EC','Clay','OCC','roughness','slope','tcurv','tri','elevation'
]

def main(model_type="FFNN"):
    df_train = pd.read_csv("2025_20May_train.csv")
    df_test  = pd.read_csv("2025_20May_test.csv")

    if model_type.upper() == "FFNN":
        results, metrics = run_ffnn(df_train, df_test, NUM_FEATURES, CAT_FEATURES, target_col='N_y')
    elif model_type.upper() == "RF":
        results, metrics = run_random_forest(df_train, df_test, NUM_FEATURES, CAT_FEATURES, target_col='N_y')
    elif model_type.upper() == "RR":
        results, metrics = run_ridge_regression(df_train, df_test, NUM_FEATURES, CAT_FEATURES, target_col='N_y')
    elif model_type.upper() == "GLM":
        results, metrics = run_glm(df_train, df_test, NUM_FEATURES, CAT_FEATURES, target_col='N_y')
    else:
        raise ValueError("Model must be one of: FFNN | RF | RR | GLM")

    out_path = f"outputs/{model_type.upper()}_predicted.csv"
    results.to_csv(out_path, index=False)
    print(metrics)
    print(f"Saved predictions to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="FFNN")
    args = parser.parse_args()
    main(args.model)
