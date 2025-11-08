import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

def run_ffnn(df_train, df_test, numerical_features, categorical_features, target_col='N_y'):
    # preprocess
    num_imp = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    Xn_tr = scaler.fit_transform(num_imp.fit_transform(df_train[numerical_features]))
    Xn_te = scaler.transform(num_imp.transform(df_test[numerical_features]))

    y_tr = df_train[target_col].to_numpy().reshape(-1,1)
    y_te = df_test[target_col].to_numpy().reshape(-1,1)

    Xn_tr_t = torch.tensor(Xn_tr, dtype=torch.float32)
    Xn_te_t = torch.tensor(Xn_te, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    y_te_t = torch.tensor(y_te, dtype=torch.float32)

    class Net(nn.Module):
        def __init__(self, n_num):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(n_num, 1024), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(128, 1)
            )
        def forward(self, x): return self.layers(x)

    model = Net(Xn_tr_t.shape[1]).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_fn = nn.MSELoss()

    tr_loader = torch.utils.data.DataLoader(list(zip(Xn_tr_t, y_tr_t)), batch_size=512, shuffle=True)
    val_loader = torch.utils.data.DataLoader(list(zip(Xn_te_t, y_te_t)), batch_size=2048)

    best, patience, waited = float('inf'), 10, 0
    for ep in range(100):
        model.train(); tr_loss = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
            tr_loss += loss.item()
        tr_loss /= len(tr_loader)

        model.eval(); val_loss = np.mean([loss_fn(model(xb.to(DEVICE)), yb.to(DEVICE)).item() for xb, yb in val_loader])
        if (ep+1)%10==0: print(f"Epoch {ep+1}: train {tr_loss:.4f} val {val_loss:.4f}")
        if val_loss + 1e-6 < best: best, waited = val_loss, 0; best_state = model.state_dict()
        else:
            waited += 1
            if waited >= patience: break
    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        y_pred = model(Xn_te_t.to(DEVICE)).cpu().numpy().flatten()
    y_true = y_te_t.numpy().flatten()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)

    results = pd.DataFrame({'predicted': y_pred, 'observed': y_true})
    return results, {'MAE': mae, 'RMSE': rmse, 'R2': r2}
