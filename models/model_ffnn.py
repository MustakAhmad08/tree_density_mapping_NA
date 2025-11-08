import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

def run_ffnn(df_train, df_test, numerical_features, categorical_features, target_col='N_y'):
    num_imp = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    Xn_tr = scaler.fit_transform(num_imp.fit_transform(df_train[numerical_features]))
    Xn_te = scaler.transform(num_imp.transform(df_test[numerical_features]))
    
    df = pd.concat([df_train, df_test], axis=0)
    cat_encoders = {}
    for f in categorical_features:
        cat = df[f].astype('category')
        cat_encoders[f] = cat.cat.categories
    
    def encode_cats(df_subset):
        encoded = []
        for f in categorical_features:
            cat = df_subset[f].astype('category')
            cat = cat.cat.set_categories(cat_encoders[f])
            codes = cat.cat.codes.values
            codes[codes == -1] = len(cat_encoders[f])
            encoded.append(codes)
        return np.column_stack(encoded) if encoded else np.zeros((len(df_subset), 0), dtype=int)
    
    Xc_tr = encode_cats(df_train)
    Xc_te = encode_cats(df_test)
    
    y_tr = df_train[target_col].to_numpy().reshape(-1, 1)
    y_te = df_test[target_col].to_numpy().reshape(-1, 1)
    
    Xn_tr_t = torch.tensor(Xn_tr, dtype=torch.float32)
    Xn_te_t = torch.tensor(Xn_te, dtype=torch.float32)
    Xc_tr_t = torch.tensor(Xc_tr, dtype=torch.long)
    Xc_te_t = torch.tensor(Xc_te, dtype=torch.long)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    y_te_t = torch.tensor(y_te, dtype=torch.float32)
    
    def emb_dim(n):
        return int(min(50, max(2, round(np.sqrt(n) * 2))))
    
    embedding_sizes = {}
    for f in categorical_features:
        base_n = len(cat_encoders[f])
        if df[f].isna().any():
            base_n += 1
        embedding_sizes[f] = (int(base_n), emb_dim(int(base_n)))
    
    class TabDataset(Dataset):
        def __init__(self, Xn, Xc, y):
            self.Xn = Xn
            self.Xc = Xc
            self.y = y
        def __len__(self):
            return self.Xn.shape[0]
        def __getitem__(self, idx):
            return self.Xn[idx], self.Xc[idx], self.y[idx]
    
    train_ds = TabDataset(Xn_tr_t, Xc_tr_t, y_tr_t)
    val_ds = TabDataset(Xn_te_t, Xc_te_t, y_te_t)
    
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False, num_workers=0, pin_memory=False)
    
    class FeedForwardNN(nn.Module):
        def __init__(self, n_num, embedding_sizes):
            super().__init__()
            self.embs = nn.ModuleList([nn.Embedding(nc, ed) for (nc, ed) in embedding_sizes.values()])
            emb_out = sum(ed for (_, ed) in embedding_sizes.values())
            in_dim = n_num + emb_out
            
            self.layers = nn.Sequential(
                nn.Linear(in_dim, 1024), nn.ReLU(), nn.Dropout(0.10),
                nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.10),
                nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.10),
                nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.10),
                nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.10),
                nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.10),
                nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.10),
            )
            self.out = nn.Linear(128, 1)
        
        def forward(self, x_num, x_cat):
            if len(self.embs):
                embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
                x = torch.cat([x_num] + embs, dim=1)
            else:
                x = x_num
            x = self.layers(x)
            return self.out(x)
    
    model = FeedForwardNN(n_num=Xn_tr_t.shape[1], embedding_sizes=embedding_sizes).to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    
    best_val = float('inf')
    best_state = None
    epochs, patience, waited = 100, 10, 0
    
    for epoch in range(epochs):
        model.train()
        tr_loss, steps = 0.0, 0
        for xb_num, xb_cat, yb in train_loader:
            xb_num = xb_num.to(DEVICE)
            xb_cat = xb_cat.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb_num, xb_cat)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            steps += 1
        tr_loss /= max(1, steps)
        
        model.eval()
        with torch.no_grad():
            vals = []
            for xb_num, xb_cat, yb in val_loader:
                xb_num = xb_num.to(DEVICE)
                xb_cat = xb_cat.to(DEVICE)
                yb = yb.to(DEVICE)
                pv = model(xb_num, xb_cat)
                vals.append(criterion(pv, yb).item())
            val_loss = float(np.mean(vals))
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/{epochs}  train {tr_loss:.4f}  val {val_loss:.4f}")
        
        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            waited = 0
        else:
            waited += 1
            if waited >= patience:
                print(f"Early stopping at epoch {epoch+1}. Best val {best_val:.4f}")
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    model.eval()
    with torch.no_grad():
        y_pred = model(Xn_te_t.to(DEVICE), Xc_te_t.to(DEVICE)).cpu().numpy().flatten()
    
    y_true = y_te_t.numpy().flatten()
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    results = pd.DataFrame({'predicted': y_pred, 'observed': y_true})
    
    return results, {'MAE': mae, 'RMSE': rmse, 'R2': r2}
