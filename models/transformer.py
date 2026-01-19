import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import copy
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import numpy as np
from torchmetrics.classification import F1Score
import warnings

num_classes = 3
patch_size = 25                      # meaning #patch_size of timesteps are corresponding to one token
sequence_length = 75                 # 1.5 sec with 50Hz
attention_heads = 8
embed_dim = 320                      # usually 768 or 512
transformer_blocks = 6
mlp_nodes = 512
num_channels = 9
nr_tokens = sequence_length // patch_size

class PatchEmbdedding(nn.Module):
    def __init__(self):
        super(PatchEmbdedding, self).__init__()
        self.patch_embed = nn.Conv1d(in_channels=num_channels, out_channels=embed_dim, kernel_size = patch_size, stride = patch_size)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.patch_embed(x)
        x = x.transpose(1,2)        
        return x

class TransformerEncoder(nn.Module):
    def __init__(self):
        super(TransformerEncoder, self).__init__()
        self.ln1 = nn.LayerNorm(embed_dim)        
        self.multi_head_attention = nn.MultiheadAttention(embed_dim, attention_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_nodes),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(mlp_nodes, mlp_nodes),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(mlp_nodes, embed_dim)
        )

    def forward(self, x):
        residual1 = x
        x = self.ln1(x)
        x = self.multi_head_attention(x, x, x)[0] + residual1   # Q, K, V = x, x ,x
        residual2 = x
        x = self.ln2(x)
        x = self.mlp(x) + residual2
        return x

class MLP_Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        #x = x[0, :]
        x = self.ln1(x)
        x = self.mlp_head(x)
        return x
    
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_emdedding = PatchEmbdedding()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))  # for each image in the batch (1) and each set of patches add (1) CLS token of dim=embed_dim
        self.position_embedding = nn.Parameter(torch.randn(1, nr_tokens+1, embed_dim))
        self.transformer_block = nn.Sequential(*[TransformerEncoder() for _ in range(transformer_blocks)])   # mehrere encoder hintereinander 
        self.mlp_head = MLP_Head()

    def forward(self, x):
        x = self.patch_emdedding(x)
        B = x.shape[0]  # last batch might be smaller than batch_size
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)   # .cat() does concatination
        x = x + self.position_embedding
        x = self.transformer_block(x)
        x = x[:, 0]   # remember, shape is (batch, cls + tokens, embed_dim), somit [:,0] returns the cls token of all images from the batch
        x = self.mlp_head(x)
        return x
    
    def evaluate(self, X, y, return_dict=True, verbose=0):
        # 1. Determine the device the model is currently on
        device = next(self.parameters()).device
        
        # 2. Convert NumPy array to Tensor and move to device
        if isinstance(X, np.ndarray):
            X_tensor = torch.from_numpy(X).float().to(device)
        else:
            X_tensor = X.to(device)

        # 3. Run prediction without calculating gradients
        self.eval() # Set model to evaluation mode
        with torch.no_grad():
            logits = self.forward(X_tensor)
            # Get the class index with the highest probability
            predictions = torch.argmax(logits, dim=1)
            # Move back to CPU and convert to NumPy for sklearn metrics
            y_pred = predictions.cpu().numpy()

        # Ensure y is a numpy array for comparison
        y = np.asarray(y).reshape(-1)

       # --- balanced accuracy ONLY between classes 1 and 2 ---
        mask_12 = np.isin(y, [1, 2])
        y_12 = y[mask_12]
        y_pred_subset = y_pred[mask_12]

        # If the model predicted 0 for a sample that was actually 1 or 2,
        # balanced_accuracy_score will complain. 
        # We keep 1 and 2 as they are, but replace 0 (or anything else) with -1.
        y_pred_12 = np.where(np.isin(y_pred_subset, [1, 2]), y_pred_subset, -1)

        if len(y_12) > 0:
            with warnings.catch_warnings():
                # This silences the specific warning about y_pred having classes not in y_true
                warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
                bal_acc_12 = balanced_accuracy_score(y_12, y_pred_subset)
        else:
            bal_acc_12 = 0.0

        # f1 per class
        f1_scores = f1_score(y_true=y, y_pred=y_pred, average=None, labels=[0, 1, 2])
        print(f"Test Set F1_slouch: {f1_scores[2]}")
        if return_dict:
            return {
                "BA_no_upr": bal_acc_12,
                "f1_sl": f1_scores[2]
            }

        return bal_acc_12

    
    
def train_transformer(X_train, y_train, X_val, y_val, epochs=150):

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    transformer = Transformer().to(device)

    # 1. Prepare DataLoaders (This is the "M2 Speed Boost")
    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long().squeeze())
    val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long().squeeze())

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    # (7462, 75, 9)
    # (7462, 1)

    optimizer = optim.Adam(transformer.parameters(), lr=5e-4, weight_decay=5e-3) # weight_decay = 1e-4
    weights = torch.tensor([0.8, 1.0, 1.15], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, min_lr=5e-6)
    f1_train = F1Score(task='multiclass', num_classes=num_classes).to(device)
    f1_val = F1Score(task='multiclass', num_classes=num_classes).to(device)

    # for early stopping
    early_st_patience = 10
    best_val = -math.inf
    bad_epochs = 0
    best_state = None

    transformer.train()
    f1_train.reset()
    for epoch in range(epochs):
        train_loss = 0
        total, correct = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            y_pred = transformer(x)
            loss = criterion(y_pred, y)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            predicted = torch.argmax(y_pred, dim=1)
            f1_train.update(predicted, y)
            correct += (predicted==y).sum().item()
            total += x.shape[0]

        epoch_f1_train = f1_train.compute()
        train_loss /= total
        train_acc = correct/total

        # LR update
        total_val, correct_val = 0, 0
        transformer.eval()
        val_loss = 0
        f1_val.reset()

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = transformer(x)
                val_loss += criterion(pred, y).item()
                f1_val.update(torch.argmax(pred, dim=1), y)
                correct_val += (torch.argmax(pred, dim=1)==y).sum().item()
                total_val += x.shape[0]

        epoch_f1_val = f1_val.compute()
        val_acc = correct_val/total_val
        val_loss /= total_val
        scheduler.step(val_loss)

        # early stopping
        if epoch_f1_val > best_val:       # <
            best_val = epoch_f1_val       # val_loss
            bad_epochs = 0
            best_state = copy.deepcopy(transformer.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= early_st_patience:
                if best_state is not None:
                    transformer.load_state_dict(best_state)
                break
            
        print(f"Epoch {epoch} train_f1: {epoch_f1_train} || val_f1: {epoch_f1_val} || train_loss: {train_loss} || val_loss: {val_loss} || lr: {optimizer.param_groups[0]['lr']:.6f}")

    history, name = None, "transformer"

    return transformer, history, name, device
            