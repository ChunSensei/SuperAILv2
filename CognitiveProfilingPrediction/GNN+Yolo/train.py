from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score 
import sys
import pickle
import os
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn

# Import tools
FOLDER_TOOLS = "/home/siamai/data/chuniji/ajanPLUSPLUS"
sys.path.append(FOLDER_TOOLS)
import Utils.feeder_skeleton as feeder_skeleton
from Utils.graph_model import Model
from Utils.Loss import ComboLoss

###########################################
# Check if GPU is available
print("GPU Available: ", torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print(device)

# Set Parameters

batch_size = 16
optim_name = "SGD"

#split path
# === SPLIT DATA WITHOUT MERGING ===
import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import shutil
import pandas as pd

# === CONFIG ===
npy_folder = "/home/siamai/data/chuniji/ajanPLUSPLUS/outputnpy"
label_path = "/home/siamai/data/chuniji/pklfile/dawddda.pkl"
output_root = "/home/siamai/data/chuniji/ajanPLUSPLUS/data_split"
train_out = os.path.join(output_root, "train_data")
val_out = os.path.join(output_root, "val_data")
os.makedirs(train_out, exist_ok=True)
os.makedirs(val_out, exist_ok=True)

# === Load label DataFrame ===
df = pickle.load(open(label_path, 'rb'))  # pandas DataFrame
label_columns = df.columns.tolist()[1:]

# === Match .npy files to DataFrame entries (and filter shape)
npy_files = [f for f in os.listdir(npy_folder) if f.endswith(".npy")]
valid_entries = []

for file in npy_files:
    full_path = os.path.join(npy_folder, file)
    try:
        arr = np.load(full_path)
        if arr.shape != (150, 17, 2):
            continue
    except:
        continue

    basename = file.replace(".npy", "")  # e.g. "s01_smiley_t0405"

    # Check if this basename exists in the pkl dataframe 'Filename' column
    row_matches = df[df["Filename"] == basename]
    if not row_matches.empty:
        for i in row_matches.index:
            if i not in [e["index"] for e in valid_entries]:
                valid_entries.append({"index": i, "file": file})
                break

print(f"✅ Matched {len(valid_entries)} npy files to DataFrame rows")

# === Split train/val
train_entries, val_entries = train_test_split(valid_entries, test_size=0.2, random_state=42)

# === Copy files to split folders
for e in train_entries:
    shutil.copy(os.path.join(npy_folder, e["file"]), os.path.join(train_out, e["file"]))
for e in val_entries:
    shutil.copy(os.path.join(npy_folder, e["file"]), os.path.join(val_out, e["file"]))

# === Save labels .pkl files
df_train = df.iloc[[e["index"] for e in train_entries]]
df_val = df.iloc[[e["index"] for e in val_entries]]

df_train.to_pickle(os.path.join(output_root, "train_label.pkl"))
df_val.to_pickle(os.path.join(output_root, "val_label.pkl"))

print("✅ Final split and save complete")

# === Set paths for Feeder (which expects folder of .npy files)
train_data_path = train_out
train_label_path = os.path.join(output_root, "train_label.pkl")

val_data_path = val_out
val_label_path = os.path.join(output_root, "val_label.pkl")

# Data loaders
data_train_feeder = feeder_skeleton.Feeder(train_data_path, train_label_path)
val_test_feeder = feeder_skeleton.Feeder(val_data_path, val_label_path)

train_dataloader = DataLoader(data_train_feeder, batch_size=batch_size, shuffle=False, num_workers=4)
val_dataloader = DataLoader(val_test_feeder, batch_size=batch_size, shuffle=False, num_workers=4)

# Model creation
in_channels = 2
num_class = 12
graph_args = {
    "layout": "yolo",  
    "strategy": "uniform", 
    "max_hop": 1,  
    "dilation": 1  
}

edge_importance_weighting = False
kwargs = {}

model = Model(in_channels, num_class, graph_args, edge_importance_weighting, **kwargs)
model = model.to(device)

weight_path = "/home/siamai/data/aj_art2/weight_saved/best_model_epoch297_F1macro0.5728.pth"
model.load_state_dict(torch.load(weight_path, map_location=device))
print(f"Loaded weights from {weight_path}")


def optimizer_setting(optim_name=optim_name, learning_rate=0.1):
    if optim_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optim_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("Unknown optimizer name.")
    return optimizer

def traintest(model, dataloader, criterion, device, optimizer, mode='train', savemodel=False): 
    
    losses = []
    yall = []
    ypredictlabelall = []
    correct = 0
    numberofdata = 0

    if mode == 'train':
        model.train()
    else:
        model.eval()

    for _, (X, y, names) in tqdm(enumerate(dataloader), total=len(dataloader)):
        X = X.to(device)

        # print(X, X.shape)
        # breaktrain_data_path,

        y = y.to(device)

        # --- Forward ---
        ypredict = model(X)            # [B, 12]
        loss = criterion(ypredict, y.float())

        # --- Prediction: Sigmoid + Threshold ---
        ypredict_sigmoid = torch.sigmoid(ypredict)
        ypredictlabel = (ypredict_sigmoid > 0.5).int()
        ylabel = y.int()

        # --- Accuracy (element-wise) ---
        correct += (ypredictlabel == ylabel).sum().item()
        numberofdata += ylabel.numel()     # B*12

        losses.append(loss.item())

        yall.extend(list(ylabel.cpu().detach().numpy()))
        ypredictlabelall.extend(list(ypredictlabel.cpu().detach().numpy()))

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # --- Confusion Matrix & F1 ---
    yall_np = np.array(yall).reshape(-1, 12)
    ypredictlabelall_np = np.array(ypredictlabelall).reshape(-1, 12)
    confusionmatrix = []
    for i in range(12):
        confusionmatrix.append(confusion_matrix(yall_np[:, i], ypredictlabelall_np[:, i]))

    accuracy = correct / numberofdata  # element-wise accuracy

    # F1 Scores
    f1_micro = f1_score(yall_np, ypredictlabelall_np, average='micro', zero_division=0)
    f1_macro = f1_score(yall_np, ypredictlabelall_np, average='macro', zero_division=0)
    f1_weighted = f1_score(yall_np, ypredictlabelall_np, average='weighted', zero_division=0)
    f1_per_class = f1_score(yall_np, ypredictlabelall_np, average=None, zero_division=0)

    return correct, numberofdata, accuracy, losses, confusionmatrix, f1_micro, f1_macro, f1_weighted, f1_per_class

# Loss Function
loss_function = nn.BCEWithLogitsLoss().to(device)
learning_rate_init = 0.1

loss_function = ComboLoss(bce_weight=1.0, dice_weight=1.0).to(device)

optimizer = optimizer_setting(optim_name=optim_name, learning_rate=learning_rate_init)

# ==== Model Saving Variables ====
best_F1_macro = 0.0  # To keep track of the best F1 macro
save_dir = "/home/siamai/data/aj_art2/weight_saved/"
os.makedirs(save_dir, exist_ok=True)

num_epoch = 20

for epoch in range(num_epoch):
    start_time = time.time()
    print(f"\n-----------> START EPOCH {epoch + 1}/{num_epoch} <-----------")

    # Optionally update learning rate
    if epoch == 10:
        learning_rate = 0.01
        optimizer = optimizer_setting(optim_name=optim_name, learning_rate=learning_rate)
    if epoch == 50:
        learning_rate = 0.001
        optimizer = optimizer_setting(optim_name=optim_name, learning_rate=learning_rate)
    if epoch == 100:
        learning_rate = 0.0001
        optimizer = optimizer_setting(optim_name=optim_name, learning_rate=learning_rate)

    # Train
    Train_Num_of_Correct, Train_Num_of_Data, Train_Accuracy, Train_Losses, Train_ConfusionMatrix, Train_F1_micro, Train_F1_macro, Train_F1_weighted, Train_F1_per_class = traintest(
        model, train_dataloader, loss_function, device, optimizer, mode='train'
    )

    # Test
    Test_Num_of_Correct, Test_Num_of_Data, Test_Accuracy, Test_Losses, Test_ConfusionMatrix, Test_F1_micro, Test_F1_macro, Test_F1_weighted, Test_F1_per_class = traintest(
        model, val_dataloader, loss_function, device, optimizer, mode='test'
    )

    # ========== SAVE MODEL IF BEST ==========
    if Test_F1_macro > best_F1_macro:
        best_F1_macro = Test_F1_macro
        save_path = os.path.join(save_dir, f"best_model_epoch{epoch+1}_F1macro{Test_F1_macro:.4f}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"\n*** Model saved at {save_path} (Best F1_macro: {Test_F1_macro:.4f}) ***")

    print(f"\n\n-----------> Result of EPOCH {epoch + 1}/{num_epoch} <-----------\n")
    print(f'Training correct: {Train_Num_of_Correct} from {Train_Num_of_Data}')
    print(f'Training Loss: {np.mean(np.array(Train_Losses, dtype=np.float32)):.8f}')
    print(f'Training Accuracy: {Train_Accuracy:.8f}')
    print(f'Train F1 (micro): {Train_F1_micro:.4f} | F1 (macro): {Train_F1_macro:.4f} | F1 (weighted): {Train_F1_weighted:.4f}')
    print(f'Train F1 (per class): {Train_F1_per_class}')

    print(f'\nTesting correct: {Test_Num_of_Correct} from {Test_Num_of_Data}')
    print(f'Testing Loss: {np.mean(np.array(Test_Losses, dtype=np.float32)):.8f}')
    print(f'Testing Accuracy: {Test_Accuracy:.8f}')
    print(f'Test F1 (micro): {Test_F1_micro:.4f} | F1 (macro): {Test_F1_macro:.4f} | F1 (weighted): {Test_F1_weighted:.4f}')
    print(f'Test F1 (per class): {Test_F1_per_class}')

    print(f'Time for Training: {time.time() - start_time:.0f} seconds')
    print(f"\n-----------> END EPOCH {epoch + 1}/{num_epoch} <-----------")



"""
test_data_path = "data/test_data_joint.npy" # just for example
data_test_feeder = feeder_skeleton.Feeder(test_data_path, mean=mean, std=std)  # no label_path!
val_dataloader = DataLoader(data_test_feeder, batch_size=batch_size, shuffle=False, num_workers=4)

def inference(model, dataloader, device):
    model.eval()
    results = []
    with torch.no_grad():
        for X, names in tqdm(dataloader):
            X = X.to(device)
            ypredict = model(X)  # [B, num_class]
            ypredict_sigmoid = torch.sigmoid(ypredict)
            ypredictlabel = (ypredict_sigmoid > 0.5).int().cpu().numpy()
            ypredict_prob = ypredict_sigmoid.cpu().numpy()
            for name, ypred, yprob in zip(names, ypredictlabel, ypredict_prob):
                results.append({"dataname": name, "prediction": ypred, "prob": yprob})
    return results


# Inference
results = inference(model, val_dataloader, device)

# Example: Save as CSV
import pandas as pd
df = pd.DataFrame([
    {
        "dataname": r["dataname"],
        **{f"class_{i}": r["prediction"][i] for i in range(len(r["prediction"]))}
    }
    for r in results
])
df.to_csv("test_predictions.csv", index=False)


"""