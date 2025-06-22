import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

class Feeder(Dataset):
    def __init__(self, npy_folder, label_path=None, label_position=[0,1,2,3,4,5,6,7,8,9,10,11]):
        self.npy_folder = npy_folder
        self.label_path = label_path
        self.npy_files = sorted([f for f in os.listdir(self.npy_folder) if f.endswith('.npy')])
        self.label_position = label_position
        self.load_data()

    def load_data(self):
        if self.label_path is not None and os.path.exists(self.label_path):
            df_labels = pd.read_pickle(self.label_path)  # Load DataFrame
            self.filenames = df_labels["Filename"].tolist()
            labels = df_labels.drop(columns=["Filename"]).values.tolist()
            self.labels = [[sublist[i] for i in self.label_position] for sublist in labels]
            # Keep only .npy files that match the DataFrame
            self.npy_files = [f for f in self.npy_files if f.replace(".npy", "") in self.filenames]

            # Match label list to file list
            label_map = {name: label for name, label in zip(self.filenames, self.labels)}
            self.labels = [label_map[f.replace(".npy", "")] for f in self.npy_files]

            assert len(self.npy_files) == len(self.labels), "Mismatch npy files and labels"
        else:
            self.labels = None

    def __len__(self):
        return len(self.npy_files)

    def __getitem__(self, idx):
        npy_file = self.npy_files[idx]
        data_numpy = np.load(os.path.join(self.npy_folder, npy_file))
        data_numpy = data_numpy[..., :2]   # Use only x, y
        data_tensor = torch.from_numpy(data_numpy).float() / 640 #data tensor can be adjusted(augment)
        # print(data_tensor.shape)
        # Use only x, y
        # data_numpy = data_numpy[..., :2]  # (T, 17, 2)

        # Define bones as list of joint index pairs: (from_joint, to_joint)
        # Example: COCO-style bones (adjust based on your skeleton layout)
        # bone_pairs = [
        #     (5, 7),  # 0 left shoulder to left elbow
        #     (7, 9),  # 1 left elbow to left wrist
        #     (6, 8),  # 2 right shoulder to right elbow
        #     (8,10),  # 3 right elbow to right wrist
        #     (11,13), # 4 left hip to left knee
        #     (13,15), # 5 left knee to left ankle
        #     (12,14), # 6 right hip to right knee
        #     (14,16), # 7 right knee to right ankle
        #     (5,6),   # 8 left shoulder to right shoulder
        #     (11,12), # 9 left hip to right hip
        #     (5,11),  # 10 left shoulder to left hip
        #     (6,12),  # 11 right shoulder to right hip
        # ]
        # # Compute bone vectors: shape will be (T, len(bone_pairs), 2)
        # bone_vectors = np.stack([
        #     data_numpy[:, j2] - data_numpy[:, j1]
        #     for j1, j2 in bone_pairs
        # ], axis=1)

        # data_tensor = torch.from_numpy(bone_vectors).float() / 640
        if self.labels is not None:
            label_tensor = torch.FloatTensor(self.labels[idx])
            return data_tensor, label_tensor, npy_file
        else:
            return data_tensor, npy_file