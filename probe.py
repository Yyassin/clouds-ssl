import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
import os
import torch
import csv

embed_root_dir = "./eval/embeddings"
embed_paths = os.listdir(os.path.abspath(embed_root_dir))
embed_paths = [f"{embed_root_dir}/{x}" for x in embed_paths]
embed_paths = [x for x in embed_paths if not os.path.isdir(x)]

# Cross-validation settings
n_splits_list = [293] # Leave one out

for n_splits in n_splits_list:
    for embed_path in embed_paths:
        torch_obj = torch.load(embed_path)
        embeds = torch_obj["embeds"].numpy()
        labels = torch_obj["labels"].numpy()

        kf = KFold(n_splits=n_splits)
        fold = 0

        all_linear = []
        all_knn_20 = []
        for train_index, test_index in kf.split(embeds):
            fold += 1

            X_train, X_test = embeds[train_index], embeds[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            
            # kNN with k=20
            knn_20_model = KNeighborsRegressor(n_neighbors=20, weights='distance', algorithm="brute")
            knn_20_model.fit(X_train, y_train)
            knn_20_y_pred = knn_20_model.predict(X_test)
            knn_20_y_pred_clipped = np.clip(knn_20_y_pred, 0, 1)
            knn_20_mae_per_class = np.mean(np.abs(knn_20_y_pred_clipped - y_test), axis=0).tolist()
            all_knn_20.append(knn_20_mae_per_class)
        
        all_knn_20 = torch.tensor(all_knn_20).mean(dim=0)
        all_knn_20_mean = torch.mean(all_knn_20).item()
        

        entry = [embed_path.split("/")[-1], n_splits] + ["k=20"] + all_knn_20.tolist() + [all_knn_20_mean]
        model_name = embed_path.split("/")[-1].split(".pt")[0]

        save_dir = "./eval/probe"
        if not os.path.exists(os.path.abspath(save_dir)):
            os.makedirs(os.path.abspath(save_dir))

        with open(os.path.abspath(fr'{save_dir}/{model_name}_loo_20.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow(entry)
            
        print(embed_path, n_splits)
