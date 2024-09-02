import os
from torchvision import transforms
import torch
from PIL import Image
import csv
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

def get_labels():
    """Get the labels from the annotations.csv file."""
    data = []
    with open(os.path.abspath('./annotations.csv'), 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            entry = []
            for i, elem in enumerate(row):
                if i == 0:
                    entry.append(elem)
                else:
                    entry.append(float(elem))
            data.append(entry)

    return data

def get_all(size):
    """Collect labeled images, and their annotations.
    
    Args:
        size (int): The size to resize the images to.
    """
    all_data = []
    data = get_labels()
    for item in data:
        image = Image.open(os.path.abspath(f"./annotated_images/{item[0]}"))
        tensor = transforms.ToTensor()(transforms.Resize(size)(image))
        annotation = item[1:]
        all_data.append([tensor, annotation])
    
    return all_data


def get_dataset(model, is_moco, size):
    """Get embeddings and labels for annotated images.
    
    Args:
        model: The model to use.
        is_moco (bool): Whether the model is MoCo (otherwise it's JEPA).
        size (int): The size to resize the images
    """
    all_data = get_all(size)
    all_embeds = []
    all_labels = []
    with torch.no_grad():
        for row in all_data:
            image, annotation = row
            image = image.unsqueeze(dim=0).to(model.device)

            if is_moco:
                embeds = model.encoder_q.forward_features(image)
            else:
                embeds = model.student_encoder.forward_features(image)
            embeds = embeds[:, 1:, :].mean(dim=1)  # global average pool
            all_embeds.append(embeds.cpu())
            all_labels.append(torch.tensor(annotation).unsqueeze(dim=0))
    
    all_embeds = torch.cat(all_embeds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_embeds, all_labels

def knn(model, is_moco=False, size=384):
    """Compute the mean absolute error using k-nearest neighbors.
    
    Args:
        model: The model to use.
        is_moco (bool): Whether the model is MoCo (otherwise it's JEPA).
        size (int): The size to resize the images
    """
    all_embeds, all_labels = get_dataset(model, is_moco, size)

    embeds = all_embeds.numpy()
    labels = all_labels.numpy()
    kf = KFold(n_splits=293) # Leave one out

    all_knn_5 = []
    for train_index, test_index in kf.split(embeds):
        X_train, X_test = embeds[train_index], embeds[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        knn_5_model = KNeighborsRegressor(n_neighbors=20, weights='distance', algorithm="brute")
        knn_5_model.fit(X_train, y_train)
        knn_5_y_pred = knn_5_model.predict(X_test)
        knn_5_y_pred_clipped = np.clip(knn_5_y_pred, 0, 1)
        knn_5_mae_per_class = np.mean(np.abs(knn_5_y_pred_clipped - y_test), axis=0).tolist()
        all_knn_5.append(knn_5_mae_per_class)

    all_knn_5 = torch.tensor(all_knn_5).mean(dim=0)
    all_knn_5_mean = torch.mean(all_knn_5).item()

    return all_knn_5_mean
