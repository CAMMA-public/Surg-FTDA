# import os
# import sys
# import argparse
# import pickle
# import math
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.cluster import KMeans
# from sklearn.model_selection import train_test_split
# from torch.utils.data import TensorDataset, DataLoader


# class MLP(nn.Module):
#     """
#     A simple MLP for modality alignment.
#     """
#     def __init__(self, input_dim, output_dim, hidden_dim=128):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, output_dim)
    
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.fc3(x)
#         return x


# def load_data(image_path, text_path):
#     """
#     Loads image and text embeddings from the specified pickle files.

#     Args:
#         image_path (str): Path to the image embedding pickle file.
#         text_path (str): Path to the text embedding pickle file.

#     Returns:
#         (np.ndarray, np.ndarray, dict, dict): 
#             image_features, text_features, original_image_dict, original_text_dict
#     """
#     with open(image_path, 'rb') as f:
#         image_data = pickle.load(f)
#     with open(text_path, 'rb') as f:
#         text_data = pickle.load(f)

#     image_features = image_data["clip_embedding_text_dave"].cpu().numpy()
#     text_features = text_data["clip_embedding_text_dave"].cpu().numpy()
#     return image_features, text_features, image_data, text_data


# def kmeans_representatives(features, n_clusters=500):
#     """
#     Applies K-Means clustering and returns representative indices.

#     Args:
#         features (np.ndarray): Feature array of shape (N, D).
#         n_clusters (int): Number of clusters for K-means.

#     Returns:
#         list: Indices of the representative points.
#     """
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     kmeans.fit(features)

#     representative_indices = []
#     for i in range(n_clusters):
#         # Get indices of points in cluster i
#         cluster_indices = np.where(kmeans.labels_ == i)[0]
#         cluster_center = kmeans.cluster_centers_[i]
#         # Find the point closest to the cluster center
#         closest_index = cluster_indices[
#             np.argmin(
#                 np.linalg.norm(features[cluster_indices] - cluster_center, axis=1)
#             )
#         ]
#         representative_indices.append(closest_index)
#     return representative_indices


# def farthest_point_sampling(data, num_samples):
#     """
#     Perform Farthest Point Sampling (FPS) on the given data.

#     Parameters:
#     - data (numpy.ndarray): The data points, shape (N, D)
#     - num_samples (int): Number of points to sample

#     Returns:
#     - list: Indices of the sampled points
#     """
#     N, D = data.shape
#     selected_indices = []
#     distances = np.full(N, np.inf)  # Initialize distances to infinity

#     # 1. Randomly select the first point
#     first_index = np.random.randint(0, N)
#     selected_indices.append(first_index)
#     distances[first_index] = 0  # Distance to itself is zero

#     # 2. Iteratively select the farthest point from the existing set
#     for _ in range(1, num_samples):
#         # Update distances: keep the minimum distance to any selected point
#         last_added = data[selected_indices[-1], :]
#         dist_to_last_added = np.linalg.norm(data - last_added, axis=1)
#         distances = np.minimum(distances, dist_to_last_added)

#         # Select the point that has the maximum distance to the selected set
#         next_index = np.argmax(distances)
#         selected_indices.append(next_index)
#         distances[next_index] = 0  # Distance to itself is zero

#     return selected_indices


# def fps_representatives(features, n_clusters=500):
#     """
#     Uses Farthest Point Sampling (FPS) to get representative indices.

#     Args:
#         features (np.ndarray): Feature array of shape (N, D).
#         n_clusters (int): Number of points (representatives) to sample.

#     Returns:
#         list: Indices of the representative points.
#     """
#     # 调用 farthest_point_sampling 获取索引
#     rep_indices = farthest_point_sampling(features, n_clusters)
#     return rep_indices


# def train_mlp(image_rep_points, text_rep_points, hidden_dim, lr, n_epochs, batch_size):
#     """
#     Trains an MLP model to align image features to text feature space.

#     Args:
#         image_rep_points (np.ndarray): Representative image features (N, D_in).
#         text_rep_points (np.ndarray): Representative text features (N, D_out).
#         hidden_dim (int): Hidden layer dimension.
#         lr (float): Learning rate.
#         n_epochs (int): Number of training epochs.
#         batch_size (int): Batch size.

#     Returns:
#         nn.Module: Trained MLP model.
#     """
#     # Convert numpy arrays to torch tensors
#     X_train_all = torch.tensor(image_rep_points, dtype=torch.float32)
#     y_train_all = torch.tensor(text_rep_points, dtype=torch.float32)

#     # Split data into train/val
#     X_train, X_val, y_train, y_val = train_test_split(
#         X_train_all, y_train_all, test_size=0.1, random_state=42
#     )

#     # Create the model, loss and optimizer
#     input_dim = image_rep_points.shape[1]
#     output_dim = text_rep_points.shape[1]
#     model = MLP(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim)
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr)

#     # DataLoader for training
#     train_dataset = TensorDataset(X_train, y_train)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#     # Training loop
#     for epoch in range(n_epochs):
#         model.train()
#         running_loss = 0.0
#         for inputs, targets in train_loader:
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()

#         # Print loss every 10 epochs
#         if (epoch + 1) % 10 == 0:
#             avg_loss = running_loss / len(train_loader)
#             print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.6f}")

#     # Validation
#     model.eval()
#     with torch.no_grad():
#         val_outputs = model(X_val)
#         val_loss = criterion(val_outputs, y_val)
#     print(f"Validation Loss: {val_loss.item():.6f}")

#     return model


# def align_features(model, features):
#     """
#     Uses the trained MLP model to align features.

#     Args:
#         model (nn.Module): Trained MLP model.
#         features (np.ndarray): Image features to be aligned.

#     Returns:
#         np.ndarray: Aligned features in text space.
#     """
#     model.eval()
#     with torch.no_grad():
#         x_tensor = torch.tensor(features, dtype=torch.float32)
#         aligned = model(x_tensor).cpu().numpy()
#     return aligned


# def main(args):
#     # 1. Load training data
#     print("Loading training data...")
#     image_features_train, text_features_train, image_data_train, _ = load_data(
#         args.image_train_original, args.text_train
#     )
#     print(f"Image features (train): {image_features_train.shape}")
#     print(f"Text features (train): {text_features_train.shape}")

#     # 2. 根据 sampling_method 选择 K-Means 或者 FPS 来选取代表点
#     if args.sampling_method == "kmeans":
#         print(f"Performing K-Means with n_clusters={args.n_clusters}...")
#         rep_indices = kmeans_representatives(
#             image_features_train, n_clusters=args.n_clusters
#         )
#     elif args.sampling_method == "fps":
#         print(f"Performing Farthest Point Sampling with n_clusters={args.n_clusters}...")
#         rep_indices = fps_representatives(
#             image_features_train, n_clusters=args.n_clusters
#         )
#     else:
#         raise ValueError("sampling_method must be one of ['kmeans', 'fps'].")

#     image_rep_points = image_features_train[rep_indices]
#     text_rep_points = text_features_train[rep_indices]

#     # 3. Train the MLP
#     print(
#         f"Training MLP with hidden_dim={args.hidden_dim}, "
#         f"lr={args.lr}, epochs={args.n_epochs}, batch_size={args.batch_size}..."
#     )
#     model = train_mlp(
#         image_rep_points,
#         text_rep_points,
#         hidden_dim=args.hidden_dim,
#         lr=args.lr,
#         n_epochs=args.n_epochs,
#         batch_size=args.batch_size,
#     )

#     # 4. Align all image features (train)
#     print("Aligning all training image features...")
#     aligned_image_features_train = align_features(model, image_features_train)

#     # 5. Load validation data
#     print("Loading validation data...")
#     image_features_val, text_features_val, image_data_val, _ = load_data(
#         args.image_val_original, args.text_val
#     )
#     print(f"Image features (val): {image_features_val.shape}")
#     print(f"Text features (val): {text_features_val.shape}")

#     # 6. Align all image features (val)
#     print("Aligning all validation image features...")
#     aligned_image_features_val = align_features(model, image_features_val)

#     # 7. Prepare output directory if it does not exist
#     out_dir = os.path.dirname(args.out_path)
#     if out_dir and not os.path.exists(out_dir):
#         os.makedirs(out_dir)

#     # 8. Save aligned embeddings for validation
#     print(f"Saving aligned embeddings to {args.out_path}...")
#     # Keep other keys from image_data_val, just replace 'clip_embedding_text_dave'
#     image_data_val["clip_embedding_text_dave"] = torch.from_numpy(aligned_image_features_val)
#     with open(args.out_path, 'wb') as f:
#         pickle.dump(
#             {
#                 "clip_embedding": image_data_val["clip_embedding"],
#                 "captions": image_data_val["captions"],
#                 "clip_embedding_text_dave": image_data_val["clip_embedding_text_dave"]
#             },
#             f
#         )
#     print("Done!")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Modality fusion script using K-Means or FPS and an MLP for alignment."
#     )
#     # Default paths from the original code
#     parser.add_argument(
#         "--image_train_original",
#         type=str,
#         default="/home/ubuntu/meddataset/meddatasetnew/train_test/embedding_step/embedding_nipsv3_v2/embedding_image_train.pkl",
#         help="Path to the training image embedding pickle file."
#     )
#     parser.add_argument(
#         "--text_train",
#         type=str,
#         default="/home/ubuntu/meddataset/meddatasetnew/train_test/embedding_step/embedding_nipsv3_v2/embedding_text_train.pkl",
#         help="Path to the training text embedding pickle file."
#     )
#     parser.add_argument(
#         "--image_val_original",
#         type=str,
#         default="/home/ubuntu/meddataset/meddatasetnew/val/embedding_text_v2/nips3/image_text_original.pkl",
#         help="Path to the validation image embedding pickle file."
#     )
#     parser.add_argument(
#         "--text_val",
#         type=str,
#         default="/home/ubuntu/meddataset/meddatasetnew/val/embedding_text_v2/nips3/embedding_text_val.pkl",
#         help="Path to the validation text embedding pickle file."
#     )
#     parser.add_argument(
#         "--out_path",
#         type=str,
#         default="/home/ubuntu/meddataset/meddatasetnew/val/embedding_text_v2/nips3/surgvlp_val_image_embedding_align_500_image_v2_test_v3.pkl",
#         help="Path to save the aligned validation image embeddings."
#     )

#     # Default hyperparameters
#     parser.add_argument(
#         "--n_clusters",
#         type=int,
#         default=500,
#         help="Number of clusters for K-Means or number of samples for FPS."
#     )
#     parser.add_argument(
#         "--hidden_dim",
#         type=int,
#         default=128,
#         help="Hidden dimension of the MLP."
#     )
#     parser.add_argument(
#         "--lr",
#         type=float,
#         default=0.001,
#         help="Learning rate for MLP training."
#     )
#     parser.add_argument(
#         "--n_epochs",
#         type=int,
#         default=15,
#         help="Number of epochs for MLP training."
#     )
#     parser.add_argument(
#         "--batch_size",
#         type=int,
#         default=16,
#         help="Batch size for MLP training."
#     )


#     parser.add_argument(
#         "--sampling_method",
#         type=str,
#         default="fps",
#         choices=["kmeans", "fps"],
#         help="Sampling method to use for representative points. Options: 'kmeans' or 'fps'."
#     )

#     args = parser.parse_args()
#     main(args)

import os
import sys
import argparse
import pickle
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

class MLP(nn.Module):
    """
    A simple MLP for modality alignment.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def load_data(image_path, text_path):
    """
    Loads image and text embeddings from the specified pickle files.

    Args:
        image_path (str): Path to the image embedding pickle file.
        text_path (str): Path to the text embedding pickle file.

    Returns:
        (np.ndarray, np.ndarray, dict, dict): 
            image_features, text_features, original_image_dict, original_text_dict
    """
    with open(image_path, 'rb') as f:
        image_data = pickle.load(f)
    with open(text_path, 'rb') as f:
        text_data = pickle.load(f)

    image_features = image_data["clip_embedding_text_dave"].cpu().numpy()
    text_features = text_data["clip_embedding_text_dave"].cpu().numpy()
    return image_features, text_features, image_data, text_data

def kmeans_representatives(features, n_clusters=500):
    """
    Applies K-Means clustering and returns representative indices.

    Args:
        features (np.ndarray): Feature array of shape (N, D).
        n_clusters (int): Number of clusters for K-means.

    Returns:
        list: Indices of the representative points.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(features)

    # Calculate clustering metrics
    labels = kmeans.labels_
    silhouette = silhouette_score(features, labels)
    calinski_harabasz = calinski_harabasz_score(features, labels)
    davies_bouldin = davies_bouldin_score(features, labels)

    print(f"Clustering Metrics for KMeans (n_clusters={n_clusters}):")
    print(f" - Silhouette Score: {silhouette:.4f}")
    print(f" - Calinski-Harabasz Score: {calinski_harabasz:.4f}")
    print(f" - Davies-Bouldin Index: {davies_bouldin:.4f}")

    representative_indices = []
    for i in range(n_clusters):
        # Get indices of points in cluster i
        cluster_indices = np.where(kmeans.labels_ == i)[0]
        cluster_center = kmeans.cluster_centers_[i]
        # Find the point closest to the cluster center
        closest_index = cluster_indices[
            np.argmin(
                np.linalg.norm(features[cluster_indices] - cluster_center, axis=1)
            )
        ]
        representative_indices.append(closest_index)
    return representative_indices

def farthest_point_sampling(data, num_samples):
    """
    Perform Farthest Point Sampling (FPS) on the given data.

    Parameters:
    - data (numpy.ndarray): The data points, shape (N, D)
    - num_samples (int): Number of points to sample

    Returns:
    - list: Indices of the sampled points
    """
    N, D = data.shape
    selected_indices = []
    distances = np.full(N, np.inf)  # Initialize distances to infinity

    # 1. Randomly select the first point
    first_index = np.random.randint(0, N)
    selected_indices.append(first_index)
    distances[first_index] = 0  # Distance to itself is zero

    # 2. Iteratively select the farthest point from the existing set
    for _ in range(1, num_samples):
        # Update distances: keep the minimum distance to any selected point
        last_added = data[selected_indices[-1], :]
        dist_to_last_added = np.linalg.norm(data - last_added, axis=1)
        distances = np.minimum(distances, dist_to_last_added)

        # Select the point that has the maximum distance to the selected set
        next_index = np.argmax(distances)
        selected_indices.append(next_index)
        distances[next_index] = 0  # Distance to itself is zero

    return selected_indices

def fps_representatives(features, n_clusters=500):
    """
    Uses Farthest Point Sampling (FPS) to get representative indices.

    Args:
        features (np.ndarray): Feature array of shape (N, D).
        n_clusters (int): Number of points (representatives) to sample.

    Returns:
        list: Indices of the representative points.
    """
    rep_indices = farthest_point_sampling(features, n_clusters)
    return rep_indices

def train_mlp(image_rep_points, text_rep_points, hidden_dim, lr, n_epochs, batch_size):
    """
    Trains an MLP model to align image features to text feature space.

    Args:
        image_rep_points (np.ndarray): Representative image features (N, D_in).
        text_rep_points (np.ndarray): Representative text features (N, D_out).
        hidden_dim (int): Hidden layer dimension.
        lr (float): Learning rate.
        n_epochs (int): Number of training epochs.
        batch_size (int): Batch size.

    Returns:
        nn.Module: Trained MLP model.
    """
    # Convert numpy arrays to torch tensors
    X_train_all = torch.tensor(image_rep_points, dtype=torch.float32)
    y_train_all = torch.tensor(text_rep_points, dtype=torch.float32)

    # Split data into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all, y_train_all, test_size=0.1, random_state=42
    )

    # Create the model, loss and optimizer
    input_dim = image_rep_points.shape[1]
    output_dim = text_rep_points.shape[1]
    model = MLP(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # DataLoader for training
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Print loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            avg_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.6f}")

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)
    print(f"Validation Loss: {val_loss.item():.6f}")

    return model

def align_features(model, features):
    """
    Uses the trained MLP model to align features.

    Args:
        model (nn.Module): Trained MLP model.
        features (np.ndarray): Image features to be aligned.

    Returns:
        np.ndarray: Aligned features in text space.
    """
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(features, dtype=torch.float32)
        aligned = model(x_tensor).cpu().numpy()
    return aligned

def main(args):
    # 1. Load training data
    print("Loading training data...")
    image_features_train, text_features_train, image_data_train, _ = load_data(
        args.image_train_original, args.text_train
    )
    print(f"Image features (train): {image_features_train.shape}")
    print(f"Text features (train): {text_features_train.shape}")

    if args.sampling_method == "kmeans":
        print(f"Performing K-Means with n_clusters={args.n_clusters}...")
        rep_indices = kmeans_representatives(
            image_features_train, n_clusters=args.n_clusters
        )
    elif args.sampling_method == "fps":
        print(f"Performing Farthest Point Sampling with n_clusters={args.n_clusters}...")
        rep_indices = fps_representatives(
            image_features_train, n_clusters=args.n_clusters
        )
    else:
        raise ValueError("sampling_method must be one of ['kmeans', 'fps'].")

    image_rep_points = image_features_train[rep_indices]
    text_rep_points = text_features_train[rep_indices]

    # 3. Train the MLP
    print(
        f"Training MLP with hidden_dim={args.hidden_dim}, "
        f"lr={args.lr}, epochs={args.n_epochs}, batch_size={args.batch_size}..."
    )
    model = train_mlp(
        image_rep_points,
        text_rep_points,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
    )

    # 4. Align all image features (train)
    print("Aligning all training image features...")
    aligned_image_features_train = align_features(model, image_features_train)

    # 5. Load validation data
    print("Loading validation data...")
    image_features_val, text_features_val, image_data_val, _ = load_data(
        args.image_val_original, args.text_val
    )
    print(f"Image features (val): {image_features_val.shape}")
    print(f"Text features (val): {text_features_val.shape}")

    # 6. Align all image features (val)
    print("Aligning all validation image features...")
    aligned_image_features_val = align_features(model, image_features_val)

    # 7. Prepare output directory if it does not exist
    out_dir = os.path.dirname(args.out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 8. Save aligned embeddings for validation
    print(f"Saving aligned embeddings to {args.out_path}...")
    # Keep other keys from image_data_val, just replace 'clip_embedding_text_dave'
    image_data_val["clip_embedding_text_dave"] = torch.from_numpy(aligned_image_features_val)
    with open(args.out_path, 'wb') as f:
        pickle.dump(
            {
                "clip_embedding": image_data_val["clip_embedding"],
                "captions": image_data_val["captions"],
                "clip_embedding_text_dave": image_data_val["clip_embedding_text_dave"]
            },
            f
        )
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Modality fusion script using K-Means or FPS and an MLP for alignment."
    )
    # Default paths from the original code
    parser.add_argument(
        "--image_train_original",
        type=str,
        default="/home/ubuntu/meddataset/cholec_process/triplet_train/v2/embedding/nips3/image_embedding_triplet_original_train.pkl",
        help="Path to the training image embedding pickle file."
    )
    parser.add_argument(
        "--text_train",
        type=str,
        default="/home/ubuntu/meddataset/cholec_process/triplet_train/v2/embedding/nips3/tiplet_text_train.pkl",
        help="Path to the training text embedding pickle file."
    )
    parser.add_argument(
        "--image_val_original",
        type=str,
        default="/home/ubuntu/meddataset/cholec_process/triplet_test/embedding_nips3/image_embedding_triplet_original_test.pkl",
        help="Path to the validation image embedding pickle file."
    )
    parser.add_argument(
        "--text_val",
        type=str,
        default="/home/ubuntu/meddataset/cholec_process/triplet_test/embedding_nips3/text_embedding_tiplet_test.pkl",
        help="Path to the validation text embedding pickle file."
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="/home/ubuntu/meddataset/meddatasetnew/val_test_for_kmeans_metrics/embedding_triplet/nips3/surgvlp_val_image_embedding_align_500_image_v2_test.pkl",
        help="Path to save the aligned validation image embeddings."
    )

    # Default hyperparameters
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=500,
        help="Number of clusters for K-Means or number of samples for FPS."
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension of the MLP."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for MLP training."
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=15,
        help="Number of epochs for MLP training."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for MLP training."
    )

    parser.add_argument(
        "--sampling_method",
        type=str,
        default="kmeans",
        choices=["kmeans", "fps"],
        help="Sampling method to use for representative points. Options: 'kmeans' or 'fps'."
    )

    args = parser.parse_args()
    main(args)
