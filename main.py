import os
import pickle
from indexer import VectorSpaceModel
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, homogeneity_score, rand_score, silhouette_samples, silhouette_score
# from sklearn.metrics.pairwise import cosine_similarity

filepath = "/home/owaisk4/Win_backup/FAST NU assignments/Information Retrieval/Assignment 3/ResearchPapers"
saved_index = os.path.join(filepath, "vector_space_index.pkl")

if __name__ == "__main__":
    model: VectorSpaceModel

    # If saved index already exists in path, simply load it from file.
    if os.path.exists(saved_index):
        with open(saved_index, "rb") as f:
            model = pickle.load(f)
        print("Loaded vector space model from file")
    # Else, create it from scratch
    else:
        files = os.listdir(filepath)
        files = [os.path.join(filepath, file) for file in files]    
        model = VectorSpaceModel(files)
        print("Created vector space model from scratch")
        with open(saved_index, "wb") as f:
            pickle.dump(model, f)
    
    # query = "machine learning"
    # result = model.process_query(query)
    # if len(result) == 0:
    #     print("NIL")
    # else:
    #     print(result)

    # PART 1:

    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(model.term_frequencies)
    X_train = model.term_frequencies
    X_test = X_train.copy()
    y_train = ["Explainable Artificial Intelligence", "Explainable Artificial Intelligence", "Explainable Artificial Intelligence",  "Explainable Artificial Intelligence", "Heart Failure", "Heart Failure", "Heart Failure", "Time Series Forecasting", "Time Series Forecasting", "Time Series Forecasting", "Time Series Forecasting", "Time Series Forecasting", "Transformer Model", "Transformer Model", "Transformer Model", "Feature Selection", "Feature Selection", "Feature Selection", "Feature Selection", "Feature Selection"]

    # Classification
    # Instantiate the KNeighborsClassifier with 4 neighbors
    classifier = KNeighborsClassifier(n_neighbors=4)
    # Train the classifier on the training data
    classifier.fit(X_train, y_train)

    # Make predictions on the test data
    y_true = y_train.copy()
    y_pred = classifier.predict(X_test)
    print(y_pred)

    # Compute accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy = {accuracy}")

    # Compute recall
    recall = recall_score(y_true, y_pred, average=None)
    print(f"Recall = {recall}")

    # Compute precision
    precision = precision_score(y_true, y_pred, average=None)
    print(f"Precision = {precision}")
    
    # Compute F1 score
    f1 = f1_score(y_true, y_pred, average=None)
    print(f"F1 score = {f1}")

    # PART 2:

    # Clustering
    clusters = 5
    # Instantiate the KMeans clustering algorithm with 5 clusters
    kmeans = KMeans(n_clusters=clusters, n_init=5, max_iter=500, random_state=42)
    # Fit the KMeans model to the training data
    kmeans.fit(X_train)
    
    # Get the cluster labels for the training data
    labels = kmeans.labels_
    print(labels)

    # Compute homogeneity score
    # Homogeneity score measures the extent to which each cluster contains only members of a single class
    homogeneity = homogeneity_score(y_true, y_pred)
    print(f"Homogeneity = {homogeneity}")

    # Compute random index
    # Random Index is a measure of similarity between two data clusterings.
    # It counts the number of pairs of documents that are either in the same cluster in both the true and predicted clusterings, 
    # or in different clusters in both the true and predicted clusterings.
    # It then divides this count by the total number of document pairs.
    random_index = rand_score(y_true, y_pred)
    print(f"Random Index = {random_index}")

    # Compute silhouette score
    # Silhouette score measures how similar an object is to its own cluster compared to other clusters
    silhouette = silhouette_score(X_train, kmeans.fit_predict(X_train))
    print(f"Silhouette = {silhouette}")