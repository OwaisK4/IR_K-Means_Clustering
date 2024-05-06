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
    if os.path.exists(saved_index):
        with open(saved_index, "rb") as f:
            model = pickle.load(f)
        print("Loaded vector space model from file")
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
    classifier = KNeighborsClassifier(n_neighbors=4)
    classifier.fit(X_train, y_train)

    y_true = y_train.copy()
    y_pred = classifier.predict(X_test)
    print(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy = {accuracy}")

    recall = recall_score(y_true, y_pred, average=None)
    print(f"Recall = {recall}")

    precision = precision_score(y_true, y_pred, average=None)
    print(f"Precision = {precision}")
    
    f1 = f1_score(y_true, y_pred, average=None)
    print(f"F1 score = {f1}")

    # PART 2:
    clusters = 5
    kmeans = KMeans(n_clusters=clusters, n_init=5, max_iter=500, random_state=42) 
    kmeans.fit(X_train)
    
    labels = kmeans.labels_
    print(labels)

    homogeneity = homogeneity_score(y_true, y_pred)
    print(f"Homogeneity = {homogeneity}")

    random_index = rand_score(y_true, y_pred)
    print(f"Random Index = {random_index}")

    silhouette = silhouette_score(X_train, kmeans.fit_predict(X_train))
    print(f"Silhouette = {silhouette}")