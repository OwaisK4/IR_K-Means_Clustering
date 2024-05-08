from preprocessor import Preprocessor
import nltk
from math import log10

class VectorSpaceModel:
    """
    Vector Space Model for Information Retrieval.

    Attributes:
        files (list): List of file paths for documents.
        features (dict): Dictionary containing unique terms and their frequencies across all documents.
        filenames (list): List of document IDs.
        N (int): Number of documents.
        alpha (float): Cutoff value for returning relevant documents.
        feature_vector (list): Sorted list of unique terms.
        length (int): Length of feature vector.
        mapping (dict): Dictionary mapping terms to their indices in the feature vector.
        term_frequencies (list): Term frequencies for each document.
        doc_frequencies (list): Document frequencies for each term.
        euclidean (list): Euclidean lengths of document vectors.
    """
    def __init__(self, files: list[str]):
        self.files: list[str] = files
        self.features: dict[str, int] = {}
        self.filenames = [int(file.split("/")[-1].split(".")[0]) for file in self.files]
        self.N: int = len(self.filenames)    # No. of documents
        self.alpha: float = 0.03   # Cutoff for returning relevant docs

        # Calculate term frequencies across documents
        for file in files:
            preprocessor = Preprocessor(file)
            preprocessor.clean_tokens()
            for token in preprocessor.tokens:
                self.features.setdefault(token, 0)
                self.features[token] += 1
        
        self.feature_vector: list[str] = sorted(list(self.features.keys()))    # Feature vector for each doc and query
        self.length: int = len(self.feature_vector)
        self.mapping: dict[str, int] = {}       # Dictionary that converts feature into index for ease of use.
        for i in range(len(self.feature_vector)):
            self.mapping[self.feature_vector[i]] = i

        self.term_frequencies: list[list[float]] = [[0.0 for i in range(self.length)] for j in range(self.N)]
        self.doc_frequencies: list[int] = [0 for i in range(self.length)]
        self.euclidean: list[int] = [0 for i in range(self.N)]
        self.calculate_document_frequencies()
        self.calculate_tf_idf()
        self.calculate_euclidean_distances()

    def calculate_document_frequencies(self):
        """
        Calculates document frequencies for each term.
        """
        for file in self.files:
            current_doc_frequencies = [False for i in range(self.length)]
            preprocessor = Preprocessor(file)
            preprocessor.clean_tokens()
            for token in preprocessor.tokens:
                index = self.mapping[token]
                current_doc_frequencies[index] = True
            for i in range(len(current_doc_frequencies)):
                if current_doc_frequencies[i] is True:
                    self.doc_frequencies[i] += 1

    def calculate_tf_idf(self):
        """
        Calculates tf-idf for each document.
        """
        index = -1
        for file in self.files:
            # index = int(file.split("/")[-1].split(".")[0])      # Doc ID used as index
            index += 1
            preprocessor = Preprocessor(file)
            preprocessor.clean_tokens()
            for token in preprocessor.tokens:
                position = self.mapping[token]
                self.term_frequencies[index][position] += 1
            for i in range(len(self.term_frequencies[index])):
                self.term_frequencies[index][i] *= log10(len(self.files) / self.doc_frequencies[i])
    
    def calculate_euclidean_distances(self):
        """
        Calculates normalization values for each tf-idf document vector.
        """
        index = -1
        for file in self.files:
            # index = int(file.split("/")[-1].split(".")[0]) - 1      # Doc ID used as index
            index += 1
            for i in range(len(self.term_frequencies[index])):
                self.euclidean[index] += (self.term_frequencies[index][i] ** 2)
            self.euclidean[index] = self.euclidean[index] ** 0.5
        
    def calculate_cosine_similarity(self, query_vector: list[int]):
        """
        Calculates cosine similarity between the query vector and document vectors.

        Args:
            query_vector (list): Vector representing the query.

        Returns:
            list: List of relevant document IDs.
        """
        scores = [0.0 for i in range(self.N)]
        query_euclidean = 0
        for i in range(len(query_vector)):
            query_euclidean += (query_vector[i] ** 2)
        query_euclidean = query_euclidean ** 0.5

        index = -1
        for file in self.files:
            # index = int(file.split("/")[-1].split(".")[0]) - 1      # Doc ID used as index
            index += 1
            for j in range(len(self.term_frequencies[index])):
                scores[index] += (self.term_frequencies[index][j] * query_vector[j])
            denom = (self.euclidean[index] * query_euclidean)
            if denom != 0:
                scores[index] /= denom
        
        ranks = []
        for i in range(len(self.filenames)):
            ranks.append((scores[i], self.filenames[i]))
            # ranks.append((scores[self.filenames[i] - 1], self.filenames[i]))
        ranks.sort(reverse=True)
        print(ranks[:10])
        result = [str(rank[1]) for rank in ranks if rank[0] >= self.alpha]
        return result

    def process_query(self, query: str):
        """
        Processes a query and returns relevant documents.

        Args:
            query (str): Query string.

        Returns:
            list: List of relevant document IDs.
        """
        stemmer = nltk.PorterStemmer()
        words = query.split()
        for i in range(len(words)):
            words[i] = words[i].strip()
            words[i] = words[i].lower()
            words[i] = stemmer.stem(words[i])
        query_vector = [0 for i in range(self.length)]

        for word in words:
            index = self.mapping[word]
            query_vector[index] += 1
        return self.calculate_cosine_similarity(query_vector)