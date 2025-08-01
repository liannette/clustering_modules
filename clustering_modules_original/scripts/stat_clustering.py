import pickle

from scipy.sparse import lil_matrix
from sklearn.cluster import AgglomerativeClustering


def load_stat_subclusters(stat_subclusters_file_path):
    stat_subclusters = list()
    with open(stat_subclusters_file_path, "r") as infile:
        infile.readline()  # skip header
        for line in infile:
            subcluster = set(line.split("\t")[5].split(","))
            stat_subclusters.append(subcluster)
    return stat_subclusters


def jaccard_similarity(list1, list2):
    intersection = len(set(list1) & set(list2))
    union = len(set(list1) | set(list2))
    return intersection / union


def build_similarity_matrix(stat_subclusters):
    n = len(stat_subclusters)
    similarity_matrix = lil_matrix((n, n))
    for i in range(n):
        for j in range(n):
            similarity_matrix[i, j] = jaccard_similarity(
                stat_subclusters[i], stat_subclusters[j]
            )
    return similarity_matrix


def main():
    # # Load the subclusters
    # stat_subclusters_file_path = "../data/PRESTO-STAT_subclusters.txt"
    # stat_subclusters = load_stat_subclusters(stat_subclusters_file_path)

    # # Build a similarity matrix
    # similarity_matrix = build_similarity_matrix(stat_subclusters)

    # with open("similarity_matrix.pkl", "wb") as f:
    #     pickle.dump(similarity_matrix, f)

    # with open("similarity_matrix.pkl", "rb") as f:
    #     similarity_matrix = pickle.load(f)

    # # Convert to distance matrix for clustering
    # distance_matrix = 1 - similarity_matrix.toarray()

    # with open("distance_matrix.pkl", "wb") as f:
    #     pickle.dump(distance_matrix, f)

    with open("../results/distance_matrix.pkl", "rb") as f:
        distance_matrix = pickle.load(f)

    # Apply clustering
    cluster_model = AgglomerativeClustering(metric="precomputed", linkage="average")
    labels = cluster_model.fit_predict(distance_matrix)

    with open("../results/cluster_model.pkl", "wb") as f:
        pickle.dump(cluster_model, f)
    with open("../results/labels.pkl", "wb") as f:
        pickle.dump(labels, f)


if __name__ == "__main__":
    main()
