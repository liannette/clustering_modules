from pathlib import Path
import argparse
import sys


def main():
    print("Command-line:", " ".join(sys.argv))

    args = parse_arguments()
    clusters_file = Path(args.clusters_file)
    background_counts_file = Path(args.background_counts_file)

    with open(clusters_file, "r") as infile:
        clusters = get_tokenised_genes_per_cluster(infile.readlines())
    tokenised_genes = sorted(get_all_tokenised_genes(clusters))
    bg_count = get_gene_background_count(tokenised_genes, clusters)

    with open(background_counts_file, "w") as outfile:
        outfile.write(f"#Total\t{len(clusters)}\n")
        outfile.writelines(
            f"{gene}\t{bg_count[gene]}\n" for gene in tokenised_genes
        )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="A script to cluster the statistical modules into families 
        "with different algorithms."
    )
    parser.add_argument(
        "-i",
        "--infile",
        help="Input tsv file with module information (stat_modules.txt). A header is present, one module per line where a module is the last element in the line. Genes separated by ',' and domains by ';'",
        required=True,
    )
    parser.add_argument(
        "-c", "--cores", help="Cores to use (default = 1)", default=1, type=int
    )
    parser.add_argument(
        "-k",
        "--k_clusters",
        help="Amount of clusters to use with k-means clustering",
        type=int,
        required=True,
    )
    return parser.parse_args()

    parser = argparse.ArgumentParser(
        description="Counts the background occurence of each tokenised gene."
    )
    parser.add_argument(
        "--clusters",
        dest="clusters_file",
        required=True,
        metavar="<file>",
        help="The filtered clusterfile from iPRESTO.",
    )
    parser.add_argument(
        "--outfile",
        dest="background_counts_file",
        required=True,
        metavar="<dir>",
        help="Output file containing the background counts of all tokenised genes.",
    )
    return parser.parse_args()


def get_tokenised_genes_per_cluster(clusterfile_lines):
    gene_clusters = dict()
    for line in clusterfile_lines:
        name, tokenized_genes = line.rstrip().split(",", 1)
        tokenized_genes = set(tokenized_genes.split(","))
        tokenized_genes.discard("-")  # genes without biosynthetic domains
        gene_clusters[name] = tokenized_genes
    return gene_clusters


def get_all_tokenised_genes(gene_clusters):
    tokenised_genes = set().union(*gene_clusters.values())
    return tokenised_genes


def get_gene_background_count(tokenised_genes, gene_clusters):
    background_count = dict()
    for gene in tokenised_genes:
        background_count[gene] = count_bgcs_containing_gene(gene_clusters, gene)
    return background_count


def count_bgcs_containing_gene(gene_clusters, gene):
    count = 0
    for bgc_genes in gene_clusters.values():
        if gene in bgc_genes:
            count += 1
    return count


if __name__ == "__main__":
    main()


#!/usr/bin/env python3

import os

# make sure that numpy only uses one thread
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
import time
import csv


def get_commands():
    parser = argparse.ArgumentParser(
        description="A script to cluster the "
        "statistical modules into families with different algorithms."
    )
    parser.add_argument(
        "-i",
        "--infile",
        help="Input tsv "
        "file with module information (stat_modules.txt). A header is "
        "present, one module per line where a module is the last element in "
        "the line. Genes separated by ',' and domains by ';'",
        required=True,
    )
    parser.add_argument(
        "-c", "--cores", help="Cores to use, default = 1", default=1, type=int
    )
    parser.add_argument(
        "-k",
        "--k_clusters",
        help="Amount of clusters to use with k-means clustering",
        type=int,
        required=True,
    )
    return parser.parse_args()


class StatModule:
    def __init__(
        self,
        module_id,
        strictest_pval,
        tokenised_genes,
    ):
        self.module_id = module_id
        self.n_genes = len(tokenised_genes)
        self.n_domains = sum(len(gene) for gene in tokenised_genes)
        self.tokenised_genes = tokenised_genes
        self.strictest_pval = strictest_pval

    def __repr__(self):
        return (
            f"StatModule("
            f"module_id={self.module_id}, "
            f"strictest_pval={self.strictest_pval}, "
            f"n_genes={self.n_genes}, "
            f"n_domains={self.n_domains}, "
            f"tokenised_genes={self.tokenised_genes})"
        )


def string_to_tokenized_genes(genes_string):
    """
    Converts a string representation of tokenized genes back into a list of tuples.

    Parameters:
        genes_string (str): A string representation of tokenized genes, where each gene is
            separated by a comma and each tuple is separated by a semicolon.

    Returns:
        list of tuples: A list of tokenized genes, where each gene is represented as a tuple of tuples.
    """
    return [tuple(gene.split(";")) for gene in genes_string.split(",")]


def read_stat_modules(file_path):
    """
    Reads statistical modules from a tab-separated file into a list of StatModules.

    Parameters:
        file_path (str): The file path to the input file.

    Returns:
        dict: A dictionary where keys are module IDs and values are StatModule objects.
    """
    with open(file_path, "r", newline="") as infile:
        reader = csv.DictReader(infile, delimiter="\t")
        modules = {}
        for row in reader:
            module_id = row["module_id"]
            strictest_pval = float(row["strictest_pval"])
            tokenised_genes = string_to_tokenized_genes(row["tokenised_genes"])
            if module_id in modules:
                print(
                    f"Warning: Duplicate module ID {module_id}. Keeping the first one."
                )
                continue
            module = StatModule(
                module_id=module_id,
                strictest_pval=strictest_pval,
                tokenised_genes=tokenised_genes,
            )
            modules[module_id] = module
        print(f"Read {len(modules)} STAT modules.")
        return modules


def cluster_kmeans(
    sparse_m, modules, num_clusters, rownames, colnames, prefix, header, cores=1
):
    """Kmeans clustering on sparse_m with num_clusters and writes to file

    sparse_m: csr_matrix, shape(n_samples, n_features)
    modules: dict {mod_num:[info,modules]}
    num_clusters: int, number of clusters
    rownames: list of ints, [mod_nums], sequential mod_nums, keeping track of
        rows of sparse_m
    colnames: list of str, all domains sequential order to keep track of
        columns of sparse_m
    prefix: str, prefix of outfile
    cores: int, amount of cores to use
    header: str, header of module file
    """
    print("\nRunning k-means")
    # outfiles
    kmeans_pre = f"_{num_clusters}_families"
    out_mods = prefix + kmeans_pre + ".txt"
    out_clusts = prefix + kmeans_pre + "_by_family.txt"

    # running algorithm
    kmeans = KMeans(
        n_clusters=num_clusters,
        n_init=20,
        max_iter=1000,
        random_state=595,
        verbose=0,
        tol=0.000001,
        n_jobs=cores,
    ).fit(sparse_m)
    print(kmeans)
    clust_centers = sp.csr_matrix(kmeans.cluster_centers_)
    labels = kmeans.labels_
    cluster_dict = defaultdict(list)
    np.set_printoptions(precision=2)
    print("Within-cluster sum-of-squares (inertia):", kmeans.inertia_)
    # link each module to a family/k-means cluster
    with open(out_mods, "w") as outf:
        outf.write(header + "\tFamily\n")
        for subcl, cl in zip(rownames, labels):
            cluster_dict[cl].append(subcl)
            outf.write(
                "{}\t{}\t{}\n".format(subcl, "\t".join(modules[subcl]), cl)
            )
    # write file of listing all families/clusters by family with their modules
    avg_clst_size = []
    with open(out_clusts, "w") as outf_c:
        for i in range(clust_centers.shape[0]):
            matches = cluster_dict[i]
            l_matches = len(matches)
            avg_clst_size.append(l_matches)
            counts = Counter(
                [dom for m in matches for dom in modules[m][-1].split(",")]
            )
            spars = clust_centers[i]
            feat_inds = spars.nonzero()[1]
            feat_tups = [(spars[0, ind], colnames[ind]) for ind in feat_inds]
            feat_format = [
                "{}:{:.2f}".format(dom, float(score))
                for score, dom in sorted(feat_tups, reverse=True)
            ]
            outf_c.write(
                "#Subcluster-family {}, {} subclusters\n".format(i, l_matches)
            )
            outf_c.write(
                "#Occurrences: {}\n".format(
                    ", ".join(
                        [dom + ":" + str(c) for dom, c in counts.most_common()]
                    )
                )
            )
            outf_c.write("#Features: {}\n".format(", ".join(feat_format)))
            # maybe as a score the distance to the cluster center?
            for match in matches:
                outf_c.write(
                    "{}\t{}\n".format(match, "\t".join(modules[match]))
                )
    print("\nAverage clustersize:", np.mean(avg_clst_size))


def plot_svd_components(sparse_m):
    """Plots first two components of truncatedSVD analysis (PCA)

    sparse_m: scipy_sparse_matrix
    """
    svd = TruncatedSVD(n_components=2, n_iter=7, random_state=595)
    components = svd.fit_transform(sparse_m)
    # print(components)
    print(svd.explained_variance_ratio_)
    x, y = zip(*components)
    plt.scatter(x, y)
    plt.show()


if __name__ == "__main__":
    cmd = get_commands()

    # outfiles
    out_prefix = cmd.infile.split(".txt")[0]

    # construct feature matrix
    modules = {}  # keep track of info
    rownames = []  # in case mod_nums are not sequential
    corpus = []  # list of strings
    vectorizer = CountVectorizer(
        lowercase=False, binary=True, dtype=np.int32, token_pattern=r"(?u)[^,]+"
    )  # finds everything separated by ','

    with open(cmd.infile, "r") as inf:
        print("\nReading module file")
        # {mod_num:[info]}
        header = inf.readline().strip("\n")  # header
        for line in inf:
            line = line.strip().split("\t")
            mod_num = int(line[0])
            modules[mod_num] = line[1:]
            rownames.append(mod_num)
            corpus.append(line[-1])
    print("\nBuilding sparse matrix representation of module features")
    sparse_feat_matrix = vectorizer.fit_transform(corpus)
    colnames = vectorizer.get_feature_names()
    print("  {} features".format(len(colnames)))

    cluster_kmeans(
        sparse_feat_matrix,
        modules,
        cmd.k_clusters,
        rownames,
        colnames,
        out_prefix,
        header=header,
        cores=cmd.cores,
    )

    end = time.time()
    t = end - start
    t_str = "{}h{}m{}s".format(
        int(t / 3600), int(t % 3600 / 60), int(t % 3600 % 60)
    )
    print("\nScript completed in {}".format(t_str))
