import argparse
import sys

from pathlib import Path
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", dest="input_file", required=True, metavar="<file>", help=""
    )
    parser.add_argument(
        "-k", dest="num_clusters", required=True, type=int, metavar="<int>", help=""
    )
    parser.add_argument(
        "--output_file", dest="output_file", required=True, metavar="<file>", help=""
    )
    return parser.parse_args()


def get_tokenised_genes_from_match_line(match_line):
    tokenised_genes = list()
    match = match_line.split()[2]
    for gene_and_topic_prob in match.split(","):
        gene, _ = gene_and_topic_prob.split(":")
        tokenised_genes.append(gene)
    return tokenised_genes


def get_module_matches(input_file):
    module_matches = defaultdict(list)
    with open(input_file, "r") as infile:
        for line in infile:
            # skip header lines
            if line.startswith("#"):
                continue
            tokenised_genes = get_tokenised_genes_from_match_line(line)
            if len(tokenised_genes) < 2:
                # skip matches with less than 2 (unique) genes
                continue
            bgc_id = line.split()[3]
            module_matches[tuple(tokenised_genes)].append(bgc_id)
    return module_matches


def kmeans_clustering(modules, n_clusters):
    """
    Perform KMeans clustering on the subcluster modules.
    """
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(modules)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)

    return kmeans.labels_


def main():
    print("Command-line:", " ".join(sys.argv))

    args = parse_arguments()
    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    num_clusters = args.num_clusters

    module_matches = get_module_matches(input_file)
    top_modules = list(module_matches.keys())

    labels = kmeans_clustering(top_modules, n_clusters=num_clusters)

    modules_per_label = defaultdict(list)
    for module, label in zip(top_modules, labels):
        modules_per_label[label].append(module)

    with open(output_file, "w") as f:
        for label in sorted(modules_per_label.keys()):
            modules = modules_per_label[label]
            f.write(f"#model: {label}, matches: {len(modules)}\n")
            for module in sorted(modules):
                bgc_ids = module_matches[module]
                for bgc_id in bgc_ids:
                    f.write(f"{bgc_id}\t{','.join(module)}\n")
