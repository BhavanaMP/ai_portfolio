import warnings
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
from Feature_extraction import FeatureExtractor
from scipy.sparse import vstack
import matplotlib.pyplot as plt
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from active_semi_clustering.semi_supervised.pairwise_constraints import PCKMeans
from gensim.test.utils import get_tmpfile
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import os
import pickle


warnings.filterwarnings('ignore')


def unsupervised_PCK(input_vector, k, ml=[], cl=[]):
    print("Unsupervised training using PCK Means")
    print("Required clusters : {}".format(k))
    clusterer = PCKMeans(n_clusters=k)
    clusterer.fit(input_vector, ml=ml, cl=cl)
    # Plotting the graphs
    plotgraph(input_vector, clusterer)
    return clusterer


# Get the similiar and disimilar docs by sorting based on similarity values
def sort_on_similarity(initial_constraints):
    return sorted(
        initial_constraints, key=lambda tup: tup[1][1], reverse=True
    )


# Extracting instances/docs after sorting based on similarity
def extract_docs(ml_sim_sorted):
    li = []
    for item in ml_sim_sorted:
        li.append(((item[0], item[1][0])))
    return li


# Sorting each tuple inside the list to follow ascend order to find duplicates
def inner_tuple_sort(extracted_docs):
    sort_tuples = []
    for element in extracted_docs:
        sort_tuples.append(tuple(sorted(element)))
    return sort_tuples


# Getting unique tuples preserving the order of tuples
def unique(sequence):
    seen = set()
    return [x for x in sequence
            if not (tuple(x) in seen or seen.add(tuple(x)))]


# Retrieve initial pairwise constraints
def get_init_constraints(tokenized_lemm_articles, pretrained_model):
    ml_init_constraints = []
    cl_init_constraints = []
    for doc_id in range(len(tokenized_lemm_articles)):
        # Compare and print the most-similar, disimilar document
        sim_doc = pretrained_model.docvecs.most_similar(
            doc_id, topn=len(pretrained_model.docvecs))[0]
        if sim_doc[1] >= 0.8:
            ml_init_constraints += [[doc_id, sim_doc]]
        dissim_doc = pretrained_model.docvecs.most_similar(
            doc_id, topn=len(pretrained_model.docvecs))[-1]
        if dissim_doc[1] <= 0.1:
            cl_init_constraints += [[doc_id, dissim_doc]]
    return ml_init_constraints, cl_init_constraints


# Generating Pairwise constraints
def generate_constraints(input_data):
    print("Generating pairwise constraints")
    tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(input_data)]
    pretrained_doc2vec_model = Doc2Vec(
        tagged_data, window=2, vector_size=256,
        min_count=3, workers=4, epochs=100
    )
    # Initialize N
    N = 50
    ml_init_constraints, cl_init_constraints =\
        get_init_constraints(input_data, pretrained_doc2vec_model)

    # Perform auxiliary transformations on initial ML constraints
    ml_sim_sorted = sort_on_similarity(ml_init_constraints)
    extracted_ml_docs = extract_docs(ml_sim_sorted)
    sorted_ml_tuples = inner_tuple_sort(extracted_ml_docs)

    # Perform auxiliary transformations on initial CL constraints
    cl_sim_sorted = sort_on_similarity(cl_init_constraints)
    extracted_cl_docs = extract_docs(cl_sim_sorted)
    sorted_cl_tuples = inner_tuple_sort(extracted_cl_docs)

    final_ml_constraints = unique(sorted_ml_tuples)[:N]
    final_cl_constraints = unique(sorted_cl_tuples)[:N]

    return final_ml_constraints, final_cl_constraints, pretrained_doc2vec_model


def plotgraph(input, clusterer):
    fig, (ax2) = plt.subplots(1)
    fig.set_size_inches(10, 10)
    no_of_clusters = 6
    cluster_labels = clusterer.labels_
    silhouette_avg = silhouette_score(input, cluster_labels)
    print(
        "For n_clusters =", no_of_clusters,
        "The average silhouette_score is :", silhouette_avg,
    )

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / no_of_clusters)
    ax2.scatter(
        input[:, 0], input[:, 1],
        marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )
    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0], centers[:, 1], marker="o",
        c="white", alpha=1, s=400, edgecolor="k",
    )
    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=14, edgecolor="k")
    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")
    plt.suptitle(
        "Silhouette analysis for MbkMeans clustering on"
        "sample data with n_clusters = %d"
        % no_of_clusters, fontsize=no_of_clusters, fontweight="bold",
    )
    plt.show()


def explainability(doc, clusterer, doc2vec_model, key_phrases):
    rootdir = os.getcwd()
    key_phrases_pkl = os.path.join(rootdir, 'keyphrase.pkl')
    with open(key_phrases_pkl, 'rb') as f:
        key_phrases = pickle.load(f)
    print("************************************************")
    print("Target document is {}".format(doc))
    doc_phrases = key_phrases[doc]
    doc_phrase = [i[0] for i in doc_phrases]
    print("Key phrases of this document are: {}".format(doc_phrase))
    print("The most similar documents are as follows:")
    similar_docs = doc2vec_model.docvecs.most_similar(doc, topn=3)
    for i in range(len(similar_docs)):
        print(
            "Document:"+str(similar_docs[i][0]) + " Similarity Score:"+str(
                similar_docs[i][1]
            ))
    print("************************************************")
    print("The chosen document belongs to cluster:"+str(clusterer.labels_[doc]))
    print("************************************************")
    for k in similar_docs:
        doc_number = k[0]
        key_phrases_per_doc = key_phrases[doc_number]
        print("Key phrases for document "+str(k[0])+":")
        for kp in key_phrases_per_doc:
            print(kp[0])
        print("This document belongs to cluster: "+str(
            clusterer.labels_[doc_number]
        ))
        print("************************************************")


if __name__ == '__main__':
    import argparse
    from distutils.util import strtobool

    parser = argparse.ArgumentParser(
        description='Running PCK Means Clustering Algorithm')
    parser.add_argument('--datadir', metavar='path', required=True,
                        help='the path to data files')
    parser.add_argument('--is_new_data', default=False, required=False,
                        type=lambda x: bool(strtobool(str(x))),
                        help='set this to True if the data is new')
    parser.add_argument('--k', type=int, default=6,
                        required=False, help="Required Clusters",)
    parser.add_argument('--datasample', type=int, required=False,
                        help='Find the cluster label of the sample')
    parser.add_argument('--type', type=str, required=False,
                        help='Give the type of clustering type.'
                        'stack/w2v/lda/tfidf')
    args = parser.parse_args()

    if ('datasample' in vars(args) and 'type' not in vars(args)):
        parser.error(
            'The -datasample argument requires the -type.'
            'Please give stack/ w2v/ lda/ tfidf'
        )
    if args.type and args.type.lower() not in ('w2v', 'lda', 'stack', 'tfidf'):
        parser.error(
            'The -datasample argument requires the -type.'
            'Please give stack/ w2v/ lda/ tfidf'
        )

    # Downloading dataset and Feature extraction
    extractor = FeatureExtractor()
    transformed_vector = \
        extractor.extract_features(
            args.datadir, is_new_data=args.is_new_data,
        )
    w2v_vector = extractor.w2v_vector
    lda_vector = extractor.lda_vector
    tfidf_vector = extractor.tfdif_vector
    input_lemmatized_data = extractor.dataset
    key_phrases = extractor.key_phrases

    # Generating Pairwise constraints
    ml_constraints, cl_constraints, pretrained_doc2vec_model =\
        generate_constraints(input_lemmatized_data)

    # Performing PCK Means with constraints
    stack_clusterer = unsupervised_PCK(transformed_vector, args.k,
                                       ml_constraints, cl_constraints)
    word2vec_clusterer = unsupervised_PCK(w2v_vector,  args.k,
                                          ml_constraints, cl_constraints,)
    lda_clusterer = unsupervised_PCK(lda_vector, args.k,
                                     ml_constraints, cl_constraints)
    tfidf_clusterer = unsupervised_PCK(tfidf_vector,  args.k,
                                       ml_constraints, cl_constraints,)

    # Performing PCK Means without constraints
    stack_clusterer = unsupervised_PCK(transformed_vector, args.k)
    word2vec_clusterer = unsupervised_PCK(w2v_vector, args.k)
    lda_clusterer = unsupervised_PCK(lda_vector, args.k)
    tfidf_clusterer = unsupervised_PCK(tfidf_vector, args.k)
  
    # Explainability
    if args.datasample and args.type:
       
        if args.type.lower() == 'w2v':
            explainability(
                args.datasample, word2vec_clusterer,
                pretrained_doc2vec_model, key_phrases
            )
        if args.type.lower() == 'lda':
            explainability(
                args.datasample, lda_clusterer,
                pretrained_doc2vec_model, key_phrases
            )
        if args.type.lower() == 'stack':
            explainability(
                args.datasample, stack_clusterer,
                pretrained_doc2vec_model, key_phrases
            )
        if args.type.lower() == 'tfidf':
            explainability(
                args.datasample, tfidf_clusterer,
                pretrained_doc2vec_model, key_phrases
            )
