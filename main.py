import pandas as pd
import os
from helpers.similarity_flair import *

# --------------------------------------------------------------------------
# BiomedNLP
dataset_01 = {
    "PROJECT_DIR": "arXiv_sample_20000_BiomedNLP",
    "SOURCE_FILE": "data/arxiv-metadata-oai-snapshot.json",
    "DATAFIELDS": ['title', 'abstract'],
    "USE_SAMPLE": True,
    "SAMPLE_SIZE": 20000,
    "CHECKPOINT": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "MAX_LENGTH": 512,
    "SHOW_FIG": True,
    "GENERAL_CATEGORIES": True,
    "CLUSTER_EMBEDDINGS": True
}

# SciNCL
dataset_02 = {
    "PROJECT_DIR": "arXiv_sample_20000_SciNCL",
    "SOURCE_FILE": "data/arxiv-metadata-oai-snapshot.json",
    "DATAFIELDS": ['title', 'abstract'],
    "USE_SAMPLE": True,
    "SAMPLE_SIZE": 20000,
    "CHECKPOINT": "malteos/scincl",
    "MAX_LENGTH": 512,
    "SHOW_FIG": True,
    "GENERAL_CATEGORIES": True,
    "CLUSTER_EMBEDDINGS": True
}

# RoBERTa Base
dataset_03 = {
    "PROJECT_DIR": "arXiv_sample_20000_RoBERTa",
    "SOURCE_FILE": "data/arxiv-metadata-oai-snapshot.json",
    "DATAFIELDS": ['title', 'abstract'],
    "USE_SAMPLE": True,
    "SAMPLE_SIZE": 20000,
    "CHECKPOINT": "xlm-roberta-base",
    "MAX_LENGTH": 512,
    "SHOW_FIG": True,
    "GENERAL_CATEGORIES": True,
    "CLUSTER_EMBEDDINGS": True
}

# BigBird with Fulltext
dataset_04 = {
    "PROJECT_DIR": "arXiv_cs_AI_sample_20000_fulltext_BigBird",
    "SOURCE_FILE": "data/arxiv-fulltext-snapshot-02-23-cs-AI.json",
    "DATAFIELDS": ['text'],
    "USE_SAMPLE": True,
    "SAMPLE_SIZE": 20000,
    "CHECKPOINT": "google/bigbird-pegasus-large-arxiv",
    "MAX_LENGTH": 4096,
    "SHOW_FIG": True,
    "GENERAL_CATEGORIES": False,
    "CLUSTER_EMBEDDINGS": True
}

# BigBird with title and abstract
dataset_05 = {
    "PROJECT_DIR": "arXiv_cs_AI_sample_20000_BigBird",
    "SOURCE_FILE": "data/arxiv-metadata-oai-snapshot.json",
    "DATAFIELDS": ['title', 'abstract'],
    "USE_SAMPLE": True,
    "SAMPLE_SIZE": 20000,
    "CHECKPOINT": "google/bigbird-pegasus-large-arxiv",
    "MAX_LENGTH": 4096,
    "SHOW_FIG": True,
    "GENERAL_CATEGORIES": True,
    "CLUSTER_EMBEDDINGS": True
}

# SciNCL with individual categories
dataset_06 = {
    "PROJECT_DIR": "arXiv_sample_20000_SciNCL",
    "SOURCE_FILE": "data/arxiv-metadata-oai-snapshot.json",
    "DATAFIELDS": ['title', 'abstract'],
    "USE_SAMPLE": True,
    "SAMPLE_SIZE": 20000,
    "CHECKPOINT": "malteos/scincl",
    "MAX_LENGTH": 512,
    "SHOW_FIG": False,
    "GENERAL_CATEGORIES": False,
    "CLUSTER_EMBEDDINGS": True
}

# SciBERT cased
dataset_07 = {
    "PROJECT_DIR": "arXiv_sample_20000_SciBERT",
    "SOURCE_FILE": "data/arxiv-metadata-oai-snapshot.json",
    "DATAFIELDS": ['title', 'abstract'],
    "USE_SAMPLE": True,
    "SAMPLE_SIZE": 20000,
    "CHECKPOINT": "allenai/scibert_scivocab_cased",
    "MAX_LENGTH": 512,
    "SHOW_FIG": False,
    "GENERAL_CATEGORIES": True,
    "CLUSTER_EMBEDDINGS": True
}

# For testing purposes
test_dataset = {
    "PROJECT_DIR": "arXiv_test",
    "SOURCE_FILE": "data/arxiv-metadata-oai-snapshot-10000.json",
    "DATAFIELDS": ['title', 'abstract'],
    "USE_SAMPLE": True,
    "SAMPLE_SIZE": 1000,
    "CHECKPOINT": "malteos/scincl",
    "MAX_LENGTH": 512,
    "SHOW_FIG": True,
    "GENERAL_CATEGORIES": True,
    "CLUSTER_EMBEDDINGS": True
}


# dataset_pipeline = [
#     dataset_01,
#     dataset_02,
#     dataset_03,
#     dataset_04,
#     dataset_05,
#     dataset_06,
#     dataset_07,
# ]

dataset_pipeline = [dataset_02]
# --------------------------------------------------------------------------


# Create the Embeddings Class

for dataset in dataset_pipeline:
    ACTIVE_DATASET = dataset
    
    try:
        arxiv_embeddings = EMBEDDING_BOT(
            source_file = ACTIVE_DATASET["SOURCE_FILE"],
            project_dir = ACTIVE_DATASET["PROJECT_DIR"],
            checkpoint = ACTIVE_DATASET["CHECKPOINT"],
            datafields = ACTIVE_DATASET["DATAFIELDS"],
            max_length = ACTIVE_DATASET["MAX_LENGTH"]
        )

        # Load the dataset
        arxiv_embeddings.load_dataset(
            sample = ACTIVE_DATASET["USE_SAMPLE"],
            sample_size = ACTIVE_DATASET["SAMPLE_SIZE"]
        )

        # Create the embeddings
        arxiv_embeddings.create_embeddings()

        # Reduce the dimensionality
        arxiv_embeddings.reduce_dimensionality()

        # Plot the embeddings
        arxiv_embeddings.plot_embeddings(
            show_fig=ACTIVE_DATASET["SHOW_FIG"],
            sample=False,
            general_categories=ACTIVE_DATASET["GENERAL_CATEGORIES"],
            cluster_embeddings=ACTIVE_DATASET["CLUSTER_EMBEDDINGS"]
        )

        # Save the dataset
        arxiv_embeddings.save_dataset()

    
    except Exception as e:
        print("Error in dataset: ", ACTIVE_DATASET["PROJECT_DIR"])
        print(e)
        continue
    