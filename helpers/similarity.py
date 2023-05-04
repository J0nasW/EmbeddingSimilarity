# This simple script should load the arXiv dataset and compute embeddings with SciNci BERT model. Then these embeddings alongside with the categories of the papers should be plotted in a 2D scatter plot with plotly using t-SNE.

import os
import numpy as np
import pandas as pd

from sklearn.manifold import TSNE
import plotly.express as px

from alive_progress import alive_bar

# HuggingFace YEAH!
from transformers import AutoTokenizer, AutoModel

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512" # If you have problems with CUDA memory allocation
import torch
torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Torch is using device: {device}")

# load model and tokenizer
checkpoint = "malteos/scincl"
# checkpoint = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint).to(device)

# SOME VARIABLES
BATCH_SIZE = 100


@torch.no_grad()
def generate_embeddings(abstracts, tokenizer, model, device):
    """Generate embeddings using BERT-based model.
    Code from Luca Schmidt.

    Parameters
    ----------
    abstracts : list
        Abstract texts.
    tokenizer : transformers.models.bert.tokenization_bert_fast.BertTokenizerFast
        Tokenizer.
    model : transformers.models.bert.modeling_bert.BertModel
        BERT-based model.
    device : str, {"cuda", "cpu"}
        "cuda" if torch.cuda.is_available() else "cpu".
        
    Returns
    -------
    embedding_cls : ndarray
        [CLS] tokens of the abstracts.
    embedding_sep : ndarray
        [SEP] tokens of the abstracts.
    embedding_av : ndarray
        Average of tokens of the abstracts.
    """
    # preprocess the input
    inputs = tokenizer(
        abstracts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512,
    ).to(device)

    # inference
    outputs = model(**inputs)[0].cpu().detach() 

    embedding_av = torch.mean(outputs, [0, 1]).numpy()
    embedding_sep = outputs[:, -1, :].numpy()
    embedding_cls = outputs[:, 0, :].numpy()

    
    return embedding_cls, embedding_sep, embedding_av

# https://github.com/malteos/scincl

with alive_bar(title="Loading arXiv dataset"):
    # Load the arxiv json as a pandas dataframe (for now limited to 10.000 papers)
    df = pd.read_json('arxiv-metadata-oai-snapshot-10000.json', lines=True)

# # Create a train and test set
# train = df.sample(frac=0.8, random_state=0)
# test = df.drop(train.index)

with alive_bar((int(np.round(len(df)/BATCH_SIZE))), title="Tokenizing and encoding texts") as bar:    
    # Tokenize and encode the texts in BATCH
    for i in range(0, len(df), BATCH_SIZE):
        df.loc[i:i+BATCH_SIZE]
        
        # Delete leading and trailing whitespaces
        df['title'] = df['title'].str.strip()
        df['abstract'] = df['abstract'].str.strip()

        # concatenate title and abstract with [SEP] token and save them in a list
        title_abs = df['title'].str.cat(df['abstract'], sep=' [SEP] ').tolist()
        
        # Embeddings
        embedding_cls, embedding_sep, embedding_av = generate_embeddings(title_abs, tokenizer, model, device)
        
        print(f"Embedding CLS shape: {embedding_cls.shape}")
        print(f"Embedding SEP shape: {embedding_sep.shape}")
        print(f"Embedding AV shape: {embedding_av.shape}")
        
        # Add embeddings to dataframe
        df['embedding_cls'] = embedding_cls.tolist()
        df['embedding_sep'] = embedding_sep.tolist()
        # df['embedding_av'] = embedding_av.tolist()
        
        # Append batch to dataframe
        df = df.concat(df, df.loc[i:i+BATCH_SIZE])
        torch.cuda.empty_cache()
        bar()
    
    print(df.head(10))