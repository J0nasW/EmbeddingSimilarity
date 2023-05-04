# This simple script should load the arXiv dataset and compute embeddings with SciNci BERT model. Then these embeddings alongside with the categories of the papers should be plotted in a 2D scatter plot with plotly using t-SNE.

import numpy as np
import pandas as pd
import os

from sklearn.manifold import TSNE
import hdbscan

import plotly.express as px
import plotly.graph_objects as go
import matplotlib as plt

from alive_progress import alive_bar

from flair.embeddings import TransformerDocumentEmbeddings
from flair.data import Sentence

from termcolor import colored

import torch
torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Torch is using device: {device}")

# Dont show warnings
import warnings
warnings.filterwarnings("ignore")

class EMBEDDING_BOT():
    '''
    Class to create embeddings with a specific model.
    '''
    def __init__(self, source_file, project_dir, checkpoint, datafields, max_length=512, use_3d=False):
        '''
        Initializing the embedding bot.
        INPUTS:
            checkpoint: string with the path to the model checkpoint (Huggingface)
            datafields: list of strings that represent columns in the dataframe
            max_length: integer with the maximum token count of the chosen model (typically 512)
        OUTPUTS:
            None
        '''
        print(colored("Initializing embedding bot...", "green"))
        self.checkpoint = checkpoint
        self.datafields = datafields
        self.max_length = max_length
        self.source_file = source_file
        self.project_dir = project_dir
        self.use_3d = use_3d
        print(colored(f"Project Name: {self.project_dir}", "yellow"))
        self.embedding = TransformerDocumentEmbeddings(checkpoint)
        print(colored(f"Embedding using checkpoint: {self.checkpoint}", "yellow"))
        self.clusterer = hdbscan.HDBSCAN(
            algorithm='best',
            alpha=1.0,
            approx_min_span_tree=True,
            gen_min_span_tree=False,
            leaf_size=40,
            metric='euclidean',
            min_cluster_size=5,
            min_samples=None,
            p=None
        )
        self.df = None
        self.embeddings = None
        self.mean_df = None
        self.target_file = ""
        self.file_format = ""
        self.device = self.check_torch_device()
        self.init_project_dir()
        self.color_list = ["#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
        "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
        "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
        "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
        "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
        "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
        "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
        "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
        "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
        "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
        "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
        "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
        "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
        "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
        "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
        "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58",
        "#7A7BFF", "#D68E01", "#353339", "#78AFA1", "#FEB2C6", "#75797C", "#837393", "#943A4D",
        "#B5F4FF", "#D2DCD5", "#9556BD", "#6A714A", "#001325", "#02525F", "#0AA3F7", "#E98176",
        "#DBD5DD", "#5EBCD1", "#3D4F44", "#7E6405", "#02684E", "#962B75", "#8D8546", "#9695C5",
        "#E773CE", "#D86A78", "#3E89BE", "#CA834E", "#518A87", "#5B113C", "#55813B", "#E704C4",
        "#00005F", "#A97399", "#4B8160", "#59738A", "#FF5DA7", "#F7C9BF", "#643127", "#513A01",
        "#6B94AA", "#51A058", "#A45B02", "#1D1702", "#E20027", "#E7AB63", "#4C6001", "#9C6966",
        "#64547B", "#97979E", "#006A66", "#391406", "#F4D749", "#0045D2", "#006C31", "#DDB6D0",
        "#7C6571", "#9FB2A4", "#00D891", "#15A08A", "#BC65E9", "#FFFFFE", "#C6DC99", "#203B3C",
        "#671190", "#6B3A64", "#F5E1FF", "#FFA0F2", "#CCAA35", "#374527", "#8BB400", "#797868",
        "#C6005A", "#3B000A", "#C86240", "#29607C", "#402334", "#7D5A44", "#CCB87C", "#B88183",
        "#AA5199", "#B5D6C3", "#A38469", "#9F94F0", "#A74571", "#B894A6", "#71BB8C", "#00B433",
        "#789EC9", "#6D80BA", "#953F00", "#5EFF03", "#E4FFFC", "#1BE177", "#BCB1E5", "#76912F",
        "#003109", "#0060CD", "#D20096", "#895563", "#29201D", "#5B3213", "#A76F42", "#89412E",
        "#1A3A2A", "#494B5A", "#A88C85", "#F4ABAA", "#A3F3AB", "#00C6C8", "#EA8B66", "#958A9F",
        "#BDC9D2", "#9FA064", "#BE4700", "#658188", "#83A485", "#453C23", "#47675D", "#3A3F00",
        "#061203", "#DFFB71", "#868E7E", "#98D058", "#6C8F7D", "#D7BFC2", "#3C3E6E", "#D83D66",
        "#2F5D9B", "#6C5E46", "#D25B88", "#5B656C", "#00B57F", "#545C46", "#866097", "#365D25",
        "#252F99", "#00CCFF", "#674E60", "#FC009C", "#92896B"]
        self.arxiv_cs_taxonomy = {
            "cs.AI": "Artificial Intelligence",
            "cs.AR": "Hardware Architecture",
            "cs.CC": "Computational Complexity",
            "cs.CE": "Computational Engineering, Finance, and Science",
            "cs.CG": "Computational Geometry",
            "cs.CL": "Computation and Language",
            "cs.CR": "Cryptography and Security",
            "cs.CV": "Computer Vision and Pattern Recognition",
            "cs.CY": "Computers and Society",
            "cs.DB": "Databases",
            "cs.DC": "Distributed, Parallel, and Cluster Computing",
            "cs.DL": "Digital Libraries",
            "cs.DM": "Discrete Mathematics",
            "cs.DS": "Data Structures and Algorithms",
            "cs.ET": "Emerging Technologies",
            "cs.FL": "Formal Languages and Automata Theory",
            "cs.GL": "General Literature",
            "cs.GR": "Graphics",
            "cs.GT": "Computer Science and Game Theory",
            "cs.HC": "Human-Computer Interaction",
            "cs.IR": "Information Retrieval",
            "cs.IT": "Information Theory",
            "cs.LG": "Machine Learning",
            "cs.LO": "Logic in Computer Science",
            "cs.MA": "Multiagent Systems",
            "cs.MM": "Multimedia",
            "cs.MS": "Mathematical Software",
            "cs.NA": "Numerical Analysis",
            "cs.NE": "Neural and Evolutionary Computing",
            "cs.NI": "Networking and Internet Architecture",
            "cs.OH": "Other Computer Science",
            "cs.OS": "Operating Systems",
            "cs.PF": "Performance",
            "cs.PL": "Programming Languages",
            "cs.RO": "Robotics",
            "cs.SC": "Symbolic Computation",
            "cs.SD": "Sound",
            "cs.SE": "Software Engineering",
            "cs.SI": "Social and Information Networks",
            "cs.SY": "Systems and Control",
        }
        print(colored("✔️ Embedding bot initialized.", "green"))
        
    def check_torch_device(self):
        '''
        Checking if torch is using the GPU or CPU.
        INPUTS:
            None
        OUTPUTS:
            device: string with the device torch is using
        '''
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if mps_available:
            device = "mps"
        print(colored(f"Torch is using device: {device}.", "yellow"))
        return device
    
    def init_project_dir(self):
        '''
        Initializing the project directory.
        INPUTS:
            None
        OUTPUTS:
            None
        '''
        # Create directory if it does not exist
        if not os.path.exists("projects"):
            os.makedirs("projects")
            
        # Create project directory if it does not exist
        if not os.path.exists(os.path.join("projects", self.project_dir)):
            os.makedirs(os.path.join("projects", self.project_dir))
        # Update project dir file path
        self.project_dir = os.path.join("projects", self.project_dir)
        
    def load_dataset(self, sample=False, sample_size=10000):
        '''
        Loading a dataset from a json or csv file.
        INPUTS:
            file: string with the path to the file
            sample: boolean to sample the dataset
            sample_size: integer with the number of samples
        OUTPUTS:
            df: pandas dataframe with the dataset
        '''
        with alive_bar(title="Loading dataset"):
            # Check if file exists
            if not os.path.exists(self.source_file):
                raise ValueError(f"File {self.source_file} does not exist")
            # Check the filetype for json or csv
            if self.source_file.endswith(".json"):
                df = pd.read_json(self.source_file, lines=True)
                self.file_format = "json"
            elif self.source_file.endswith(".csv"):
                df = pd.read_csv(self.source_file)
                self.file_format = "csv"
            else:
                raise ValueError(f"File {self.source_file} is not a json or csv file")
            if sample:
                df = df.sample(sample_size)
        self.df = df
        return df

    def save_dataset(self):
        '''
        Saving a dataset to a json file.
        INPUTS:
            df: pandas dataframe with the dataset
            file: string with the path to the file
        OUTPUTS:
            None
        '''
        try:
            with alive_bar(title="Saving dataset"):
                # Create filenames with source file names in project dir
                embedding_filename = os.path.join(self.project_dir, "embedding_" + os.path.basename(self.source_file))
                embedding_mean_filename = os.path.join(self.project_dir, "mean_embedding_" + os.path.basename(self.source_file))
                # Check if file exists
                if os.path.exists(embedding_filename) or os.path.exists(embedding_mean_filename):
                    print(f"File already exists, overwriting it.")
                # Check the filetype for json or csv
                if self.file_format == "json":
                    self.df.to_json(embedding_filename, orient="records", lines=True)
                    self.mean_df.to_json(embedding_mean_filename, orient="records", lines=True)
                    print(f"Saving to {embedding_filename} and {embedding_mean_filename}.")
                elif self.file_format == "csv":
                    self.df.to_csv(embedding_filename, index=False)
                    self.mean_df.to_csv(embedding_mean_filename, index=False)
                    print(f"Saving to {embedding_filename} and {embedding_mean_filename}.")
                else:
                    raise ValueError(f"File {self.target_file} is not a json or csv file")
            print(colored(f"✔️ Dataset saved to {self.project_dir}.", "green"))
        except Exception as e:
            print(colored(f"❌ Error saving dataset: {e}", "red"))
            return None

    def create_embeddings(self):
        '''
        Creating embeddings with a specific model.
        INPUTS:
            df: pandas dataframe
            datafields: list of strings that represent columns in the dataframe
        OUTPUTS:
            df: pandas dataframe with embeddings
            embeddings: numpy array with embeddings
        '''
        try:
            # Check if datafields is valid
            if not isinstance(self.datafields, list):
                raise ValueError("Datafields must be a list of strings")
            # Check if datafields are in the dataframe and if they have the same length
            for field in self.datafields:
                if field not in self.df.columns:
                    raise ValueError(f"Field {field} not in dataframe")
                if len(self.df[field]) != len(self.df):
                    raise ValueError(f"Field {field} has different length than dataframe")

            with alive_bar(len(self.df), title="Creating embeddings") as bar:
                # Delete leading and trailing whitespaces
                for field in self.datafields:
                    self.df[field] = self.df[field].str.strip()

                # Embed the datafields. If there are more than one, concatenate them with " [SEP] ". Also check for max_length
                embeddings = []
                for i in range(len(self.df)):
                    # Create a sentence object
                    sentence = Sentence(
                        " [SEP] ".join(
                            self.df[self.datafields].iloc[i].values
                        )[:self.max_length]
                    )
                    # Embed the sentence
                    self.embedding.embed(sentence)
                    # Append the embedding to the list
                    embeddings.append(sentence.embedding.cpu().detach().numpy())
                    bar()

                # Add embeddings to dataframe
                self.df['embedding'] = embeddings
                # Convert the embeddings to numpy arrays
                embeddings = np.array(embeddings)

                print(f"Embedding shape: {embeddings[0].shape}")
                # print(df.head(10))
            self.embeddings = embeddings
            return self.df, embeddings
        
        except Exception as e:
            print(e)
            return None, None

    def reduce_dimensionality(self):
        '''
        Reducing the dimensionality of the embeddings with t-SNE.
        INPUTS:
            embeddings: numpy array with embeddings
        OUTPUTS:
            embeddings: numpy array with embeddings
        '''
        try:
            # Use t-SNE to reduce the dimensionality of the embeddings and write them into the pandas df
            with alive_bar(title="Reducing dimensionality of embeddings"):
                if self.use_3d:
                    embeddings_tsne = TSNE(
                        n_components=3,
                        perplexity=30,
                        random_state=42
                    ).fit_transform(
                        self.embeddings.reshape(
                            len(self.embeddings), -1
                        )
                    )
                    self.df['embedding_tsne_x'] = embeddings_tsne[:, 0]
                    self.df['embedding_tsne_y'] = embeddings_tsne[:, 1]
                    self.df['embedding_tsne_z'] = embeddings_tsne[:, 2]
                else:
                    embeddings_tsne = TSNE(
                        n_components=2,
                        perplexity=30,
                        random_state=42
                    ).fit_transform(
                        self.embeddings.reshape(
                            len(self.embeddings), -1
                        )
                    )
                    self.df['embedding_tsne_x'] = embeddings_tsne[:, 0]
                    self.df['embedding_tsne_y'] = embeddings_tsne[:, 1]
                
                # print(f"Embedding t-SNE shape: {embeddings_tsne[0].shape}")
                
                # Separate the categories " " and write them into a list
                self.df['categories'] = self.df['categories'].str.split(' ').tolist()
                
                # Create duplicate rows for each category
                self.df = self.df.explode('categories')
                
                # Calculate the mean of the embeddings for each category
                self.mean_df = self.df.groupby('categories').mean().reset_index()
                
                # print("Mean embeddings for each category: ")
                # print(self.mean_df.head(10))
                
                # Calculate a vector from the mean embedding to the embedding x and y of each paper
                if self.use_3d:
                    self.df['vector'] = self.df.apply(lambda x: np.array([x['embedding_tsne_x'] - self.mean_df[self.mean_df['categories'] == x['categories']]['embedding_tsne_x'].values[0], x['embedding_tsne_y'] - self.mean_df[self.mean_df['categories'] == x['categories']]['embedding_tsne_y'].values[0], x['embedding_tsne_z'] - self.mean_df[self.mean_df['categories'] == x['categories']]['embedding_tsne_z'].values[0]]), axis=1)
                else:
                    self.df['vector'] = self.df.apply(lambda x: np.array([x['embedding_tsne_x'] - self.mean_df[self.mean_df['categories'] == x['categories']]['embedding_tsne_x'].values[0], x['embedding_tsne_y'] - self.mean_df[self.mean_df['categories'] == x['categories']]['embedding_tsne_y'].values[0]]), axis=1)
                
                # Calculate the euclidean distance between the embedding x and y and the mean embedding
                self.df['distance'] = self.df.apply(lambda x: np.linalg.norm(x['vector']), axis=1)
            return self.df, self.mean_df
        
        except Exception as e:
            print(e)
            return None

    def plot_embeddings(self, show_fig=False, sample=False, sample_size=10000, general_categories=False, cluster_embeddings=True):
        '''
        Plot the embeddings with plotly.
        INPUTS:
            df: pandas dataframe with embeddings
            sample: boolean if the embeddings should be sampled
            sample_size: size of the sample
            general_categories: boolean if the general categories should be used
        OUTPUTS:
            fig: plotly figure
        '''
        # Plot the embeddings with plotly
        with alive_bar(title="Plotting the embeddings"):
            if sample:
                self.df = self.df.sample(sample_size)
            # Create a set of unique categories and sort them
            categories = set(self.df['categories'].tolist())
            sorted_categories = sorted(categories)
            
            # Create a dict with key the general category (first part of category split by . or -) and value the subcategory (second part of category split by . or -) as list
            # general_categories_dict = {}
            # for category in sorted_categories:
            #     general_category = category.split('.', 1)[0]
            #     # general_category = category.split('.', 1)[0].split('-', 1)[0]
            #     if general_category not in general_categories_dict:
            #         general_categories_dict[general_category] = [category]
            #     else:
            #         general_categories_dict[general_category].append(category)
            general_categories_taxonomy = {
                "cs": "Computer Science",
                "econ": "Economics",
                "eess": "Electrical Engineering and Systems Science",
                "math": "Mathematics",
                "astro-ph": "Astrophysics",
                "cond-mat": "Condensed Matter",
                "gr-qc": "General Relativity and Quantum Cosmology",
                "hep-ex": "High Energy Physics",
                "hep-lat": "High Energy Physics",
                "hep-ph": "High Energy Physics",
                "hep-th": "High Energy Physics",
                "math-ph": "Mathematical Physics",
                "nlin": "Nonlinear Sciences",
                "nucl-ex": "Nuclear Physics",
                "nucl-th": "Nuclear Physics",
                "quant-ph": "Quantum Physics",
                "physics": "Physics",
                "q-bio": "Quantitative Biology",
                "q-fin": "Quantitative Finance",
                "stat": "Statistics"
            }
            
            # Create a new column for the general category and fill it with the general category
            self.df['general_category'] = self.df['categories'].str.split('.', 1).str[0]
            # Create a new column and match the general category with the general category taxonomy. If the general category is not in the taxonomy, use "other"
            self.df['general_category_taxonomy'] = self.df['general_category'].map(general_categories_taxonomy).fillna("Other")
            # Create a color palette for the general categories
            general_categories_palette = {
                "Computer Science": "#1f77b4",
                "Economics": "#ff7f0e",
                "Electrical Engineering and Systems Science": "#2ca02c",
                "Mathematics": "#d62728",
                "Physics": "#9467bd",
                "Astrophysics": "#8c564b",
                "Condensed Matter": "#e377c2",
                "General Relativity and Quantum Cosmology": "#7f7f7f",
                "High Energy Physics": "#bcbd22",
                "Mathematical Physics": "#2ca02c",
                "Nonlinear Sciences": "#d62728",
                "Nuclear Physics": "#9467bd",
                "Quantum Physics": "#e377c2",
                "Quantitative Biology": "#8c564b",
                "Quantitative Finance": "#e377c2",
                "Statistics": "#7f7f7f",
                "Other": "#bcbd22"
            }
            
            # Create a new column for the color and fill it with the color of the general category
            self.df['color'] = self.df['general_category_taxonomy'].map(general_categories_palette)
            
            # Create a sorted_categories list with the general categories from A to Z
            sorted_categories = sorted(general_categories_palette.keys())
            
            # Create a new column in the mean_df for the general category and fill it with the general category
            self.mean_df['general_category'] = self.mean_df['categories'].str.split('.', 1).str[0]
            # Create a new column and match the general category with the general category taxonomy. If the general category is not in the taxonomy, use "other"
            self.mean_df['general_category_taxonomy'] = self.mean_df['general_category'].map(general_categories_taxonomy).fillna("Other")
            # Create a new column for the color and fill it with the color of the general category
            self.mean_df['color'] = self.mean_df['general_category_taxonomy'].map(general_categories_palette)

            if not general_categories:
                # Plot the embeddings with custom hover data
                fig = px.scatter(
                    self.df,
                    x='embedding_tsne_x',
                    y='embedding_tsne_y',
                    color='categories',
                    hover_data=['title', 'authors', 'categories'],
                    title='Embedding similarity',
                    category_orders={'categories': sorted_categories},
                    hover_name="title",  # set the hover name to the paper title
                    # color_discrete_sequence=px.colors.qualitative.Light24,
                    # template='plotly_dark',
                    render_mode='webgl',
                    # color_discrete_map=sub_colors,
                    labels={'categories': 'Category'}
                )

                # Set the x and y axis labels
                fig.update_xaxes(title_text="t-SNE 1")
                fig.update_yaxes(title_text="t-SNE 2")
            else:
                # Plot the embeddings with custom hover data
                fig = px.scatter(
                    self.df,
                    x='embedding_tsne_x',
                    y='embedding_tsne_y',
                    color='general_category_taxonomy',
                    hover_data=['title', 'authors', 'categories'],
                    title='Embedding similarity',
                    category_orders={'categories': sorted_categories},
                    hover_name="title",  # set the hover name to the paper title
                    # color_discrete_sequence=px.colors.qualitative.Light24,
                    # template='plotly_dark',
                    render_mode='webgl',
                    color_discrete_map=general_categories_palette,
                    labels={'general_category_taxonomy': 'General category'}
                )

            # Set the x and y axis labels
            fig.update_xaxes(title_text="t-SNE 1")
            fig.update_yaxes(title_text="t-SNE 2")
               
            # Add the mean embeddings to the scatter plot. I only want to see the text of the category on top of the scatter plot, not as hover text. Make the size of the text object dependend on the number of papers in the category
            if not general_categories:
                for index, row in self.mean_df.iterrows():
                    count = self.df[self.df['categories'] == row['categories']].count()[0]
                    fig.add_annotation(
                        x=row['embedding_tsne_x'],
                        y=row['embedding_tsne_y'],
                        text=row['categories'],
                        showarrow=False,
                        font=dict(
                            size = 10 + count/len(self.df)*100,
                            color="black",
                        ),
                        bgcolor="white",
                        opacity=0.8,
                    )
            else:  
                # Add the mean embeddings of the general categories to the scatter plot. I only want to see the text of the category on top of the scatter plot, not as hover text. Make the size of the text object dependend on the number of papers in the category
                
                # Calculate the mean per general category and write it into a new dataframe
                general_mean_df = self.df.groupby('general_category_taxonomy').mean().reset_index()
                # print(general_mean_df)
                for index, row in general_mean_df.iterrows():
                    count = self.df[self.df['general_category_taxonomy'] == row['general_category_taxonomy']].count()[0]
                    fig.add_annotation(
                        x=row['embedding_tsne_x'],
                        y=row['embedding_tsne_y'],
                        text=row['general_category_taxonomy'],
                        showarrow=False,
                        font=dict(
                            size = 10 + count/len(self.df)*100,
                            color="black",
                        ),
                        bgcolor="white",
                        opacity=0.8,
                    )
        
        if cluster_embeddings:
            with alive_bar(title="Clustering the embeddings"):
                # Cluster tsne coordinates using HDBSCAN
                embedding_list = self.df['embedding'].tolist()
                tsne_x_list = self.df['embedding_tsne_x'].tolist()
                tsne_y_list = self.df['embedding_tsne_y'].tolist()
                min_cluster_size_calc = int(len(self.df) / 300)
                print(f"Calculated minimum Cluster Size: {min_cluster_size_calc}")
                clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size_calc, algorithm="best", metric="euclidean", cluster_selection_method='eom')
                clusterer.fit(np.array([tsne_x_list, tsne_y_list]).T)
                self.df['embedding_tsne_cluster'] = clusterer.labels_
                
                # Assign topic names to the clusters
                for cluster in self.df['embedding_tsne_cluster'].unique():
                    if cluster != -1:
                        # Get the most common topic in the cluster
                        topic = self.df[self.df['embedding_tsne_cluster'] == cluster]['categories'].value_counts().index[0]
                        # Assign the topic name to the cluster
                        self.df.loc[self.df['embedding_tsne_cluster'] == cluster, 'embedding_tsne_cluster'] = topic
                    else:
                        self.df.loc[self.df['embedding_tsne_cluster'] == cluster, 'embedding_tsne_cluster'] = "Noise"
                
                print(f"Number of clusters: {len(self.df['embedding_tsne_cluster'].unique())}")
                print(f"Number of points in the clusters: {len(self.df[self.df['embedding_tsne_cluster'] != 'Noise'])}")
                print(f"Number of points in the noise cluster: {len(self.df[self.df['embedding_tsne_cluster'] == 'Noise'])}")
                print(f"Number of points in the dataset: {len(self.df)}")
                print("")
                print(f"Number of points in the largest cluster: {len(self.df[self.df['embedding_tsne_cluster'] == self.df['embedding_tsne_cluster'].value_counts().index[0]])}")
                print(f"Number of points in the second largest cluster: {len(self.df[self.df['embedding_tsne_cluster'] == self.df['embedding_tsne_cluster'].value_counts().index[1]])}")
                print(f"Number of points in the third largest cluster: {len(self.df[self.df['embedding_tsne_cluster'] == self.df['embedding_tsne_cluster'].value_counts().index[2]])}")
                
                # Plot a pie chart using plotly to show the distribution of the clusters
                fig_pie = px.pie(
                    self.df,
                    names="embedding_tsne_cluster",
                    title="Distribution of the clusters",
                    hole=0.5
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                
                if show_fig:
                    fig_pie.show()
                    
                # Create a shape from all coordinates from each cluster. Use the outtermost coordinates to draw a line around the cluster. This will be used to draw a border around the clusters
                shapes = []
                for cluster in self.df['embedding_tsne_cluster'].unique():
                    if cluster != "Noise":
                        x_list = self.df[self.df['embedding_tsne_cluster'] == cluster]['embedding_tsne_x'].tolist()
                        y_list = self.df[self.df['embedding_tsne_cluster'] == cluster]['embedding_tsne_y'].tolist()
                        x_max = max(x_list)
                        x_min = min(x_list)
                        y_max = max(y_list)
                        y_min = min(y_list)
                        shapes.append(
                            dict(
                                type="circle",
                                xref="x",
                                yref="y",
                                x0=x_min,
                                y0=y_min,
                                x1=x_max,
                                y1=y_max,
                                line=dict(
                                    color="blue",
                                    width=2,
                                ),
                                fillcolor="white",
                                opacity=0.1,
                            )
                        )
                        # Add the cluster name to the plot in the top left corner of the cluster shape
                        fig.add_annotation(
                            x=x_max,
                            y=(y_max + y_min) / 2,
                            xanchor="left",
                            text=cluster,
                            showarrow=False,
                            font=dict(
                                size=12,
                                color="blue",
                            ),
                            bgcolor="white",
                            opacity=0.8,
                            hovertext=", ".join(set([category.split(".")[0] for category in self.df['categories'].tolist()])),
                        )
                        
                        
                fig.update_layout(shapes=shapes)
                
                # Add buttons to enable/disable the clusters
                # fig.update_layout(
                #     updatemenus=[
                #         dict(
                #             type="buttons",
                #             direction="right",
                #             active=0,
                #             x=0.57,
                #             y=1.2,
                #             buttons=list([
                #                 dict(
                #                     label="Show all scatter points",
                #                     method="update",
                #                     args=[{"visible": [True] * len(self.df['embedding_tsne_cluster'].unique())}],
                #                 ),
                #                 dict(
                #                     label="Hide all scatter points",
                #                     method="update",
                #                     args=[{"visible": [False] * len(self.df['embedding_tsne_cluster'].unique())}],
                #                 ),
                #                 #  Add buttons to show and hide shapes
                #                 dict(
                #                     label="Show all HDBSCAN clusters",
                #                     method="relayout",
                #                     args=[{"shapes.visible": True}],
                #                 ),
                #                 dict(
                #                     label="Hide all HDBSCAN clusters",
                #                     method="relayout",
                #                     args=[{"shapes.visible": False}],
                #                 ),
                #                 # Add buttons to show and hide t-sne mean lables
                #                 dict(
                #                     label="Show all t-sne mean labels",
                #                     method="update",
                #                     args=[{"annotations.visible": True}],
                #                 ),
                #                 dict(
                #                     label="Hide all t-sne mean labels",
                #                     method="update",
                #                     args=[{"annotations.visible": False}],
                #                 ),
                #             ]),
                #         ),
                #     ]
                # )
                
                
                if show_fig:
                    fig.show()
                    
                # Save figure
                output_html = os.path.join(self.project_dir, "embedding_tsne.html")
                output_png = os.path.join(self.project_dir, "embedding_tsne.png")
                output_svg = os.path.join(self.project_dir, "embedding_tsne.svg")
                fig.write_html(output_html)
                fig.write_image(output_png, scale=2, width=2000, height=1200)
                fig.write_image(output_svg, scale=2, width=2000, height=1200)
        else:
            # Save the plot as a html file
            output_html = os.path.join(self.project_dir, "embedding_similarity.html")
            output_png = os.path.join(self.project_dir, "embedding_similarity.png")
            output_svg = os.path.join(self.project_dir, "embedding_similarity.svg")
            fig.write_html(output_html)
            fig.write_image(output_png, scale=2, width=2000, height=1200)
            fig.write_image(output_svg, scale=2, width=2000, height=1200)
        
            if show_fig:
                fig.show()
        
        print("Done!")
        
        return fig
