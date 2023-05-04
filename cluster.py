'''
Cluster the embeddings using HDBSCAN
'''
import pandas as pd
import hdbscan
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from tqdm import tqdm

with tqdm(total=1, desc="Loading JSON") as pbar:
    df = pd.read_json('projects/arXiv_sample_20000_SciNCL/embedding_arxiv-metadata-oai-snapshot.json', lines=True)
    pbar.update()
    
embedding_list = df['embedding'].tolist()

tsne_x_list = df['embedding_tsne_x'].tolist()
tsne_y_list = df['embedding_tsne_y'].tolist()

# Cluster tsne coordinates using HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=135, algorithm="best", metric="manhattan", cluster_selection_method='eom')
clusterer.fit(np.array([tsne_x_list, tsne_y_list]).T)
df['embedding_tsne_cluster'] = clusterer.labels_

# Assign topic names to the clusters
for cluster in df['embedding_tsne_cluster'].unique():
    if cluster != -1:
        # Get the most common topic in the cluster
        topic = df[df['embedding_tsne_cluster'] == cluster]['categories'].value_counts().index[0]
        # Assign the topic name to the cluster
        df.loc[df['embedding_tsne_cluster'] == cluster, 'embedding_tsne_cluster'] = topic
    else:
        df.loc[df['embedding_tsne_cluster'] == cluster, 'embedding_tsne_cluster'] = "Noise"
        
print(f"Number of clusters: {len(df['embedding_tsne_cluster'].unique())}")
print(f"Number of points in the clusters: {len(df[df['embedding_tsne_cluster'] != 'Noise'])}")
print(f"Number of points in the noise cluster: {len(df[df['embedding_tsne_cluster'] == 'Noise'])}")
print(f"Number of points in the dataset: {len(df)}")
print("")
print(f"Number of points in the largest cluster: {len(df[df['embedding_tsne_cluster'] == df['embedding_tsne_cluster'].value_counts().index[0]])}")
print(f"Number of points in the second largest cluster: {len(df[df['embedding_tsne_cluster'] == df['embedding_tsne_cluster'].value_counts().index[1]])}")
print(f"Number of points in the third largest cluster: {len(df[df['embedding_tsne_cluster'] == df['embedding_tsne_cluster'].value_counts().index[2]])}")

# Plot a pie chart using plotly to show the distribution of the clusters
fig_pie = px.pie(
    df,
    names="embedding_tsne_cluster",
    title="Distribution of the clusters",
    hole=0.5
)
fig_pie.update_traces(textposition='inside', textinfo='percent+label')
fig_pie.show()


# print(df['embedding_tsne_cluster'].head())
# Scatter plot of the tsne embeddings with the clusters. Draw a line around all the points in the cluster. Colorize the points by category.
fig = px.scatter(
    df,
    x="embedding_tsne_x",
    y="embedding_tsne_y",
    color="general_category_taxonomy",
    hover_name="title",
    # Add to the hover data the cluster name, categories, and authors
    hover_data={
        "embedding_tsne_cluster": True,
        "categories": True,
        "authors": True,
    },
    opacity=0.5
)

# Create a shape from all coordinates from each cluster. Use the outtermost coordinates to draw a line around the cluster.
for cluster in df['embedding_tsne_cluster'].unique():
    if cluster != "Noise":
        # Get the points in the cluster
        cluster_df = df[df['embedding_tsne_cluster'] == cluster]
        # Get the x and y coordinates of the points
        x_list = cluster_df['embedding_tsne_x'].tolist()
        y_list = cluster_df['embedding_tsne_y'].tolist()
        # Get the outtermost coordinates
        x_min = min(x_list)
        x_max = max(x_list)
        y_min = min(y_list)
        y_max = max(y_list)
        # Add the shape to the plot
        fig.add_shape(
            type="rect",
            x0=x_min,
            y0=y_min,
            x1=x_max,
            y1=y_max,
            line=dict(
                color="black",
                width=1,
            ),
        )
        
# Add lables to the shapes to show the cluster name
for cluster in df['embedding_tsne_cluster'].unique():
    if cluster != "Noise":
        # Get the points in the cluster
        cluster_df = df[df['embedding_tsne_cluster'] == cluster]
        # Get the x and y coordinates of the points
        x_list = cluster_df['embedding_tsne_x'].tolist()
        y_list = cluster_df['embedding_tsne_y'].tolist()
        # Get the outtermost coordinates
        x_min = min(x_list)
        x_max = max(x_list)
        y_min = min(y_list)
        y_max = max(y_list)
        # Add the label to the plot
        fig.add_annotation(
            x=x_min,
            y=y_max,
            # text should be name of the cluster + the number of points in the cluster. Cut the cluster name before the "."
            text=f"{cluster.split('.')[0]} ({len(cluster_df)})",
            showarrow=False,
            font=dict(
                size=10,
                color="black",
            ),
            bgcolor="white",
            opacity=0.8,
            # Add a hover label which shows the categories inside this cluster. Only show the string before the ".". Make a set of the categories to remove duplicates.
            hovertext=", ".join(set([category.split(".")[0] for category in cluster_df['categories'].tolist()])),
        )    

# Connect each point in every cluster with a line. The lines should be named after the cluster. Use label and hovertext to show the cluster name.
# for cluster in df['embedding_tsne_cluster'].unique():
#     if cluster != "Noise":
#         # Get the points in the cluster
#         cluster_df = df[df['embedding_tsne_cluster'] == cluster]
#         # Get the x and y coordinates of the points
#         x_list = cluster_df['embedding_tsne_x'].tolist()
#         y_list = cluster_df['embedding_tsne_y'].tolist()
#         # Add the line to the plot
#         fig.add_trace(
#             go.Scatter(
#                 x=x_list,
#                 y=y_list,
#                 mode="lines",
#                 name=cluster,
#                 hovertext=cluster,
#                 hoverinfo="text",
#                 line=dict(
#                     color="black",
#                     width=1,
#                 ),
#             )
#         )

        
# Save the plot as a html file
output_html = ("embedding_similarity_clustered.html")
output_png = ("embedding_similarity_clustered.png")
output_svg = ("embedding_similarity_clustered.svg")

fig.write_html(output_html)
fig.write_image(output_png, scale=2, width=2000, height=1200)
fig.write_image(output_svg, scale=2, width=2000, height=1200)

fig.show()

print("Done!")