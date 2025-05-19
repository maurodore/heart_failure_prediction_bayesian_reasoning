import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pathlib

from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, TreeSearch, BDeuScore, MaximumLikelihoodEstimator, K2Score
from pgmpy.inference import VariableElimination
from typing import Dict, Optional 
from pgmpy.factors.discrete import State
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm
import time
from sklearn.metrics import accuracy_score, f1_score

def plot_values(values,
                label_map: Optional[Dict[int, str]] = None,
                x='Scenario',
                state_col='HeartDisease',
                y='Probability'):
    
    num_models = len(values)
    ncols = int(np.ceil(np.sqrt(num_models)))
    nrows = int(np.ceil(num_models / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows), sharey=True, squeeze=False)
    axes_flat = axes.flatten() 

    plot_index = 0
    for ax, name in zip(axes_flat, values):
        ax.set_title(f"{name} model")
        probs_df = pd.DataFrame([[str(scenario), heart_disease_state, float(probability)] 
                            for scenario in values[name]
                            for heart_disease_state, probability in enumerate(values[name][scenario])], 
                            columns=[x, state_col, y])
        
        if label_map is not None:
            probs_df[state_col] = probs_df[state_col].map(lambda x: label_map.get(x, x))

        sns.barplot(x=x, y=y, hue=state_col, data=probs_df, ax=ax)

        plot_index += 1

    for i in range(plot_index, len(axes_flat)):
         fig.delaxes(axes_flat[i])

    plt.tight_layout()
    plt.show()


def predict(X_test, evidence_dict, target_variable, inference_engine):
    predictions_list = []

    print(f"\nPredizione di '{target_variable}' per il Test Set...")
    for index, row in tqdm(X_test.iterrows(), total=X_test.shape[0], desc="Predicting Test"):
        try:
            query_result = inference_engine.query(
                variables=[target_variable],
                evidence=evidence_dict,
                show_progress=False
            )
            predicted_idx = np.argmax(query_result.values)

            predictions_list.append(predicted_idx)
        except Exception as e:
            print(f"\nErrore predizione riga {index}: {e}")
            predictions_list.append(np.nan) 

    return pd.Series(predictions_list, index=X_test.index)


def show_markov_blanket_nx(network, project_dir, initial_node, blanket=[]):
    G = nx.DiGraph()
    nodes = network.nodes()
    edges = network.edges()
    
    for node in nodes: 

        if node == initial_node:
            G.add_node(node, level=1, color='red')
        elif node in blanket:
            G.add_node(node, level=2, color='orange')
        else:
            G.add_node(node, level=3, color='blue')

    G.add_edges_from(edges)

    node_colors = [G.nodes[n]['color'] for n in G.nodes()]

    pos = nx.circular_layout(G)

    plt.figure(figsize=(7, 7))
    nx.draw(
        G,
        pos,
        with_labels=True,
        arrows=True,
        node_color=node_colors,
        node_size=2000,
        font_size=9,
        edgecolors='black',
        arrowsize=20,
    )
    plt.title('Markov Blanket for ' + initial_node)

    plt.savefig(pathlib.Path.joinpath(project_dir,"out/hc_constraint_model.png"))

    plt.show()


        