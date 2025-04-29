import pathlib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, TreeSearch, BDeuScore, MaximumLikelihoodEstimator, K2Score
from pgmpy.inference import VariableElimination
from typing import Dict, Optional 
from pgmpy.factors.discrete import State


def plot_values(values,
                label_map: Optional[Dict[int, str]] = None,
                x='Scenario',
                state_col='HeartDisease',
                y='Probability'):
    
    num_models = len(values)
    ncols = int(np.ceil(np.sqrt(num_models)))
    nrows = int(np.ceil(num_models / ncols))

    # Create subplots
    _, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows), sharey=True, squeeze=False)
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

    # Hide any unused axes at the end
    for i in range(plot_index, len(axes_flat)):
         fig.delaxes(axes_flat[i])

    plt.tight_layout()
    plt.show()