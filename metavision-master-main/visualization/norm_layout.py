import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot

from dash import Output, Input, State, callback, dcc, html, ALL
import dash
from flask import session
import numpy as np
import plotly.graph_objects as go
import logging
import matplotlib.pyplot as plt
import io
import base64

logger = logging.getLogger("metavision")

def get_norm_layout(cache):
    compound_matrix = cache.get(f"{session['session_id']}:compound_matrix")
    if compound_matrix is None:
        raise ValueError("No compound matrix found in cache.")
    
    # Create the normalization boxplot
    image = create_norm_boxplot(compound_matrix)
    
    return html.Div([
        html.H3("Slice Intensity Distribution", className="section-title"),
        html.P("This plot shows the distribution of non-zero intensity values for each slice.", 
               className="section-description"),
        # dcc.Graph(
        #     id="normalization-boxplot",
        #     figure=fig,
        #     config={
        #         'displayModeBar': True,
        #         'modeBarButtonsToRemove': ['lasso2d', 'select2d']
        #     },
        #     className="norm-boxplot"
        # )
        html.Img(
            id="normalization-boxplot",
            src=f"data:image/png;base64,{image}",
            className="norm-boxplot",
            style={
                'width': '100%',
                'height': 'auto',
                'border-radius': '10px',
                'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.2)',
                'margin-top': '20px'
            }
        )
    ], id="normalization-plot", className="norm-container")

def create_norm_boxplot(matrix):
    """
    Creates a boxplot showing the distribution of non-zero intensity values for each slice.
    """
    # Reshape matrix to combine rows and columns for each slice
    slices, rows, cols = matrix.shape
    matrix_reshape = matrix.reshape(slices, -1)
    
    # Extract non-zero values for each slice
    non_zero_list = []
    for row in matrix_reshape:
        non_zero_values = row[np.nonzero(row)]
        non_zero_list.append(non_zero_values)
    
    fig, ax = plt.subplots(figsize=(6, 15))
    ax.boxplot(non_zero_list, vert=False, showfliers=False)
    ax.invert_yaxis()
    ax.set_xlabel('Intensity',fontweight='bold',fontsize=15)
    ax.set_ylabel('Slice',fontweight='bold',fontsize=15)
    ax.tick_params(axis='both', which='both', labelsize=10)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    labels = [label.set_fontweight('bold') for label in labels]
    
    # Convert matplotlib figure to base64 encoded PNG
    buf = io.BytesIO()
    fig.savefig(buf, format='png', transparent=True, 
                bbox_inches='tight', pad_inches=0, 
                facecolor='none', edgecolor='none')
    plt.close(fig)  # Close figure to free memory
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('ascii')
    
    return img_str