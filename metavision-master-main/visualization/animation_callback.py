import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot

from dash import Output, Input, State, callback, dcc, html, ALL, ctx
import dash
from flask import session
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import io
import base64
import logging

logger = logging.getLogger("metavision")

def register_animation_callback(app, cache):
    # Show main content and hide placeholder when a colormap is selected
    @callback(
        Output('animation-main-content', 'children'),
        Output('animation-placeholder', 'style'),
        [Input('colormap-radio-animation', 'value')],
        prevent_initial_call=True
    )
    def show_main_content(colormap):
        if not colormap:
            # Only show the placeholder, not the graph, slider, or colorbar
            return html.Div(), {"textAlign": "center", "color": "#b0bec5", "fontSize": "1.2rem", "marginTop": "40px", "display": "block"}
        compound_matrix = cache.get(f"{session['session_id']}:compound_matrix")
        if compound_matrix is None:
            return html.Div("No data available for animation.", className="error-message"), {"display": "none"}
        slices = compound_matrix.shape[0]
        # Show the graph, slider, and colorbar only after a colormap is selected
        return [
            dcc.Graph(
                id="animation-main-graph",
                config={
                    'displayModeBar': False,
                    'staticPlot': False,
                    'responsive': True
                },
                style={"width": "auto", "height": "auto", "background": "transparent", "border": "none", "boxShadow": "none", "padding": "0", "margin": "0"}
            ),
            dcc.Slider(
                id="animation-slice-slider",
                min=0,
                max=slices-1,
                value=0,
                marks={i: str(i+1) for i in range(slices)},
                step=1,
                updatemode="drag",
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            html.Div(id="animation-colorbar-container", style={"marginTop": "16px", "textAlign": "center"}),
        ], {"display": "none"}

    @callback(
        Output('animation-main-graph', 'figure'),
        [Input('animation-slice-slider', 'value'),
         Input('colormap-radio-animation', 'value')],
        prevent_initial_call=True
    )
    def update_scatter_figure(slice_idx, colormap):
        compound_matrix = cache.get(f"{session['session_id']}:compound_matrix")
        if compound_matrix is None:
            return dash.no_update
        slices, rows, cols = compound_matrix.shape
        if slice_idx is None or slice_idx < 0 or slice_idx >= slices:
            slice_idx = 0
        data = compound_matrix[slice_idx]
        y, x = np.nonzero(data)
        values = data[y, x]
        vmin = 0.0
        vmax = float(np.max(compound_matrix))  # Use max instead of percentile for brighter colors
        
        # Enhanced bright colorscales - prioritize brightest ones
        bright_colorscales = {
            'Hot': 'Hot',
            'Jet': 'Jet',
            'Rainbow': 'Rainbow',
            'Turbo': 'Turbo',
            'Viridis': 'Viridis',
            'Plasma': 'Plasma',
            'Inferno': 'Inferno',
            'Magma': 'Magma',
            'Reds': 'Reds',
            'Blues': 'Blues',
            'Greens': 'Greens',
            'Purples': 'Purples',
            'Oranges': 'Oranges'
        }
        
        # Use very bright red colorscale for maximum visibility
        colorscale = [[0, 'rgba(255,0,0,0.5)'], [0.3, 'rgba(255,0,0,0.8)'], [0.7, 'rgba(255,0,0,1)'], [1, 'rgba(255,0,0,1)']]
        
        scatter = go.Scattergl(
            x=x,
            y=y,
            mode='markers',
                            marker=dict(
                    color='#FF3333',  # Force enhanced bright red color
                    size=14,  # Increased size for better visibility
                    showscale=False,
                    line=dict(width=0),
                    opacity=1.0  # Full opacity for maximum visibility
                ),
            hovertemplate='<b>🧠 Brain Data</b><br>x: %{x}<br>y: %{y}<br>Intensity: %{marker.color:.3f}<extra></extra>'
        )
        fig = go.Figure(data=[scatter])
        fig.update_layout(
            xaxis=dict(
                visible=False, 
                showgrid=False, 
                zeroline=False, 
                showticklabels=False, 
                constrain='domain',
                showline=False,
                mirror=False
            ),
            yaxis=dict(
                visible=False, 
                showgrid=False, 
                zeroline=False, 
                showticklabels=False, 
                scaleanchor='x', 
                scaleratio=1, 
                constrain='domain', 
                autorange='reversed',
                showline=False,
                mirror=False
            ),
            plot_bgcolor='rgba(255,255,255,1)',  # White background
            paper_bgcolor='rgba(255,255,255,1)',  # White background
            margin=dict(l=0, r=0, t=0, b=0),  # No margins
                            autosize=True,
                height=700,
                width=700,
            dragmode=False,
                            title=dict(
                    text=f"🧠 Enhanced Brain Animation {slice_idx + 1}",
                    font=dict(size=20, color='#FF3333'),
                    x=0.5,
                    xanchor='center'
                )
        )
        return fig

    @callback(
        Output('animation-colorbar-container', 'children'),
        [Input('colormap-radio-animation', 'value'),
         Input('animation-slice-slider', 'value')],
        prevent_initial_call=True
    )
    def update_colorbar(colormap, slice_idx):
        compound_matrix = cache.get(f"{session['session_id']}:compound_matrix")
        if compound_matrix is None:
            return None
        slices, rows, cols = compound_matrix.shape
        if slice_idx is None or slice_idx < 0 or slice_idx >= slices:
            slice_idx = 0
        data = compound_matrix[slice_idx]
        vmax = float(np.percentile(compound_matrix[compound_matrix != 0], 99))
        vmin = 0.0
        
        # Enhanced bright colorscales - prioritize brightest ones
        bright_colorscales = {
            'Hot': 'Hot',
            'Jet': 'Jet',
            'Rainbow': 'Rainbow',
            'Turbo': 'Turbo',
            'Viridis': 'Viridis',
            'Plasma': 'Plasma',
            'Inferno': 'Inferno',
            'Magma': 'Magma',
            'Reds': 'Reds',
            'Blues': 'Blues',
            'Greens': 'Greens',
            'Purples': 'Purples',
            'Oranges': 'Oranges'
        }
        
        colorscale = bright_colorscales.get(colormap, colormap)
        
        import matplotlib.pyplot as plt
        import io, base64
        fig, ax = plt.subplots(figsize=(6, 0.8), dpi=120)
        fig.subplots_adjust(bottom=0.5, top=1, left=0.05, right=0.95)
        norm = plt.Normalize(vmin, vmax)
        cb1 = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(colorscale)),
            cax=ax, 
            orientation='horizontal', 
            ticks=[vmin, (vmin+vmax)/2, vmax]
        )
        cb1.outline.set_visible(False)  # Remove colorbar border
        cb1.ax.tick_params(color='#2c3e50', labelcolor='#2c3e50', labelsize=12, length=0)
        cb1.set_label('Intensity', color='#2c3e50', fontsize=14, fontweight='bold')
        fig.patch.set_alpha(0)  # Transparent background
        ax.set_frame_on(False)
        for spine in ax.spines.values():
            spine.set_visible(False)  # Remove all borders
        buf = io.BytesIO()
        fig.savefig(buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0, dpi=120)
        plt.close(fig)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('ascii')
        return html.Img(
            src=f"data:image/png;base64,{img_str}", 
            style={
                "maxWidth": "400px", 
                "height": "auto", 
                "margin": "0 auto", 
                "display": "block", 
                "background": "transparent", 
                "border": "none", 
                "boxShadow": "none"
            }
        )

