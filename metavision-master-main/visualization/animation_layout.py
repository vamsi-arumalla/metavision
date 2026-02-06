from dash import dcc, html
import plotly.graph_objects as go
import numpy as np
from flask import session

def get_animation_layout(cache):
    session_id = session.get('session_id', 'default')
    compound_matrix = cache.get(f"{session_id}:compound_matrix")
    if compound_matrix is None:
        return html.Div([
            html.H3("No data available for animation.", style={"color": "#f44336", "textAlign": "center"})
        ])
    
    n_slides = compound_matrix.shape[0]
    vmax = float(np.nanmax(compound_matrix))
    vmin = 0
    
    # Enhanced bright colorscales
    bright_colorscales = {
        'Viridis': 'viridis',
        'Plasma': 'plasma', 
        'Inferno': 'inferno',
        'Magma': 'magma',
        'Turbo': 'turbo',
        'Hot': 'hot',
        'Jet': 'jet',
        'Rainbow': 'rainbow'
    }
    
    # Create frames with enhanced styling
    frames = [
        go.Frame(
            data=[go.Heatmap(
                z=compound_matrix[i],
                colorscale=[[0, 'rgba(255,255,255,0)'], [0.15, 'rgba(255,100,100,0.8)'], [0.4, 'rgba(255,60,60,0.9)'], [1, 'rgba(255,20,20,1)']],  # Enhanced bright red and white only
                zmin=vmin,
                zmax=vmax,
                colorbar=dict(
                    title=dict(
                        text='Intensity',
                        font=dict(size=14, color='#333')
                    ),
                    thickness=20,
                    len=0.8,
                    x=1.02,
                    outlinewidth=0,  # Remove colorbar border
                    tickfont=dict(size=12, color='#333'),
                    tickcolor='#333'
                ),
                zsmooth='best',  # Enable smoothing for better appearance
                connectgaps=False,
                hoverongaps=False,
                hoverinfo='z',
                hovertemplate='<b>Intensity:</b> %{z}<extra></extra>',
                showscale=True  # Show the colorbar for intensity
            )],
            name=str(i)
        )
        for i in range(n_slides)
    ]
    
    initial_data = [go.Heatmap(
        z=compound_matrix[0],
        colorscale=[[0, 'rgba(255,255,255,0)'], [0.15, 'rgba(255,100,100,0.8)'], [0.4, 'rgba(255,60,60,0.9)'], [1, 'rgba(255,20,20,1)']],
        zmin=vmin,
        zmax=vmax,
        colorbar=dict(
            title=dict(
                text='Intensity',
                font=dict(size=14, color='#333')
            ),
            thickness=20,
            len=0.8,
            x=1.02,
            outlinewidth=0,
            tickfont=dict(size=12, color='#333'),
            tickcolor='#333'
        ),
        zsmooth='best',
        connectgaps=False,
        hoverongaps=False,
        hoverinfo='z',
        hovertemplate='<b>Intensity:</b> %{z}<extra></extra>',
        showscale=True  # Show the colorbar for intensity
    )]
    
    fig = go.Figure(
        data=initial_data,
        layout=go.Layout(
            title=dict(
                text="🎬 Interactive Brain Animation",
                font=dict(size=20, color='#666666'),
                x=0.5,
                xanchor='center'
            ),
            updatemenus=[{
                "type": "buttons",
                "buttons": [
                    {
                        "label": "▶️ Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 200, "redraw": True}, "fromcurrent": True}]
                    },
                    {
                        "label": "⏸️ Pause", 
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
                "bgcolor": "rgba(0,0,0,0)",
                "bordercolor": "rgba(0,0,0,0)",
                "borderwidth": 0
            }],
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 16, "color": "#666666"},
                    "prefix": "🧠 Slide: ",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 200, "easing": "cubic-in-out"},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "bgcolor": "rgba(0,0,0,0)",
                "bordercolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "activebgcolor": "rgba(0,0,0,0)",
                "steps": [
                    {
                        "args": [
                            [str(i)],
                            {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}
                        ],
                        "label": str(i+1),
                        "method": "animate"
                    } for i in range(n_slides)
                ]
            }],
            width=800,
            height=800,
            plot_bgcolor="rgba(255,255,255,1)",  # White background
            paper_bgcolor="rgba(255,255,255,1)",  # White background
            margin=dict(l=0, r=0, t=60, b=0, pad=0),  # Remove all margins and padding
            xaxis=dict(
                showticklabels=False, 
                showgrid=False, 
                zeroline=False,
                showline=False,
                mirror=False,
                showspikes=False,
                spikethickness=0,
                range=[0, compound_matrix.shape[2]],
                constrain='domain'
            ),
            yaxis=dict(
                showticklabels=False, 
                showgrid=False, 
                zeroline=False,
                showline=False,
                mirror=False,
                scaleanchor='x',
                scaleratio=1,
                showspikes=False,
                spikethickness=0,
                range=[0, compound_matrix.shape[1]],
                constrain='domain',
                autorange='reversed'
            ),
        ),
        frames=frames
    )
    
    return html.Div([
        html.H2("🎬 Interactive Brain Animation", 
                style={
                    "textAlign": "center", 
                    "color": "#666666", 
                    "marginBottom": "24px",
                    "fontSize": "28px",
                    "fontWeight": "bold",
                    "textShadow": "0 2px 4px rgba(0,0,0,0.1)"
                }),
        html.Div([
            html.P("✨ Explore your brain data through time with smooth animations", 
                   style={
                       "textAlign": "center", 
                       "color": "#7f8c8d", 
                       "marginBottom": "20px",
                       "fontSize": "16px"
                   }),
            dcc.Graph(
                figure=fig, 
                config={
                    "displayModeBar": True, 
                    "scrollZoom": True,
                    "modeBarButtonsToRemove": ["pan2d", "lasso2d", "select2d"],
                    "displaylogo": False
                },
                style={
                    "background": "transparent",
                    "border": "none",
                    "boxShadow": "none"
                }
            )
        ], style={
            "background": "transparent",
            "borderRadius": "20px",
            "padding": "30px",
            "boxShadow": "none",
            "border": "none"
        })
    ], style={
        "background": "transparent", 
        "borderRadius": "0px", 
        "padding": "0px", 
        "boxShadow": "none",
        "border": "none"
    })