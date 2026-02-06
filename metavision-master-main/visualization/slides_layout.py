from dash import dcc, html, Input, Output, State, callback_context, ALL, ctx, no_update
import plotly.graph_objects as go
import numpy as np
from flask import session
import logging
import json

logger = logging.getLogger("metavision")

def get_slides_layout(cache):
    """
    Returns the layout for the Slides visualization with a grid of thumbnails on the left 
    and a larger selected image on the right.
    """
    # Create a dynamic layout that will be populated by callbacks
    # This avoids the issue of trying to generate thumbnails before compound matrix is available

        # Enhanced 3D Slides Layout with colormap selection
    # Create colormap selection buttons
    colormaps = [
        ("Reds", "#d32f2f", "🔴"),
        ("Viridis", "#43a047", "🟢"), 
        ("Magma", "#ec407a", "🟣"),
        ("Cividis", "#ff9800", "🟡"),
        ("Rainbow", "#9c27b0", "🌈"),
        ("Greys", "#333", "⚫")
    ]
    
    # View type selection buttons
    view_types = [
        ("2d", "#2196f3", "📊", "2D Heatmaps"),
        ("3d", "#ff5722", "🧊", "3D Surfaces"),
        ("both", "#9c27b0", "🔄", "Both Views")
    ]
    
    colormap_bar = html.Div([
        html.H3("Choose View Type and Colormap:", style={
            "textAlign": "center", 
            "color": "#2196f3", 
            "marginBottom": "20px",
            "fontSize": "1.5rem"
        }),
        
        # View type selection
        html.Div([
            html.Label("View Type:", style={
                "display": "block",
                "textAlign": "center",
                "color": "#666",
                "marginBottom": "10px",
                "fontSize": "1.1rem",
                "fontWeight": "bold"
            }),
            html.Div([
                *[
                    html.Button(
                        [emoji, " ", label],
                        id={"type": "view-btn", "view": view},
                        n_clicks=0,
                        style={
                            "background": color,
                            "color": "#fff",
                            "border": "none",
                            "borderRadius": "8px",
                            "padding": "12px 24px",
                            "marginRight": "16px",
                            "marginBottom": "8px",
                            "fontWeight": "bold",
                            "fontSize": "1.1rem",
                            "cursor": "pointer",
                            "boxShadow": "0 2px 4px rgba(0,0,0,0.2)",
                            "transition": "all 0.3s ease"
                        }
                    ) for view, color, emoji, label in view_types
                ]
            ], style={
                "display": "flex", 
                "justifyContent": "center", 
                "flexWrap": "wrap",
                "marginBottom": "20px"
            })
        ]),
        
        # Colormap selection
        html.Div([
            html.Label("Colormap:", style={
                "display": "block",
                "textAlign": "center",
                "color": "#666",
                "marginBottom": "10px",
                "fontSize": "1.1rem",
                "fontWeight": "bold"
            }),
            html.Div([
                *[
                    html.Button(
                        [emoji, " ", cmap],
                        id={"type": "colormap-btn", "cmap": cmap},
                        n_clicks=0,
                        style={
                            "background": color,
                            "color": "#fff" if cmap != "Greys" else "#222",
                            "border": "none",
                            "borderRadius": "8px",
                            "padding": "12px 24px",
                            "marginRight": "16px",
                            "marginBottom": "8px",
                            "fontWeight": "bold",
                            "fontSize": "1.1rem",
                            "cursor": "pointer",
                            "boxShadow": "0 2px 4px rgba(0,0,0,0.2)",
                            "transition": "all 0.3s ease"
                        }
                    ) for cmap, color, emoji in colormaps
                ]
            ], style={
                "display": "flex", 
                "justifyContent": "center", 
                "flexWrap": "wrap",
                "marginBottom": "32px"
            })
        ])
    ])
    
    # Store for selected colormap and view type
    colormap_store = dcc.Store(id="slides-colormap-store", data="Reds")
    view_store = dcc.Store(id="slides-view-store", data="2d")
    
    # Placeholder for slides (will be populated by callback)
    slides_placeholder = html.Div([
        html.Div([
            html.I(className="fas fa-images", style={"color": "#b0bec5", "fontSize": "4rem", "marginBottom": "16px"}),
            html.H3("Select view type and colormap above to generate slides", style={"color": "#b0bec5"}),
            html.P("Choose between 2D heatmaps, 3D surfaces, or both views", style={"color": "#9e9e9e", "fontSize": "1rem"})
        ], style={"textAlign": "center", "padding": "80px 20px"})
    ], id="slides-3d-grid", style={
        "border": "2px dashed #e0e0e0",
        "borderRadius": "12px", 
        "minHeight": "400px",
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "center",
        "background": "#fafafa"
    })
    
    return html.Div([
        html.H2("Brain Slides Visualization", style={
            "textAlign": "center", 
            "color": "#d32f2f", 
            "marginBottom": "32px",
            "fontSize": "2.2rem",
            "fontWeight": "bold"
        }),
        colormap_bar,
        colormap_store,
        view_store,
        slides_placeholder
    ], className="slides-visualization-container", style={"padding": "20px"})

# Callback functions for 3D slides visualization
def register_slides_colormap_callbacks(app, cache):
    @app.callback(
        Output("slides-colormap-store", "data"),
        [Input({"type": "colormap-btn", "cmap": ALL}, "n_clicks")],
        [State({"type": "colormap-btn", "cmap": ALL}, "id")],
        prevent_initial_call=True
    )
    def set_colormap(n_clicks_list, id_list):
        if not n_clicks_list or not id_list:
            return no_update
        triggered = ctx.triggered_id
        if triggered and isinstance(triggered, dict) and triggered.get("type") == "colormap-btn":
            print(f"[DEBUG] Colormap button clicked: {triggered['cmap']}")
            return triggered["cmap"]
        return no_update
    
    @app.callback(
        Output("slides-view-store", "data"),
        [Input({"type": "view-btn", "view": ALL}, "n_clicks")],
        [State({"type": "view-btn", "view": ALL}, "id")],
        prevent_initial_call=True
    )
    def set_view_type(n_clicks_list, id_list):
        if not n_clicks_list or not id_list:
            return no_update
        triggered = ctx.triggered_id
        if triggered and isinstance(triggered, dict) and triggered.get("type") == "view-btn":
            print(f"[DEBUG] View type button clicked: {triggered['view']}")
            return triggered["view"]
        return no_update

    @app.callback(
        Output("slides-3d-grid", "children"),
        [Input("slides-colormap-store", "data"),
         Input("slides-view-store", "data")],
        prevent_initial_call=True
    )
    def update_slides_3d_grid(selected_cmap, selected_view):
        print(f"[DEBUG] === SLIDES VISUALIZATION CALLBACK ===")
        print(f"[DEBUG] Slides callback triggered - colormap: {selected_cmap}, view: {selected_view}")
        print(f"[DEBUG] This should create SOLID heatmaps, not scattered points")
        
        session_id = session.get('session_id', 'default')
        compound_matrix = cache.get(f"{session_id}:compound_matrix")
        
        if compound_matrix is None:
            print("[DEBUG] No compound matrix available for 3D slides")
            print(f"[DEBUG] Session ID: {session_id}")
            try:
                cache_keys = cache.get('*')
                if cache_keys:
                    session_keys = [k for k in cache_keys if session_id in str(k)]
                    print(f"[DEBUG] Available cache keys for session: {session_keys}")
                else:
                    print("[DEBUG] No cache keys available")
            except Exception as e:
                print(f"[DEBUG] Error accessing cache: {e}")
            return html.Div([
                html.Div([
                    html.I(className="fas fa-exclamation-triangle", style={"color": "#ff9800", "fontSize": "3rem", "marginBottom": "16px"}),
                    html.H3("No Data Available", style={"color": "#ff9800", "marginBottom": "8px"}),
                    html.P("Please upload and process a file first, then select a molecule.", style={"color": "#b0bec5"}),
                    html.P(f"Session: {session_id}", style={"color": "#b0bec5", "fontSize": "0.8rem"}),
                ], style={"textAlign": "center", "padding": "60px 20px"})
            ])
        
        try:
            print(f"[DEBUG] Generating 3D slides with matrix shape: {compound_matrix.shape}")
            print(f"[DEBUG] Matrix data type: {compound_matrix.dtype}")
            print(f"[DEBUG] Matrix min/max: {np.nanmin(compound_matrix)}/{np.nanmax(compound_matrix)}")
            print(f"[DEBUG] Matrix has NaN: {np.isnan(compound_matrix).any()}")
            print(f"[DEBUG] Matrix has Inf: {np.isinf(compound_matrix).any()}")
            print(f"[DEBUG] Non-zero elements: {np.count_nonzero(compound_matrix)}")
            
            # Clean the data and ensure proper intensity scaling
            compound_matrix = np.nan_to_num(compound_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Ensure we have a proper 2D array for each slice
            print(f"[DEBUG] Original matrix shape: {compound_matrix.shape}")
            
            # Normalize the data to ensure good contrast
            if np.max(compound_matrix) > 0:
                compound_matrix = compound_matrix / np.max(compound_matrix)
            
            print(f"[DEBUG] After normalization - min: {np.min(compound_matrix)}, max: {np.max(compound_matrix)}")
            print(f"[DEBUG] Non-zero elements: {np.count_nonzero(compound_matrix)}")
            print(f"[DEBUG] Total elements: {compound_matrix.size}")
            print(f"[DEBUG] Data type: {compound_matrix.dtype}")
            
            slide_figs = []
            
            # Enhanced custom colorscales with vibrant intensity-based coloring
            custom_colorscales = {
                "Reds": [
                    [0.0, '#ffffff'],  # White for zero values
                    [0.1, '#ffcdd2'],  # Light red
                    [0.3, '#ef9a9a'],  # Medium light red
                    [0.5, '#e57373'],  # Medium red
                    [0.7, '#f44336'],  # Bright red
                    [0.9, '#d32f2f'],  # Dark red
                    [1.0, '#b71c1c']   # Very dark red
                ],
                "Viridis": [
                    [0.0, '#ffffff'],  # White for zero values
                    [0.1, '#440154'],  # Dark purple
                    [0.3, '#31688e'],  # Blue
                    [0.5, '#35b779'],  # Green
                    [0.7, '#90d743'],  # Light green
                    [0.9, '#fde725'],  # Yellow
                    [1.0, '#ffff00']   # Bright yellow
                ],
                "Magma": [
                    [0.0, '#ffffff'],  # White for zero values
                    [0.1, '#000004'],  # Black
                    [0.3, '#1b0c41'],  # Dark purple
                    [0.5, '#4f0a6d'],  # Purple
                    [0.7, '#a52c60'],  # Pink
                    [0.9, '#ed6925'],  # Orange
                    [1.0, '#fbf976']   # Bright yellow
                ],
                "Cividis": [
                    [0.0, '#ffffff'],  # White for zero values
                    [0.1, '#00204c'],  # Dark blue
                    [0.3, '#424086'],  # Blue
                    [0.5, '#6c5e7b'],  # Purple
                    [0.7, '#a38a4c'],  # Brown
                    [0.9, '#d6bd3d'],  # Gold
                    [1.0, '#f9e721']   # Bright yellow
                ],
                "Greys": [
                    [0.0, '#ffffff'],  # White for zero values
                    [0.1, '#f7fafc'],  # Very light grey
                    [0.3, '#edf2f7'],  # Light grey
                    [0.5, '#cbd5e0'],  # Medium grey
                    [0.7, '#a0aec0'],  # Dark grey
                    [0.9, '#4a5568'],  # Very dark grey
                    [1.0, '#1a202c']   # Almost black
                ],
                "Rainbow": [
                    [0.0, '#ffffff'],  # White for zero values
                    [0.1, '#ff0000'],  # Red
                    [0.2, '#ff8000'],  # Orange
                    [0.3, '#ffff00'],  # Yellow
                    [0.4, '#80ff00'],  # Lime
                    [0.5, '#00ff00'],  # Green
                    [0.6, '#00ff80'],  # Light green
                    [0.7, '#00ffff'],  # Cyan
                    [0.8, '#0080ff'],  # Light blue
                    [0.9, '#0000ff'],  # Blue
                    [1.0, '#8000ff']   # Purple
                ]
            }
            
            # Generate 3D surfaces for each slice (more robust filtering)
            start_slice = 0  # Start from first slice
            max_slides = 70  # Increased limit for more slides
            generated_count = 0
            
            # Find valid slices with data
            valid_slices = []
            for i in range(compound_matrix.shape[0]):
                slide_data = compound_matrix[i]
                # More lenient threshold - check if slice has any non-zero data
                if np.any(slide_data > 0) and not np.all(np.isnan(slide_data)):
                    valid_slices.append(i)
            
            print(f"[DEBUG] Found {len(valid_slices)} valid slices out of {compound_matrix.shape[0]} total slices")
            
            # Use valid slices or fall back to sequential if none found
            if len(valid_slices) == 0:
                print("[DEBUG] No valid slices found, using sequential approach")
                valid_slices = list(range(min(compound_matrix.shape[0], max_slides)))
            
            for i in valid_slices[:max_slides]:
                slide_data = compound_matrix[i]
                
                print(f"[DEBUG] Processing slice {i} - shape: {slide_data.shape}")
                print(f"[DEBUG] Slice {i} - min: {np.min(slide_data)}, max: {np.max(slide_data)}")
                print(f"[DEBUG] Slice {i} - non-zero: {np.count_nonzero(slide_data)} out of {slide_data.size}")
                
                # Skip slices with no significant data (more lenient)
                if np.max(slide_data) < 1e-10:
                    print(f"[DEBUG] Skipping slice {i} - max value too low: {np.max(slide_data)}")
                    continue
                
                # Apply Gaussian smoothing to create more solid areas
                from scipy.ndimage import gaussian_filter
                try:
                    # Apply light smoothing to fill gaps and create solid areas
                    slide_data = gaussian_filter(slide_data, sigma=0.5)
                    print(f"[DEBUG] Applied Gaussian smoothing to slice {i}")
                except ImportError:
                    print(f"[DEBUG] scipy not available, using original data for slice {i}")
                except Exception as e:
                    print(f"[DEBUG] Error applying smoothing to slice {i}: {e}")
                
                # Create mesh coordinates
                y, x = np.mgrid[0:slide_data.shape[0], 0:slide_data.shape[1]]
                
                # Calculate colormap range
                cmin = 0
                cmax = float(np.nanmax(slide_data))
                if cmax == cmin:
                    cmax = cmin + 1e-8
                
                # Get colormap
                cmap = custom_colorscales.get(selected_cmap, selected_cmap)
                
                # Create 2D heatmap with solid intensity-based coloring
                fig_2d = go.Figure(
                    data=go.Heatmap(
                        z=slide_data,
                        colorscale=cmap,
                        showscale=True,  # Show colorbar for intensity reference
                        zsmooth='best',
                        connectgaps=True,
                        hoverongaps=False,
                        hoverinfo='z',
                        zmin=cmin,
                        zmax=cmax,
                        hovertemplate='<b>Brain Slice</b><br>X: %{x}<br>Y: %{y}<br>Intensity: %{z:.3f}<extra></extra>',
                        colorbar=dict(
                            title=dict(
                                text="Intensity",
                                side="right",
                                font=dict(size=12, color="#2c3e50")
                            ),
                            thickness=15,
                            len=0.6,
                            x=1.02,
                            tickfont=dict(size=10, color="#2c3e50")
                        ),
                        # Force solid rendering
                        autocolorscale=False,
                        reversescale=False
                    )
                )
                
                fig_2d.update_layout(
                    title=dict(
                        text=f"2D Slice {generated_count + 1}",
                        x=0.5,
                        font=dict(size=14, color="#2c3e50")
                    ),
                    xaxis=dict(
                        showticklabels=False, 
                        showgrid=False, 
                        zeroline=False,
                        showspikes=False,
                        spikethickness=0
                    ),
                    yaxis=dict(
                        showticklabels=False, 
                        scaleanchor="x", 
                        scaleratio=1, 
                        showgrid=False,
                        zeroline=False,
                        autorange='reversed',
                        showspikes=False,
                        spikethickness=0
                    ),
                    plot_bgcolor='rgba(255,255,255,1)',
                    paper_bgcolor='rgba(255,255,255,1)',
                    margin=dict(l=0, r=0, t=40, b=0, pad=0),
                    width=300,
                    height=300
                )
                
                # Create 3D surface plot
                fig_3d = go.Figure(
                    data=go.Surface(
                        z=slide_data,
                        x=x,
                        y=y,
                        colorscale=cmap,
                        showscale=True,
                        opacity=0.9,
                        cmin=cmin,
                        cmax=cmax,
                        colorbar=dict(
                            title=dict(
                                text="Intensity",
                                side="right"
                            ),
                            thickness=15,
                            len=0.7,
                            x=1.02
                        )
                    )
                )
                
                fig_3d.update_layout(
                    title=dict(
                        text=f"3D Surface - Slice {generated_count + 1}",
                        x=0.5,
                        font=dict(size=16, color="#2c3e50")
                    ),
                    scene=dict(
                        xaxis_title='X Coordinate',
                        yaxis_title='Y Coordinate', 
                        zaxis_title='Intensity',
                        aspectmode='manual',
                        aspectratio=dict(x=1, y=1, z=0.3),
                        bgcolor='rgba(248,249,250,1)',
                        camera=dict(
                            eye=dict(x=1.5, y=1.5, z=1.2),
                            center=dict(x=0, y=0, z=0)
                        ),
                        xaxis=dict(
                            showbackground=True,
                            backgroundcolor='rgba(230,230,230,0.5)',
                            showgrid=True,
                            gridcolor='rgba(255,255,255,0.8)'
                        ),
                        yaxis=dict(
                            showbackground=True,
                            backgroundcolor='rgba(230,230,230,0.5)',
                            showgrid=True,
                            gridcolor='rgba(255,255,255,0.8)'
                        ),
                        zaxis=dict(
                            showbackground=True,
                            backgroundcolor='rgba(230,230,230,0.5)',
                            showgrid=True,
                            gridcolor='rgba(255,255,255,0.8)'
                        )
                    ),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    margin=dict(l=0, r=0, b=0, t=50),
                    width=450,
                    height=450
                )
                
                # Create slide based on selected view type
                if selected_view == "2d":
                    # Only 2D heatmap
                    slide_figs.append(
                        html.Div([
                            dcc.Graph(
                                figure=fig_2d,
                                config={
                                    "displayModeBar": True,
                                    "scrollZoom": True,
                                    "displaylogo": False,
                                    "modeBarButtonsToRemove": ['pan2d', 'lasso2d', 'select2d']
                                }
                            )
                        ], style={
                            "display": "inline-block",
                            "margin": "15px",
                            "verticalAlign": "top",
                            "width": "300px",
                            "background": "transparent",
                            "border": "none",
                            "boxShadow": "none",
                            "padding": "0"
                        })
                    )
                elif selected_view == "3d":
                    # Only 3D surface
                    slide_figs.append(
                        html.Div([
                            dcc.Graph(
                                figure=fig_3d,
                                config={
                                    "displayModeBar": True,
                                    "scrollZoom": True,
                                    "displaylogo": False,
                                    "modeBarButtonsToRemove": ['pan2d', 'lasso2d', 'select2d']
                                }
                            )
                        ], style={
                            "display": "inline-block",
                            "margin": "15px",
                            "verticalAlign": "top",
                            "width": "450px",
                            "background": "transparent",
                            "border": "none",
                            "boxShadow": "none",
                            "padding": "0"
                        })
                    )
                else:  # "both" - show both side by side
                    slide_figs.append(
                        html.Div([
                            # 2D and 3D side by side
                            html.Div([
                                # 2D Image (left)
                                html.Div([
                                    dcc.Graph(
                                        figure=fig_2d,
                                        config={
                                            "displayModeBar": True,
                                            "scrollZoom": True,
                                            "displaylogo": False,
                                            "modeBarButtonsToRemove": ['pan2d', 'lasso2d', 'select2d']
                                        }
                                    )
                                ], style={
                                    "display": "inline-block",
                                    "margin": "0",
                                    "verticalAlign": "top",
                                    "width": "300px",
                                    "background": "transparent",
                                    "border": "none",
                                    "boxShadow": "none",
                                    "padding": "0"
                                }),
                                # 3D Surface (right)
                                html.Div([
                                    dcc.Graph(
                                        figure=fig_3d,
                                        config={
                                            "displayModeBar": True,
                                            "scrollZoom": True,
                                            "displaylogo": False,
                                            "modeBarButtonsToRemove": ['pan2d', 'lasso2d', 'select2d']
                                        }
                                    )
                                ], style={
                                    "display": "inline-block",
                                    "margin": "0",
                                    "verticalAlign": "top",
                                    "width": "450px",
                                    "background": "transparent",
                                    "border": "none",
                                    "boxShadow": "none",
                                    "padding": "0"
                                })
                            ], style={
                                "display": "flex",
                                "justifyContent": "center",
                                "alignItems": "flex-start",
                                "gap": "20px"
                            })
                        ], style={
                            "display": "inline-block",
                            "margin": "15px",
                            "verticalAlign": "top",
                            "width": "770px",  # Increased to accommodate both images
                            "background": "transparent",
                            "border": "none",
                            "boxShadow": "none",
                            "padding": "0"
                        })
                    )
                
                generated_count += 1
                if generated_count >= max_slides:
                    break
            
            if len(slide_figs) == 0:
                return html.Div([
                    html.Div([
                        html.I(className="fas fa-search", style={"color": "#ffa726", "fontSize": "3rem", "marginBottom": "16px"}),
                        html.H3("No Valid Slices Found", style={"color": "#ffa726", "marginBottom": "8px"}),
                        html.P("Try selecting a different molecule or check your data.", style={"color": "#b0bec5"}),
                    ], style={"textAlign": "center", "padding": "60px 20px"})
                ])
            
            print(f"[DEBUG] Generated {len(slide_figs)} 3D surface plots")
            return html.Div(slide_figs, style={"textAlign": "center"})
            
        except Exception as e:
            print(f"[DEBUG] Error generating 3D slides: {e}")
            return html.Div([
                html.Div([
                    html.I(className="fas fa-exclamation-circle", style={"color": "#f44336", "fontSize": "3rem", "marginBottom": "16px"}),
                    html.H3("Error Generating 3D Slides", style={"color": "#f44336", "marginBottom": "8px"}),
                    html.P(f"Error: {str(e)}", style={"color": "#b0bec5"}),
                ], style={"textAlign": "center", "padding": "60px 20px"})
            ])
