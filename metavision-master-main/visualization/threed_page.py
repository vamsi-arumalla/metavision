import dash
from dash import html, dcc
import plotly.graph_objects as go
import numpy as np
from flask import session
import logging

logger = logging.getLogger("metavision")

def get_3d_visualization_page(cache):
    """Create the full 3D visualization page"""
    try:
        # Get data from cache
        session_id = session.get('session_id', 'default')
        compound_matrix = cache.get(f"{session_id}:compound_matrix")
        
        # Debug logging
        logger.info(f"[3D Page] Session ID: {session_id}")
        logger.info(f"[3D Page] Cache key: {session_id}:compound_matrix")
        logger.info(f"[3D Page] Compound matrix found: {compound_matrix is not None}")
        if compound_matrix is not None:
            logger.info(f"[3D Page] Matrix shape: {compound_matrix.shape}")
        
        # Create a simple 3D visualization even if no data is available
        if compound_matrix is None:
            # Create a demo 3D visualization with proper brain-like structure
            compound_matrix = np.zeros((18, 60, 60))  # Larger size for better detail
            
            # Create a solid brain-like structure with continuous regions and high-intensity red areas
            for z in range(18):
                for y in range(60):
                    for x in range(60):
                        # Create a more realistic brain shape
                        center_x, center_y, center_z = 30, 30, 9
                        
                        # Create two hemispheres (left and right brain)
                        left_center_x = center_x - 8
                        right_center_x = center_x + 8
                        
                        # Distance from left hemisphere
                        dx_left = (x - left_center_x) / 12.0
                        dy_left = (y - center_y) / 10.0
                        dz_left = (z - center_z) / 6.0
                        distance_left = np.sqrt(dx_left**2 + dy_left**2 + dz_left**2)
                        
                        # Distance from right hemisphere
                        dx_right = (x - right_center_x) / 12.0
                        dy_right = (y - center_y) / 10.0
                        dz_right = (z - center_z) / 6.0
                        distance_right = np.sqrt(dx_right**2 + dy_right**2 + dz_right**2)
                        
                        # Use the closer hemisphere
                        distance = min(distance_left, distance_right)
                        
                        if distance < 1.0:  # Brain volume
                            # Create solid, continuous regions with high-intensity red areas
                            
                            # High intensity cortical regions (bright red)
                            cortical_intensity = 500 * np.exp(-distance * 1.2)
                            
                            # Medium-high intensity regions (orange/yellow)
                            medium_intensity = 350 * np.exp(-distance * 1.6)
                            
                            # Medium intensity regions (green)
                            subcortical_intensity = 250 * np.exp(-distance * 1.9)
                            
                            # Low intensity deep regions (blue)
                            deep_intensity = 150 * np.exp(-distance * 2.2)
                            
                            # Create bright cortical folds with higher intensity
                            cortical_pattern = np.sin(x * 0.15) * np.cos(y * 0.12) * np.sin(z * 0.25)
                            fold_intensity = 120 * np.abs(cortical_pattern)
                            
                            # Create high-intensity hippocampus structure (bright red)
                            hippo_distance = np.sqrt((x - center_x)**2 + (y - (center_y + 5))**2)
                            if hippo_distance < 10 and y > center_y:
                                hippocampus_intensity = 600 * np.exp(-hippo_distance / 3)
                            else:
                                hippocampus_intensity = 0
                            
                            # Create high-intensity thalamus region (bright red)
                            thalamus_distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                            if thalamus_distance < 6:
                                thalamus_intensity = 700 * np.exp(-thalamus_distance / 2)
                            else:
                                thalamus_intensity = 0
                            
                            # Combine all effects with solid regional variations
                            if distance < 0.25:  # Outer cortical layer (bright red)
                                intensity = cortical_intensity + fold_intensity + hippocampus_intensity + thalamus_intensity
                            elif distance < 0.5:  # Middle cortical layer (orange/yellow)
                                intensity = medium_intensity + fold_intensity * 0.7 + hippocampus_intensity * 0.5
                            elif distance < 0.75:  # Subcortical layer (green)
                                intensity = subcortical_intensity + fold_intensity * 0.3
                            else:  # Deep regions (blue)
                                intensity = deep_intensity
                            
                            # Add very minimal noise for clarity
                            noise = np.random.normal(0, 0.2)
                            intensity += noise
                            
                            compound_matrix[z, y, x] = max(0, intensity)
            
            # Apply stronger smoothing for better initial appearance
            try:
                from scipy.ndimage import gaussian_filter
                compound_matrix = gaussian_filter(compound_matrix, sigma=0.8)
            except:
                pass
            
            logger.info("[3D Page] Using proper brain demo data for 3D visualization")
        
        # Create 2D slice figure
        def create_2d_slice_figure(arr, slice_idx=0):
            """Create 2D brain slice visualization with solid colors and high-intensity red regions"""
            try:
                # Get the slice data
                slice_data = arr[slice_idx]
                
                # Apply stronger smoothing for solid, smooth appearance
                try:
                    from scipy.ndimage import gaussian_filter
                    slice_data = gaussian_filter(slice_data, sigma=1.5)  # Stronger smoothing for solid appearance
                except:
                    pass
                
                # Normalize data for better visualization with enhanced contrast
                if np.max(slice_data) > 0:
                    # Apply contrast enhancement
                    slice_data = np.power(slice_data, 0.8)  # Enhance contrast
                    slice_data = slice_data / np.max(slice_data)
                
                # Create the main heatmap
                fig = go.Figure()
                
                # Add the heatmap with bright, vibrant colorscale for high quality
                fig.add_trace(go.Heatmap(
                    z=slice_data,
                    colorscale=[
                        [0.0, '#FFFFFF'],    # White for lowest intensity
                        [0.1, '#87CEEB'],    # Sky blue
                        [0.3, '#32CD32'],    # Lime green
                        [0.5, '#FFD700'],    # Gold
                        [0.7, '#FF6347'],    # Tomato red
                        [0.9, '#FF1493'],    # Deep pink
                        [1.0, '#FF0000']     # Bright red for highest intensity
                    ],
                    colorbar=dict(
                        title=dict(text='Intensity', font=dict(size=14, color='white')),
                        x=1.02, 
                        len=0.8,
                        thickness=15,
                        tickfont=dict(size=12, color='white'),
                        tickcolor='white',
                                            tickmode='array',
                    tickvals=[0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                    ticktext=['0', '0.01', '0.02', '0.03', '0.04', '0.05', '0.06']
                    ),
                    zsmooth='best',  # Enable smoothing for solid, smooth appearance
                    connectgaps=True,
                    hoverongaps=False,
                    hovertemplate='<b>Brain Slice</b><br>X: %{x}<br>Y: %{y}<br>Intensity: %{z:.4f}<extra></extra>',
                    autocolorscale=False,
                    reversescale=False
                ))
                
                # Add crosshair lines (4 lines around the slide)
                rows, cols = slice_data.shape
                
                # Horizontal line through center
                fig.add_trace(go.Scatter(
                    x=[0, cols-1],
                    y=[rows//2, rows//2],
                    mode='lines',
                    line=dict(color='red', width=3),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Vertical line through center
                fig.add_trace(go.Scatter(
                    x=[cols//2, cols//2],
                    y=[0, rows-1],
                    mode='lines',
                    line=dict(color='red', width=3),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Add labels for the crosshair
                fig.add_annotation(
                    x=cols//2 + 3,
                    y=rows//2,
                    text="L",
                    font=dict(color="white", size=14, family="Arial Black"),
                    showarrow=False
                )
                
                fig.add_annotation(
                    x=cols//2,
                    y=rows//2 - 3,
                    text="A",
                    font=dict(color="white", size=14, family="Arial Black"),
                    showarrow=False
                )
                
                # Update the layout
                fig.update_layout(
                        margin=dict(l=10, r=50, t=40, b=10),
                        title=dict(
                            text=f"Slice {slice_idx+1} of {arr.shape[0]}",
                            font=dict(size=18, color='white'),
                            x=0.5
                        ),
                        xaxis=dict(
                            showticklabels=False, 
                            showgrid=False, 
                            zeroline=False,
                            showspikes=False,
                            visible=False,
                            range=[0, cols-1]
                        ),
                        yaxis=dict(
                            showticklabels=False, 
                            showgrid=False, 
                            zeroline=False,
                            showspikes=False,
                            visible=False,
                            range=[0, rows-1],
                            scaleanchor="x",
                            scaleratio=1
                        ),
                        plot_bgcolor='black',
                        paper_bgcolor='black',
                        height=650,
                        width=650,
                        autosize=True
                    )
                
                return fig
                
            except Exception as e:
                logger.error(f"Failed to create 2D slice: {e}")
                # Fallback
                fig = go.Figure()
                fig.add_trace(go.Heatmap(z=[[0, 1], [1, 0]], colorscale='rainbow'))
                fig.update_layout(
                    title=dict(text="Error loading slice", font=dict(color='white')),
                    plot_bgcolor='black',
                    paper_bgcolor='black'
                )
                return fig
        
        # Create 3D figure
        def create_3d_figure(arr):
            """Create 3D brain visualization"""
            try:
                from skimage import measure
                from scipy.ndimage import gaussian_filter
                
                # Create a more visible 3D visualization
                fig = go.Figure()
                
                # Add multiple visualization methods for better visibility
                
                # 1. Scatter3D points for immediate visibility
                x_coords, y_coords, z_coords = np.where(arr > np.percentile(arr, 50))
                if len(x_coords) > 1000:  # Limit points for performance
                    indices = np.random.choice(len(x_coords), 1000, replace=False)
                    x_coords = x_coords[indices]
                    y_coords = y_coords[indices]
                    z_coords = z_coords[indices]
                
                if len(x_coords) > 0:
                    fig.add_trace(go.Scatter3d(
                        x=x_coords,
                        y=y_coords,
                        z=z_coords,
                        mode='markers',
                        marker=dict(
                            size=3,
                            color=arr[x_coords, y_coords, z_coords],
                            colorscale='rainbow',
                            opacity=0.8
                        ),
                        name='Brain Points'
                    ))
                
                # 2. Surface visualization for solid appearance
                try:
                    # Smooth the data for better surface generation
                    smoothed_data = gaussian_filter(arr, sigma=1.0)
                    
                    # Normalize data
                    data_min = np.nanmin(smoothed_data)
                    data_max = np.nanmax(smoothed_data)
                    
                    if data_max > data_min:
                        # Create isosurface
                        threshold = np.percentile(smoothed_data[smoothed_data > 0], 70)
                        
                        if threshold > data_min:
                            verts, faces, _, _ = measure.marching_cubes(
                                smoothed_data, 
                                level=threshold, 
                                spacing=(1, 1, 1),
                                step_size=1,
                                allow_degenerate=False
                            )
                            
                            if len(verts) > 0 and len(faces) > 0:
                                x, y, z = verts.T
                                i, j, k = faces.T
                                
                                fig.add_trace(go.Mesh3d(
                                    x=x, y=y, z=z,
                                    i=i, j=j, k=k,
                                    color='rgba(255,0,0,0.3)',
                                    opacity=0.3,
                                    lighting=dict(
                                        ambient=0.4,
                                        diffuse=0.8,
                                        fresnel=0.1,
                                        specular=0.05,
                                        roughness=0.1
                                    ),
                                    lightposition=dict(x=100, y=100, z=1000),
                                    name='Brain Surface',
                                    showlegend=False
                                ))
                except Exception as e:
                    logger.warning(f"Could not create surface: {e}")
                
                # 3. Add some reference points for orientation
                fig.add_trace(go.Scatter3d(
                    x=[0, arr.shape[2]],
                    y=[0, arr.shape[1]],
                    z=[0, arr.shape[0]],
                    mode='markers',
                    marker=dict(size=10, color='white'),
                    name='Reference Points'
                ))
                
                return fig
                
                # Create multiple isosurface levels for solid brain appearance
                traces = []
                percentiles = [20, 40, 60, 80]
                opacities = [0.2, 0.3, 0.4, 0.6]
                colors = ['rgba(139,0,0,{})', 'rgba(178,34,34,{})', 'rgba(220,20,60,{})', 'rgba(255,0,0,{})']
                
                for i, (percentile, opacity) in enumerate(zip(percentiles, opacities)):
                    try:
                        threshold = np.percentile(smoothed_data[smoothed_data > 0], percentile)
                        
                        if threshold > data_min:
                            verts, faces, _, _ = measure.marching_cubes(
                                smoothed_data, 
                                level=threshold, 
                                spacing=(1, 1, 1),
                                step_size=1,
                                allow_degenerate=False
                            )
                            
                            if len(verts) > 0 and len(faces) > 0:
                                x, y, z = verts.T
                                i, j, k = faces.T
                                
                                mesh = go.Mesh3d(
                                    x=x, y=y, z=z,
                                    i=i, j=j, k=k,
                                    color=colors[i].format(opacity),
                                    opacity=opacity,
                                    lighting=dict(
                                        ambient=0.4,
                                        diffuse=0.8,
                                        fresnel=0.1,
                                        specular=0.05,
                                        roughness=0.1
                                    ),
                                    lightposition=dict(x=100, y=100, z=1000),
                                    name=f'Brain Layer {percentile}%',
                                    showlegend=False,
                                    flatshading=False
                                )
                                traces.append(mesh)
                    
                    except Exception as e:
                        logger.warning(f"Could not create isosurface at {percentile}%: {e}")
                        continue
                
                # Create the figure
                fig = go.Figure(data=traces)
                
                # Configure layout for full-page visualization
                fig.update_layout(
                    title=dict(
                        text="3D Brain Visualization",
                        font=dict(color='white', size=20),
                        x=0.5
                    ),
                    scene=dict(
                        xaxis=dict(title="", showgrid=False, showticklabels=False, showbackground=False, showspikes=False, zeroline=False, visible=False),
                        yaxis=dict(title="", showgrid=False, showticklabels=False, showbackground=False, showspikes=False, zeroline=False, visible=False),
                        zaxis=dict(title="", showgrid=False, showticklabels=False, showbackground=False, showspikes=False, zeroline=False, visible=False),
                        camera=dict(
                            eye=dict(x=1.5, y=1.5, z=1.2),
                            center=dict(x=0, y=0, z=0)
                        ),
                        bgcolor='black',
                        aspectmode='cube'
                    ),
                    margin=dict(l=10, r=10, t=60, b=10),
                    paper_bgcolor='black',
                    plot_bgcolor='black',
                    height=650,
                    width=650,
                    autosize=True,
                    showlegend=False,
                    font=dict(color='white')
                )
                
                return fig
                
            except Exception as e:
                logger.error(f"Failed to create 3D brain: {e}")
                # Emergency fallback
                fig = go.Figure()
                fig.add_trace(go.Scatter3d(
                    x=[10, 20, 30],
                    y=[10, 20, 30],
                    z=[10, 20, 30],
                    mode='markers',
                    marker=dict(size=15, color='red'),
                    name='Error'
                ))
                
                # Update the layout
                fig.update_layout(
                    title=dict(text="3D Visualization Error", font=dict(color='white')),
                    scene=dict(bgcolor='black'),
                    paper_bgcolor='black',
                    height=800
                )
                return fig
        
        # Create slide options for dropdown
        num_slices = compound_matrix.shape[0]
        slide_options = [{'label': f'Slide {i+1}', 'value': i} for i in range(num_slices)]
        
        # Create the 3D visualization page layout
        return html.Div([
            # Header with title and close button
            html.Div([
                html.H1("3D Brain Visualization", 
                       style={"color": "#ffffff", "margin": "0", "float": "left", "fontSize": "2.5rem", "fontWeight": "bold"}),
                html.Button("×", id="close-3d-page", 
                          style={"float": "right", "background": "none", "border": "none", 
                                "color": "#ffffff", "fontSize": "36px", "cursor": "pointer", "padding": "0", "fontWeight": "bold"})
            ], style={"backgroundColor": "#1a1a1a", "borderBottom": "2px solid #333", "padding": "25px", "overflow": "hidden"}),
            
            # Slide selector dropdown at the top
            html.Div([
                html.H3("Select Slide:", style={"color": "#ffffff", "marginRight": "15px", "display": "inline-block"}),
                dcc.Dropdown(
                    id="slide-selector-3d-page",
                    options=slide_options,
                    value=0,  # Default to first slide
                    style={
                        "backgroundColor": "#2d2d2d", 
                        "color": "#ffffff", 
                        "border": "1px solid #555", 
                        "width": "300px",
                        "display": "inline-block"
                    }
                )
            ], style={"backgroundColor": "#1a1a1a", "padding": "20px", "borderBottom": "1px solid #333", "textAlign": "center"}),
            
            # Hidden stores for data
            dcc.Store(id="selected-point-3d", data=None),
            
            # Main content area with 2D and 3D images side by side
            html.Div([
                # Left side - 2D slice image with controls
                html.Div([
                    html.H3("2D Brain Slice", style={"color": "#ffffff", "textAlign": "center", "marginBottom": "15px"}),
                    html.Div([
                        html.P("Click on a point in the 2D image to select it, then click 'Run 3D Analysis'", 
                               style={"color": "#cccccc", "textAlign": "center", "fontSize": "14px", "marginBottom": "10px"}),
                        html.Button(
                            "Run 3D Analysis",
                            id="run-3d-analysis-btn",
                            n_clicks=0,
                            style={
                                "backgroundColor": "#4CAF50",
                                "color": "white",
                                "border": "none",
                                "padding": "10px 20px",
                                "borderRadius": "6px",
                                "fontSize": "16px",
                                "fontWeight": "bold",
                                "cursor": "pointer",
                                "margin": "0 auto 15px auto",
                                "display": "block"
                            }
                        )
                    ], style={"textAlign": "center", "marginBottom": "15px"}),
                    dcc.Graph(
                        id="2d-slice-graph-3d-page",
                        figure=go.Figure().add_annotation(
                            text="Loading 2D Brain Slice...<br>Please select a slide from the dropdown above",
                            xref="paper", yref="paper",
                            x=0.5, y=0.5, showarrow=False,
                            font=dict(size=20, color="white")
                        ).update_layout(
                            plot_bgcolor='black',
                            paper_bgcolor='black',
                            height=650,
                            width=650,
                            margin=dict(l=10, r=10, t=10, b=10)
                        ),
                        style={"height": "700px", "width": "100%", "backgroundColor": "black"},
                        config={
                            'displayModeBar': True,
                            'displaylogo': False,
                            'modeBarButtonsToRemove': ['pan2d', 'select2d', 'lasso2d'],
                            'responsive': True
                        }
                    )
                ], style={"flex": "1", "marginRight": "20px", "backgroundColor": "#2a2a2a", "padding": "15px", "borderRadius": "8px", "minHeight": "800px", "overflow": "visible"}),
                
                # Right side - 3D visualization
                html.Div([
                    html.H3("3D Brain Visualization", style={"color": "#ffffff", "textAlign": "center", "marginBottom": "15px"}),
                    html.Div("Select a slide from the dropdown above to view 2D slice, then click 'Run 3D Analysis' to generate 3D visualization", 
                             id="3d-analysis-status", 
                             style={"color": "#cccccc", "textAlign": "center", "fontSize": "14px", "marginBottom": "10px"}),
                    dcc.Graph(
                        id="3d-graph-3d-page",
                        figure=go.Figure().add_annotation(
                            text="Loading 3D Brain Visualization...<br>Please select a slide and click 'Run 3D Analysis'",
                            xref="paper", yref="paper",
                            x=0.5, y=0.5, showarrow=False,
                            font=dict(size=20, color="white")
                        ).update_layout(
                            plot_bgcolor='black',
                            paper_bgcolor='black',
                            height=650,
                            width=650,
                            margin=dict(l=10, r=10, t=10, b=10)
                        ),
                        style={"height": "700px", "width": "100%", "backgroundColor": "black"},
                        config={
                            'displayModeBar': True,
                            'displaylogo': False,
                            'modeBarButtonsToRemove': ['pan2d', 'select2d', 'lasso2d'],
                            'responsive': True
                        }
                    )
                ], style={"flex": "1", "marginLeft": "20px", "backgroundColor": "#2a2a2a", "padding": "15px", "borderRadius": "8px", "minHeight": "800px", "overflow": "visible"})
                
            ], style={"display": "flex", "padding": "20px", "minHeight": "800px", "overflow": "visible"}),
            
            # Return to Dashboard button at the bottom - Fixed position
            html.Div([
                html.Button(
                    "Return to Dashboard",
                    id="return-to-dashboard-from-3d",
                    n_clicks=0,
                    style={
                        "backgroundColor": "#9c27b0",
                        "color": "white",
                        "border": "none",
                        "padding": "15px 30px",
                        "borderRadius": "8px",
                        "fontSize": "18px",
                        "fontWeight": "bold",
                        "cursor": "pointer",
                        "margin": "20px auto",
                        "display": "block",
                        "position": "relative",
                        "zIndex": "1000"
                    }
                )
            ], style={"textAlign": "center", "padding": "20px", "borderTop": "2px solid #333", "backgroundColor": "#1a1a1a", "position": "relative", "zIndex": "999"})
            
        ], style={"backgroundColor": "#1a1a1a", "minHeight": "100vh", "color": "#ffffff", "display": "flex", "flexDirection": "column", "overflow": "auto"})
        
    except Exception as e:
        logger.error(f"Error creating 3D visualization page: {e}")
        return html.Div([
            html.H1("Error Loading 3D Visualization", style={"color": "#ffffff", "textAlign": "center"}),
            html.P(f"An error occurred: {str(e)}", style={"color": "#cccccc", "textAlign": "center"}),
            html.Button("Return to Dashboard", id="return-to-dashboard-from-3d", 
                      style={"backgroundColor": "#9c27b0", "color": "white", "border": "none", 
                            "padding": "12px 24px", "borderRadius": "6px", "fontSize": "16px", 
                            "fontWeight": "bold", "cursor": "pointer", "margin": "20px auto", "display": "block"})
        ], style={"backgroundColor": "#1a1a1a", "minHeight": "100vh", "color": "#ffffff", "padding": "50px"}) 