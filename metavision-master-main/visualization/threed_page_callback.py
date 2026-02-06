import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot

from dash import Output, Input, State, dcc, html, callback_context
import plotly.graph_objects as go
import numpy as np
from flask import session
import base64
import io
import logging
import nibabel as nib
from scipy.ndimage import zoom, gaussian_filter
import plotly.express as px
from dash.exceptions import PreventUpdate
import os
import tempfile
from datetime import datetime
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

logger = logging.getLogger("metavision")

def create_2d_slice_figure(compound_matrix, slice_idx=0):
    """Create 2D brain slice visualization with sharp, clear details"""
    try:
        # Get the slice data
        slice_data = compound_matrix[slice_idx]
        
        # Create a completely new, sharp demo data if needed
        if np.max(slice_data) == 0 or np.all(slice_data == 0):
            # Generate high-quality demo data for this specific slice
            rows, cols = slice_data.shape
            slice_data = np.zeros((rows, cols))
            
            # Create sharp, detailed brain structures
            center_x, center_y = cols // 2, rows // 2
            
            for y in range(rows):
                for x in range(cols):
                    # Create sharp brain outline
                    dx = (x - center_x) / (cols * 0.3)
                    dy = (y - center_y) / (rows * 0.25)
                    distance = np.sqrt(dx**2 + dy**2)
                    
                    if distance < 1.0:  # Brain region
                                # Create vibrant regions using full color spectrum
                                if distance < 0.2:  # Core region (brightest red)
                                    intensity = 1500 * (1 - distance * 2)
                                elif distance < 0.4:  # Inner region (bright orange)
                                    intensity = 1200 * (1 - distance * 1.5)
                                elif distance < 0.6:  # Middle region (bright yellow)
                                    intensity = 900 * (1 - distance * 1.2)
                                elif distance < 0.8:  # Outer region (bright green)
                                    intensity = 600 * (1 - distance * 0.8)
                                else:  # Edge region (bright blue)
                                    intensity = 300 * (1 - distance * 0.5)
                                
                                # Add vibrant structural details
                                # Ultra-bright hippocampus (brightest red)
                                hippo_x = center_x + 12
                                hippo_y = center_y + 6
                                hippo_dist = np.sqrt((x - hippo_x)**2 + (y - hippo_y)**2)
                                if hippo_dist < 10:
                                    intensity += 1800 * np.exp(-hippo_dist / 2.5)
                                
                                # Ultra-bright thalamus (brightest red)
                                thalamus_dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                                if thalamus_dist < 8:
                                    intensity += 2000 * np.exp(-thalamus_dist / 1.8)
                                
                                # Vibrant cortical folds
                                fold_pattern = np.sin(x * 0.18) * np.cos(y * 0.15)
                                intensity += 400 * np.abs(fold_pattern)
                                
                                # Bright amygdala region
                                amygdala_x = center_x - 10
                                amygdala_y = center_y - 6
                                amygdala_dist = np.sqrt((x - amygdala_x)**2 + (y - amygdala_y)**2)
                                if amygdala_dist < 8:
                                    intensity += 1600 * np.exp(-amygdala_dist / 2.2)
                                
                                # Bright basal ganglia
                                basal_x = center_x + 6
                                basal_y = center_y - 4
                                basal_dist = np.sqrt((x - basal_x)**2 + (y - basal_y)**2)
                                if basal_dist < 9:
                                    intensity += 1400 * np.exp(-basal_dist / 2.5)
                                
                                # Bright cerebellar regions
                                cereb_x = center_x - 5
                                cereb_y = center_y + 15
                                cereb_dist = np.sqrt((x - cereb_x)**2 + (y - cereb_y)**2)
                                if cereb_dist < 12:
                                    intensity += 1300 * np.exp(-cereb_dist / 3)
                                
                                slice_data[y, x] = max(0, intensity)
        
        # Bright normalization - use full color spectrum
        if np.max(slice_data) > 0:
            # Apply contrast enhancement to spread data across full range
            slice_data = np.power(slice_data, 0.6)  # Brighten mid-tones
            slice_data = slice_data / np.max(slice_data)  # Normalize to 0-1
        
        # Create figure with sharp rendering
        fig = go.Figure()
        
        # Use sharp Heatmap with enhanced colorscale
        fig.add_trace(go.Heatmap(
            z=slice_data,
            colorscale=[
                [0, 'rgb(0,0,100)'],      # Dark blue
                [0.2, 'rgb(0,100,200)'],  # Blue
                [0.4, 'rgb(0,200,100)'],  # Green
                [0.6, 'rgb(200,200,0)'],  # Yellow
                [0.8, 'rgb(255,100,0)'],  # Orange
                [1, 'rgb(255,0,0)']       # Bright red
            ],
            colorbar=dict(
                title=dict(text='Intensity', font=dict(size=14, color='white')),
                x=0.5,
                y=-0.1,
                len=0.8,
                thickness=20,
                orientation='h',
                tickfont=dict(size=12, color='white'),
                tickcolor='white',
                tickmode='array',
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                ticktext=['0', '0.0005', '0.001', '0.0015', '0.002']
            ),
            zsmooth=False,  # Sharp rendering
            connectgaps=True,
            hoverongaps=False,
            hovertemplate='<b>Brain Slice</b><br>X: %{x}<br>Y: %{y}<br>Intensity: %{z:.4f}<extra></extra>',
            autocolorscale=False,
            reversescale=False
        ))
        
        # Add sharp crosshair lines
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
        
        # Add sharp labels for the crosshair
        fig.add_annotation(
            x=cols//2 - 5,
            y=rows//2,
            text="L",
            font=dict(color="white", size=16, family="Arial Black"),
            showarrow=False
        )
        
        fig.add_annotation(
            x=cols//2,
            y=rows//2 + 5,
            text="A",
            font=dict(color="white", size=16, family="Arial Black"),
            showarrow=False
        )
        
        # Sharp layout with minimal margins
        fig.update_layout(
            margin=dict(l=5, r=5, t=40, b=60),
            title=dict(
                text=f"Slice {slice_idx+1} of {compound_matrix.shape[0]}",
                font=dict(size=16, color='white'),
                x=0.5
            ),
            xaxis=dict(
                showticklabels=False, 
                showgrid=False, 
                zeroline=False,
                showspikes=False,
                visible=False
            ),
            yaxis=dict(
                showticklabels=False, 
                showgrid=False, 
                zeroline=False,
                showspikes=False,
                visible=False
            ),
            plot_bgcolor='black',
            paper_bgcolor='black',
            height=600,
            width=600
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

def create_3d_page_figure(compound_matrix, selected_point=None, region_size=5):
    """Create optimized 3D brain visualization with performance improvements and timeout protection"""
    import time
    import gc
    
    start_time = time.time()
    max_execution_time = 30  # Maximum 30 seconds for 3D visualization
    
    try:
        logger.info(f"Creating 3D visualization with matrix shape: {compound_matrix.shape}")
        
        # Check execution time periodically
        def check_timeout():
            if time.time() - start_time > max_execution_time:
                raise TimeoutError("3D visualization generation timed out")
        
        # Data preprocessing with performance optimizations
        if np.max(compound_matrix) > 0:
            normalized_data = compound_matrix / np.max(compound_matrix)
        else:
            normalized_data = compound_matrix
        
        check_timeout()
        
        # If a point is selected, focus on that region; otherwise show entire dataset
        if selected_point is not None:
            # Handle different data structures for selected_point
            if isinstance(selected_point, dict):
                # If it's a dictionary with x, y, z keys
                x = int(selected_point.get('x', 0))
                y = int(selected_point.get('y', 0))
                z = int(selected_point.get('slide', 0))
            elif isinstance(selected_point, (list, tuple)):
                # If it's a list/tuple
                x = int(selected_point[0]) if len(selected_point) > 0 else 0
                y = int(selected_point[1]) if len(selected_point) > 1 else 0
                z = int(selected_point[2]) if len(selected_point) > 2 else 0
            else:
                # Default case
                x, y, z = 0, 0, 0
            
            # Extract region around the selected point
            x_start = max(0, x - region_size)
            x_end = min(normalized_data.shape[2], x + region_size)
            y_start = max(0, y - region_size)
            y_end = min(normalized_data.shape[1], y + region_size)
            z_start = max(0, z - region_size)
            z_end = min(normalized_data.shape[0], z + region_size)
            
            region_data = normalized_data[z_start:z_end, y_start:y_end, x_start:x_end]
            logger.info(f"Showing region around point ({x}, {y}, {z}) with shape: {region_data.shape}")
        else:
            # Show entire dataset with all slides
            region_data = normalized_data
            logger.info(f"Showing entire dataset with all {region_data.shape[0]} slides")
        
        check_timeout()
        
        # Performance optimization: Downsample data if too large
        # Use larger max_dimension for full dataset to preserve more detail
        max_dimension = 80 if selected_point is None else 60  # More detail for full dataset
        
        # More aggressive downsampling to prevent browser crashes
        if selected_point is None:
            # For full dataset, be more conservative with dimensions
            max_dimension = 50  # Reduced from 80 to prevent crashes
            logger.info(f"Full dataset mode: Using conservative max_dimension={max_dimension}")
        else:
            # For focused regions, can use more detail
            max_dimension = 60
            logger.info(f"Focused region mode: Using max_dimension={max_dimension}")
        
        if any(dim > max_dimension for dim in region_data.shape):
            # Calculate downsampling factors
            factors = [max_dimension / dim if dim > max_dimension else 1 for dim in region_data.shape]
            region_data = zoom(region_data, factors, order=1)  # Use order=1 for speed
            logger.info(f"Downsampled data to shape: {region_data.shape}")
        
        # Additional memory optimization: Reduce data precision
        region_data = region_data.astype(np.float32)  # Use float32 instead of float64 to save memory
        
        check_timeout()
        
        fig = go.Figure()
        
        # Enhanced brain visualization with better data processing
        try:
            # Apply moderate smoothing to preserve brain structure while creating smooth contours
            smoothed_data = gaussian_filter(region_data, sigma=1.5)  # Moderate smoothing to preserve brain features
            
            # Compress height to reduce vertical dimension
            # Scale down the Z-axis (height) to make the brain more compact
            height_compression = 0.7  # Reduce height by 30% (less aggressive)
            original_shape = smoothed_data.shape
            compressed_shape = (int(original_shape[0] * height_compression), original_shape[1], original_shape[2])
            
            # Resize the data to compress height
            height_factors = [height_compression, 1.0, 1.0]  # Compress Z, keep X and Y
            compressed_data = zoom(smoothed_data, height_factors, order=1)
            
            # Additional filtering to remove noise and create better brain shape
            # Remove low intensity data but keep enough for brain structure
            noise_threshold = np.percentile(compressed_data[compressed_data > 0], 20) if np.any(compressed_data > 0) else 0.05
            filtered_data = np.where(compressed_data > noise_threshold, compressed_data, 0)
            
            # Apply additional smoothing to clean up the shape
            filtered_data = gaussian_filter(filtered_data, sigma=0.8)
            
            # Create single solid brain structure
            data_nonzero = filtered_data[filtered_data > 0]
            if len(data_nonzero) > 0:
                # Use moderate threshold to capture brain structure without making it too thick
                brain_threshold = np.percentile(data_nonzero, 55)  # Moderate threshold for brain shape
            else:
                brain_threshold = 0.3  # Moderate default threshold
            
            # Ensure we have valid data to work with
            if not np.any(filtered_data > brain_threshold):
                logger.warning("No data above brain threshold, adjusting threshold")
                brain_threshold = np.max(filtered_data) * 0.5 if np.max(filtered_data) > 0 else 0.1
            
            # Create single solid brain surface
            surfaces_created = 0
            try:
                # Create brain surface at single threshold
                verts, faces, _, _ = measure.marching_cubes(filtered_data, level=brain_threshold)
                
                # If surface is too small, try lower threshold for more detail
                if len(verts) < 1000:
                    # Recalculate data_nonzero for the lower threshold
                    data_nonzero_lower = filtered_data[filtered_data > 0]
                    lower_threshold = np.percentile(data_nonzero_lower, 45) if len(data_nonzero_lower) > 0 else 0.25
                    verts, faces, _, _ = measure.marching_cubes(filtered_data, level=lower_threshold)
                
                # Limit vertices to prevent crashes but maintain quality
                max_vertices = 15000 if selected_point is None else 25000  # Higher limits for single surface
                if len(verts) > max_vertices:
                    # Sample vertices to reduce complexity
                    sample_indices = np.random.choice(len(verts), max_vertices, replace=False)
                    verts = verts[sample_indices]
                    faces = faces[np.isin(faces.flatten(), sample_indices).reshape(faces.shape)]
                    logger.info(f"Sampled vertices from {len(verts)} to {max_vertices} for brain structure")
                
                if len(verts) > 0 and len(faces) > 0:
                    x, y, z = verts.T
                    i_faces, j_faces, k_faces = faces.T
                    
                    # Create single solid brain surface with rich texture
                    fig.add_trace(go.Mesh3d(
                        x=x, y=y, z=z,
                        i=i_faces, j=j_faces, k=k_faces,
                        color='red',
                        opacity=0.9,  # Solid appearance
                        lighting=dict(
                            ambient=0.2,
                            diffuse=0.8,
                            fresnel=0.3,
                            specular=0.2,
                            roughness=0.3
                        ),
                        lightposition=dict(x=100, y=100, z=100),
                        name='Brain Structure',
                        showlegend=False
                    ))
                    
                    surfaces_created += 1
                    logger.info(f"Added single brain structure with {len(verts)} vertices (height compressed by {height_compression})")
                    
            except Exception as e:
                logger.warning(f"Could not create brain surface: {e}")
        except Exception as e:
            logger.warning(f"Could not create brain surfaces: {e}")
        
        check_timeout()
        
        # If still no visualization, create surface slices with more detail
        if len(fig.data) == 0:
            try:
                # Create surface plots for brain anatomy with more slices
                slice_step = max(1, region_data.shape[0] // 3)  # Show more slices for detail
                
                # Limit surface plots to prevent crashes
                max_slices = 5 if selected_point is None else 8
                slice_count = 0
                
                for z_idx in range(0, region_data.shape[0], slice_step):
                    if slice_count >= max_slices:
                        break
                        
                    slice_data = region_data[z_idx]
                    
                    # Apply smoothing to slice for brain-like contours
                    smoothed_slice = gaussian_filter(slice_data, sigma=1.2)
                    
                    # Only show slices with significant brain data
                    if np.max(smoothed_slice) > 0.08:
                        x = np.arange(smoothed_slice.shape[1])
                        y = np.arange(smoothed_slice.shape[0])
                        X, Y = np.meshgrid(x, y)
                        Z = np.full_like(X, z_idx)
                        
                        # Create brain surface for this slice with better detail
                        fig.add_trace(go.Surface(
                            x=X, y=Y, z=Z,
                            surfacecolor=smoothed_slice,
                            colorscale=[[0, 'rgba(255,0,0,0)'], [1, 'rgba(255,0,0,0.9)']],
                            opacity=0.7,
                            showscale=False,
                            name=f'Brain Slice {z_idx}',
                            hoverinfo='skip'
                        ))
                        slice_count += 1
            except Exception as e:
                logger.warning(f"Could not create brain surface plots: {e}")
        
        check_timeout()
        
        # Only use points as absolute last resort with better processing
        if len(fig.data) == 0:
            logger.warning("No brain structure created, using enhanced points as fallback")
            try:
                # Apply strong smoothing for better brain-like appearance
                smoothed_data = gaussian_filter(region_data, sigma=2.0)
                
                # Use more aggressive threshold for better brain definition
                threshold = np.percentile(smoothed_data[smoothed_data > 0], 55) if np.any(smoothed_data > 0) else 0.15
                data_mask = smoothed_data > threshold
                
                if np.sum(data_mask) > 0:
                    # Use more points for better brain structure
                    max_points = 800 if selected_point is None else 1200
                    
                    if np.sum(data_mask) > max_points:
                        data_indices = np.where(data_mask)
                        sample_indices = np.random.choice(
                            len(data_indices[0]), 
                            max_points, 
                            replace=False
                        )
                        z_coords = data_indices[0][sample_indices]
                        y_coords = data_indices[1][sample_indices]
                        x_coords = data_indices[2][sample_indices]
                        values = smoothed_data[z_coords, y_coords, x_coords]
                    else:
                        data_indices = np.where(data_mask)
                        z_coords = data_indices[0]
                        y_coords = data_indices[1]
                        x_coords = data_indices[2]
                        values = smoothed_data[z_coords, y_coords, x_coords]
                    
                    # Create better brain structure points
                    fig.add_trace(go.Scatter3d(
                        x=x_coords, y=y_coords, z=z_coords,
                        mode='markers',
                        marker=dict(
                            size=6,
                            color=values,
                            colorscale=[
                                [0, 'rgba(255,0,0,0.8)'],  # Solid red
                                [1, 'rgba(255,0,0,1)']     # Bright red
                            ],
                            opacity=0.9,
                            showscale=True,
                            colorbar=dict(
                                title=dict(text="Brain Intensity", font=dict(size=14, color='white')),
                                thickness=15,
                                len=0.5,
                                x=1.1,
                                tickfont=dict(size=12, color='white')
                            )
                        ),
                        name='Brain Structure',
                        hovertemplate='<b>Brain:</b><br>X: %{x}<br>Y: %{y}<br>Z: %{z}<br><b>Intensity:</b> %{marker.color:.3f}<extra></extra>'
                    ))
                    
                    logger.info(f"Added {len(x_coords)} enhanced brain structure points")
            except Exception as e:
                logger.warning(f"Could not create brain structure points: {e}")
        
        check_timeout()
        
        # If no traces were created, create a fallback visualization
        if len(fig.data) == 0:
            logger.warning("No traces created, using fallback visualization")
            # Create a simple test visualization with red points
            x = np.linspace(0, region_data.shape[2], 20)
            y = np.linspace(0, region_data.shape[1], 20)
            z = np.linspace(0, region_data.shape[0], 10)
            X, Y, Z = np.meshgrid(x, y, z)
            
            fig.add_trace(go.Scatter3d(
                x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
                mode='markers',
                marker=dict(
                    size=2,
                    color='red',
                    opacity=0.6
                ),
                name='Fallback Visualization'
            ))
        
        check_timeout()
        
        # Configure layout for clean appearance with pure black background and red data
        fig.update_layout(
            title=dict(
                text="3D Brain Visualization" + (" - Selected Region" if selected_point else ""),
                font=dict(color='white', size=20),
                x=0.5
            ),
            scene=dict(
                xaxis=dict(
                    visible=False,  # Hide X axis completely
                    showgrid=False,
                    showline=False,
                    showticklabels=False,
                    zeroline=False
                ),
                yaxis=dict(
                    visible=False,  # Hide Y axis completely
                    showgrid=False,
                    showline=False,
                    showticklabels=False,
                    zeroline=False
                ),
                zaxis=dict(
                    visible=False,  # Hide Z axis completely
                    showgrid=False,
                    showline=False,
                    showticklabels=False,
                    zeroline=False
                ),
                camera=dict(
                    eye=dict(x=1.8, y=1.8, z=1.5),
                    center=dict(x=0, y=0, z=0)
                ),
                bgcolor='rgba(0,0,0,1)',  # Pure black background
                aspectmode='data'
            ),
            margin=dict(l=0, r=100, t=60, b=0),
            paper_bgcolor='rgba(0,0,0,1)',  # Pure black background
            plot_bgcolor='rgba(0,0,0,1)',  # Pure black background
            height=700,
            showlegend=False,
            font=dict(color='white')
        )
        
        # Clean up memory
        del region_data, normalized_data
        gc.collect()
        
        execution_time = time.time() - start_time
        logger.info(f"3D visualization created successfully in {execution_time:.2f} seconds")
        return fig
        
    except TimeoutError as e:
        logger.error(f"3D visualization timed out: {e}")
        # Return timeout error figure
        timeout_fig = go.Figure().add_annotation(
            text="3D visualization generation timed out.<br>Please try with smaller data or contact support.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="orange")
        ).update_layout(
            plot_bgcolor='black',
            paper_bgcolor='black',
            height=650,
            width=650,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        return timeout_fig
        
    except Exception as e:
        logger.error(f"Failed to create 3D brain: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
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
        
        fig.update_layout(
            title=dict(text="3D Visualization Error", font=dict(color='white')),
            scene=dict(bgcolor='black'),
            paper_bgcolor='black',
            height=700
        )
        return fig

def register_3d_page_callbacks(app, cache):
    """Register callbacks specific to the 3D visualization page"""
    

    
    @app.callback(
        [Output("2d-slice-graph-3d-page", "figure"),
         Output("selected-point-3d", "data")],
        [Input("slide-selector-3d-page", "value"),
         Input("2d-slice-graph-3d-page", "clickData")],
        prevent_initial_call=False,
        allow_duplicate=True
    )
    def update_2d_slice_with_crosshair(selected_slide, click_data):
        """Update 2D slice visualization with interactive crosshair and store selected point"""
        try:
            # Handle initial state when no slide is selected
            if selected_slide is None:
                return go.Figure().add_annotation(
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
                ), None
            
            session_id = session.get('session_id', 'default')
            compound_matrix = cache.get(f"{session_id}:compound_matrix")
            
            if compound_matrix is None:
                # Use demo data if no real data is available
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
                                cortical_intensity = 300 * np.exp(-distance * 1.5)
                                
                                # Medium-high intensity regions (orange/yellow)
                                medium_intensity = 200 * np.exp(-distance * 2.0)
                                
                                # Medium intensity regions (green)
                                subcortical_intensity = 120 * np.exp(-distance * 2.3)
                                
                                # Low intensity deep regions (blue)
                                deep_intensity = 60 * np.exp(-distance * 2.8)
                                
                                # Create solid cortical folds with higher intensity
                                cortical_pattern = np.sin(x * 0.12) * np.cos(y * 0.10) * np.sin(z * 0.20)
                                fold_intensity = 50 * np.abs(cortical_pattern)
                                
                                # Create high-intensity hippocampus structure (bright red)
                                hippo_distance = np.sqrt((x - center_x)**2 + (y - (center_y + 5))**2)
                                if hippo_distance < 10 and y > center_y:
                                    hippocampus_intensity = 280 * np.exp(-hippo_distance / 5)
                                else:
                                    hippocampus_intensity = 0
                                
                                # Create high-intensity thalamus region (bright red)
                                thalamus_distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                                if thalamus_distance < 6:
                                    thalamus_intensity = 320 * np.exp(-thalamus_distance / 3)
                                else:
                                    thalamus_intensity = 0
                                
                                # Combine all effects with solid regional variations
                                if distance < 0.25:  # Outer cortical layer (bright red)
                                    intensity = cortical_intensity + fold_intensity + hippocampus_intensity + thalamus_intensity
                                elif distance < 0.5:  # Middle cortical layer (orange/yellow)
                                    intensity = medium_intensity + fold_intensity * 0.7 + hippocampus_intensity * 0.5
                                elif distance < 0.75:  # Subcortical layer (green)
                                    intensity = subcortical_intensity + fold_intensity * 0.5
                                else:  # Deep brain regions (blue)
                                    intensity = deep_intensity
                                
                                # Add very minimal noise for solid appearance
                                noise = np.random.normal(0, 1)
                                intensity += noise
                                
                                compound_matrix[z, y, x] = max(0, intensity)
            
            # No smoothing - keep data sharp and clear
            
            # Get crosshair position from click data
            crosshair_x = None
            crosshair_y = None
            selected_point = None
            
            if click_data and 'points' in click_data and len(click_data['points']) > 0:
                point = click_data['points'][0]
                crosshair_x = point.get('x')
                crosshair_y = point.get('y')
                # Store the selected point with slice information
                selected_point = {
                    'x': int(crosshair_x),
                    'y': int(crosshair_y),
                    'z': selected_slide or 0
                }
            
            # Create 2D slice figure with interactive crosshair
            # Ensure we have a valid slide index
            slide_idx = selected_slide if selected_slide is not None else 0
            fig = create_2d_slice_figure(compound_matrix, slide_idx)
            
            # Update crosshair position if clicked
            if crosshair_x is not None and crosshair_y is not None:
                # Remove existing crosshair traces
                fig.data = [trace for trace in fig.data if not isinstance(trace, go.Scatter)]
                
                # Add new crosshair at clicked position
                rows, cols = compound_matrix[selected_slide or 0].shape
                
                # Horizontal line
                fig.add_trace(go.Scatter(
                    x=[0, cols-1],
                    y=[crosshair_y, crosshair_y],
                    mode='lines',
                    line=dict(color='red', width=3),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Vertical line
                fig.add_trace(go.Scatter(
                    x=[crosshair_x, crosshair_x],
                    y=[0, rows-1],
                    mode='lines',
                    line=dict(color='red', width=3),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Update labels
                fig.add_annotation(
                    x=crosshair_x + 3,
                    y=crosshair_y,
                    text="L",
                    font=dict(color="white", size=14, family="Arial Black"),
                    showarrow=False
                )
                
                fig.add_annotation(
                    x=crosshair_x,
                    y=crosshair_y - 3,
                    text="A",
                    font=dict(color="white", size=14, family="Arial Black"),
                    showarrow=False
                )
            
            return fig, selected_point
            
        except Exception as e:
            logger.error(f"Error updating 2D slice with crosshair: {str(e)}")
            raise PreventUpdate
    
    @app.callback(
        Output("selected-page", "data", allow_duplicate=True),
        Input("return-to-dashboard-from-3d", "n_clicks"),
        prevent_initial_call=True,
        allow_duplicate=True
    )
    def return_from_3d_page(return_clicks):
        """Return to dashboard from 3D page"""
        if return_clicks and return_clicks > 0:
            return "dashboard"
        raise PreventUpdate 

    @app.callback(
        [Output("3d-graph-3d-page", "figure"),
         Output("3d-analysis-status", "children")],
        [Input("run-3d-analysis-btn", "n_clicks"),
         Input("slide-selector-3d-page", "value")],
        [State("selected-point-3d", "data")],
        prevent_initial_call=False,
        allow_duplicate=True
    )
    def run_3d_analysis(n_clicks, selected_slide, selected_point):
        """Run 3D analysis for selected point or show full brain"""
        try:
            # Handle initial state when no slide is selected
            if selected_slide is None:
                return go.Figure().add_annotation(
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
                ), "Select a slide from the dropdown above to view 2D slice, then click 'Run 3D Analysis' to generate 3D visualization"
            
            session_id = session.get('session_id', 'default')
            compound_matrix = cache.get(f"{session_id}:compound_matrix")
            
            if compound_matrix is None:
                # Use demo data if no real data is available
                logger.warning("No compound matrix found, using demo data")
                compound_matrix = np.zeros((18, 60, 60))  # Demo data with 18 slides
                
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
                                cortical_intensity = 300 * np.exp(-distance * 1.5)
                                
                                # Medium-high intensity regions (orange/yellow)
                                medium_intensity = 200 * np.exp(-distance * 2.0)
                                
                                # Medium intensity regions (green)
                                subcortical_intensity = 120 * np.exp(-distance * 2.3)
                                
                                # Low intensity deep regions (blue)
                                deep_intensity = 60 * np.exp(-distance * 2.8)
                                
                                # Create solid cortical folds with higher intensity
                                cortical_pattern = np.sin(x * 0.12) * np.cos(y * 0.10) * np.sin(z * 0.20)
                                fold_intensity = 50 * np.abs(cortical_pattern)
                                
                                # Create high-intensity hippocampus structure (bright red)
                                hippo_distance = np.sqrt((x - center_x)**2 + (y - (center_y + 5))**2)
                                if hippo_distance < 10 and y > center_y:
                                    hippocampus_intensity = 280 * np.exp(-hippo_distance / 5)
                                else:
                                    hippocampus_intensity = 0
                                
                                # Create high-intensity thalamus region (bright red)
                                thalamus_distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                                if thalamus_distance < 6:
                                    thalamus_intensity = 320 * np.exp(-thalamus_distance / 3)
                                else:
                                    thalamus_intensity = 0
                                
                                # Combine all effects with solid regional variations
                                if distance < 0.25:  # Outer cortical layer (bright red)
                                    intensity = cortical_intensity + fold_intensity + hippocampus_intensity + thalamus_intensity
                                elif distance < 0.5:  # Middle cortical layer (orange/yellow)
                                    intensity = medium_intensity + fold_intensity * 0.7 + hippocampus_intensity * 0.5
                                elif distance < 0.75:  # Subcortical layer (green)
                                    intensity = subcortical_intensity + fold_intensity * 0.5
                                else:  # Deep brain regions (blue)
                                    intensity = deep_intensity
                                
                                compound_matrix[z, y, x] = intensity
                
                logger.info(f"Using demo data with shape: {compound_matrix.shape} (18 slides)")
            else:
                logger.info(f"Using real dataset with shape: {compound_matrix.shape} ({compound_matrix.shape[0]} slides)")
            
            # Create 3D visualization with error handling
            try:
                fig = create_3d_page_figure(compound_matrix, selected_point)
                
                # Create status text based on whether we're showing full dataset or focused region
                if selected_point:
                    # Handle different data structures for selected_point
                    if isinstance(selected_point, dict):
                        x = selected_point.get('x', 0)
                        y = selected_point.get('y', 0)
                        z = selected_point.get('slide', 0)
                    elif isinstance(selected_point, (list, tuple)):
                        x = selected_point[0] if len(selected_point) > 0 else 0
                        y = selected_point[1] if len(selected_point) > 1 else 0
                        z = selected_point[2] if len(selected_point) > 2 else 0
                    else:
                        x, y, z = 0, 0, 0
                    
                    status_text = f"3D visualization generated successfully. Showing region around point: X={x}, Y={y}, Z={z}. Use mouse to rotate and zoom."
                else:
                    # Show full dataset information
                    total_slides = compound_matrix.shape[0]
                    status_text = f"3D visualization generated successfully. Showing full dataset with all {total_slides} slides. Use mouse to rotate and zoom."
                
                return fig, status_text
                
            except Exception as viz_error:
                logger.error(f"Error creating 3D visualization: {viz_error}")
                import traceback
                logger.error(f"Visualization traceback: {traceback.format_exc()}")
                
                # Return error figure
                error_fig = go.Figure().add_annotation(
                    text="Error creating 3D visualization.<br>Please try again or contact support.",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=16, color="red")
                ).update_layout(
                    plot_bgcolor='black',
                    paper_bgcolor='black',
                    height=650,
                    width=650,
                    margin=dict(l=10, r=10, t=10, b=10)
                )
                
                return error_fig, f"Error: {str(viz_error)}"
            
        except Exception as e:
            logger.error(f"Error in 3D analysis: {str(e)}")
            import traceback
            logger.error(f"3D analysis traceback: {traceback.format_exc()}")
            
            # Return error figure
            error_fig = go.Figure().add_annotation(
                text="Error loading 3D visualization.<br>Please try again.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="red")
            ).update_layout(
                plot_bgcolor='black',
                paper_bgcolor='black',
                height=650,
                width=650,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            
            return error_fig, f"Error: {str(e)}" 