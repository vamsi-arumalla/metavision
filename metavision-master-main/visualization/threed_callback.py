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

class MetaAtlas3D:
    def __init__(self, matrix, resolution, thickness, gap, insert):
        self.matrix = matrix
        self.resolution = resolution
        self.thickness = thickness
        self.gap = gap
        self.insert = insert
        
    def create_nii(self):
        # reshape matrix from (z,y,x) to (x,y,z)
        output_matrix = np.transpose(self.matrix,(2,1,0))
        Ir = np.squeeze(output_matrix)
        img = nib.Nifti1Image(Ir, affine=np.eye(4))
        # adjust thickness
        header = img.header.copy()
        header['pixdim'][3] = (self.thickness+self.gap)/((1+self.insert)*self.resolution)
    
        nii_img = nib.Nifti1Image(Ir, affine=np.eye(4), header=header)
        return nii_img

def create_nii(matrix, resolution, thickness, gap, insert=0):
    # resolution: MALDI resolution (x-y axis); section thickness; gap between adjacent sections; number of inserted sections
    # reshape matrix from (z,y,x) to (x,y,z)
    output_matrix = np.transpose(matrix,(2,1,0))
    Ir = np.squeeze(output_matrix)
    img = nib.Nifti1Image(Ir, affine=np.eye(4))
    # adjust thickness
    header = img.header.copy()
    header['pixdim'][3] = (thickness+gap)/((1+insert)*resolution)

    nii_img = nib.Nifti1Image(Ir, affine=np.eye(4), header=header)
    return nii_img

def create_advanced_3d_figure(compound_matrix, projection_type, thickness, gap, max_projection):
    """
    Creates an optimized 3D visualization with performance improvements
    """
    try:
        print(f"[DEBUG] Creating 3D visualization with projection_type: {projection_type}")
        print(f"[DEBUG] Matrix shape: {compound_matrix.shape}")
        print(f"[DEBUG] Matrix stats - min: {np.min(compound_matrix)}, max: {np.max(compound_matrix)}, mean: {np.mean(compound_matrix)}")
        
        slices, rows, cols = compound_matrix.shape
        logger.info(f"Matrix shape: {compound_matrix.shape}")
        logger.info(f"Matrix min/max: {np.min(compound_matrix)}/{np.max(compound_matrix)}")
        
        # Performance optimization: Downsample data if too large
        max_dimension = 60  # Limit maximum dimension for performance
        if any(dim > max_dimension for dim in compound_matrix.shape):
            # Calculate downsampling factors
            factors = [max_dimension / dim if dim > max_dimension else 1 for dim in compound_matrix.shape]
            compound_matrix = zoom(compound_matrix, factors, order=1)  # Use order=1 for speed
            slices, rows, cols = compound_matrix.shape
            logger.info(f"Downsampled data to shape: {compound_matrix.shape}")
        
        # Enhanced data preprocessing with better threshold handling
        if projection_type == 'maximum':
            # Create maximum intensity projection with improved thresholding
            threshold_percentile = max(max_projection, 50)  # Ensure minimum 50th percentile
            threshold = np.percentile(compound_matrix[compound_matrix > 0], threshold_percentile) if np.any(compound_matrix > 0) else 0
            data = np.maximum(compound_matrix, threshold * 0.1)  # Use 10% of threshold as baseline
            print(f"[DEBUG] Using maximum projection with threshold: {threshold}")
        else:
            # Use original data with noise reduction
            data = compound_matrix.copy()
            # Remove very small values that might be noise
            noise_threshold = np.percentile(data[data > 0], 5) if np.any(data > 0) else 0
            data[data < noise_threshold] = 0
            print(f"[DEBUG] Using original data with noise threshold: {noise_threshold}")
        
        # Enhanced normalization with better handling of edge cases
        data_nonzero = data[data > 0]
        if len(data_nonzero) > 0:
            data_min = np.percentile(data_nonzero, 1)  # Use 1st percentile instead of absolute min
            data_max = np.percentile(data_nonzero, 99)  # Use 99th percentile instead of absolute max
            if data_max > data_min:
                normalized_data = np.clip((data - data_min) / (data_max - data_min), 0, 1)
            else:
                normalized_data = data / data_max if data_max > 0 else data
        else:
            normalized_data = data
            
        print(f"[DEBUG] Normalized data stats - min: {np.min(normalized_data)}, max: {np.max(normalized_data)}, nonzero count: {np.count_nonzero(normalized_data)}")
        
        # Create multiple visualization layers
        traces = []
        
        # 1. Simple scatter plot for high-intensity points (much faster than marching cubes)
        try:
            # Find high-intensity points with efficient thresholding
            high_intensity_threshold = np.percentile(normalized_data, 85)  # Higher threshold for fewer points
            high_intensity_mask = normalized_data > high_intensity_threshold
            
            if np.sum(high_intensity_mask) > 0:
                # Sample points to avoid overcrowding and improve performance
                max_points = 1500  # Limit total points for performance
                
                if np.sum(high_intensity_mask) > max_points:
                    # Randomly sample high-intensity points
                    high_intensity_indices = np.where(high_intensity_mask)
                    sample_indices = np.random.choice(
                        len(high_intensity_indices[0]), 
                        max_points, 
                        replace=False
                    )
                    z_coords = high_intensity_indices[0][sample_indices]
                    y_coords = high_intensity_indices[1][sample_indices]
                    x_coords = high_intensity_indices[2][sample_indices]
                    values = normalized_data[z_coords, y_coords, x_coords]
                else:
                    high_intensity_indices = np.where(high_intensity_mask)
                    z_coords = high_intensity_indices[0]
                    y_coords = high_intensity_indices[1]
                    x_coords = high_intensity_indices[2]
                    values = normalized_data[z_coords, y_coords, x_coords]
                
                # Create scatter plot with bright red and white color scheme
                scatter_trace = go.Scatter3d(
                    x=x_coords, y=y_coords, z=z_coords,
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=values,
                        colorscale=[
                            [0, 'rgba(255,255,255,0.3)'],  # White for low intensity
                            [0.5, 'rgba(255,100,100,0.7)'],  # Light red
                            [1, 'rgba(255,0,0,1)']  # Bright red for high intensity
                        ],
                        opacity=0.8,
                        showscale=True,
                        colorbar=dict(
                            title=dict(text="Intensity", font=dict(size=14, color='white')),
                            thickness=15,
                            len=0.5,
                            x=1.1,
                            tickfont=dict(size=12, color='white')
                        )
                    ),
                    name='High Intensity Points',
                    showlegend=True,
                    hovertemplate='<b>Position:</b><br>X: %{x}<br>Y: %{y}<br>Z: %{z}<br><b>Intensity:</b> %{marker.color:.3f}<extra></extra>'
                )
                traces.append(scatter_trace)
                logger.info(f"Added {len(x_coords)} high-intensity points")
        except Exception as e:
            logger.warning(f"Could not create scatter plot: {e}")
        
        # 2. Wireframe outline for context (lightweight)
        try:
            # Create simple wireframe of the data bounds
            x_range = np.linspace(0, cols-1, 8)
            y_range = np.linspace(0, rows-1, 8)
            z_range = np.linspace(0, slices-1, 8)
            
            # Create wireframe lines
            wireframe_x = []
            wireframe_y = []
            wireframe_z = []
            
            # X-direction lines
            for y in y_range[::2]:
                for z in z_range[::2]:
                    wireframe_x.extend([0, cols-1, None])
                    wireframe_y.extend([y, y, None])
                    wireframe_z.extend([z, z, None])
            
            # Y-direction lines
            for x in x_range[::2]:
                for z in z_range[::2]:
                    wireframe_x.extend([x, x, None])
                    wireframe_y.extend([0, rows-1, None])
                    wireframe_z.extend([z, z, None])
            
            # Z-direction lines
            for x in x_range[::2]:
                for y in y_range[::2]:
                    wireframe_x.extend([x, x, None])
                    wireframe_y.extend([y, y, None])
                    wireframe_z.extend([0, slices-1, None])
            
            wireframe_trace = go.Scatter3d(
                x=wireframe_x, y=wireframe_y, z=wireframe_z,
                mode='lines',
                line=dict(color='rgba(255,255,255,0.2)', width=1),
                name='Data Bounds',
                showlegend=True
            )
            traces.append(wireframe_trace)
        except Exception as e:
            logger.warning(f"Could not create wireframe: {e}")
        
        # If no traces were created, create a fallback visualization
        if not traces:
            logger.warning("No traces created, using fallback visualization")
            # Create a simple scatter plot of non-zero points
            non_zero_mask = normalized_data > 0.1
            if np.sum(non_zero_mask) > 0:
                non_zero_indices = np.where(non_zero_mask)
                # Sample points
                max_points = 500
                if len(non_zero_indices[0]) > max_points:
                    sample_indices = np.random.choice(len(non_zero_indices[0]), max_points, replace=False)
                    z_coords = non_zero_indices[0][sample_indices]
                    y_coords = non_zero_indices[1][sample_indices]
                    x_coords = non_zero_indices[2][sample_indices]
                    values = normalized_data[z_coords, y_coords, x_coords]
                else:
                    z_coords = non_zero_indices[0]
                    y_coords = non_zero_indices[1]
                    x_coords = non_zero_indices[2]
                    values = normalized_data[z_coords, y_coords, x_coords]
                
                fallback_trace = go.Scatter3d(
                    x=x_coords, y=y_coords, z=z_coords,
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=values,
                        colorscale=[
                            [0, 'rgba(255,255,255,0.3)'],  # White for low intensity
                            [0.5, 'rgba(255,100,100,0.7)'],  # Light red
                            [1, 'rgba(255,0,0,1)']  # Bright red for high intensity
                        ],
                        opacity=0.8,
                        showscale=True,
                        colorbar=dict(
                            title=dict(text="Intensity", font=dict(size=14, color='white')),
                            thickness=15,
                            len=0.5,
                            x=1.1,
                            tickfont=dict(size=12, color='white')
                        )
                    ),
                    name='Data Points'
                )
                traces.append(fallback_trace)
        
        # Create the figure
        fig = go.Figure(data=traces)
        
        # Enhanced layout with bright red and white theme
        fig.update_layout(
            title=dict(
                text="3D Brain Visualization",
                font=dict(size=24, color='white'),
                x=0.5
            ),
            scene=dict(
                xaxis=dict(
                    title=dict(text="X (Columns)", font=dict(size=16, color='white')),
                    tickfont=dict(size=12, color='white'),
                    gridcolor='rgba(255,255,255,0.1)',
                    zerolinecolor='rgba(255,255,255,0.2)'
                ),
                yaxis=dict(
                    title=dict(text="Y (Rows)", font=dict(size=16, color='white')),
                    tickfont=dict(size=12, color='white'),
                    gridcolor='rgba(255,255,255,0.1)',
                    zerolinecolor='rgba(255,255,255,0.2)'
                ),
                zaxis=dict(
                    title=dict(text="Z (Slices)", font=dict(size=16, color='white')),
                    tickfont=dict(size=12, color='white'),
                    gridcolor='rgba(255,255,255,0.1)',
                    zerolinecolor='rgba(255,255,255,0.2)'
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2),
                    center=dict(x=0, y=0, z=0)
                ),
                aspectmode='data',
                bgcolor='rgba(0,0,0,0.9)'
            ),
            height=800,
            margin=dict(r=100, l=50, b=50, t=80),
            showlegend=True,
            legend=dict(
                x=1.02,
                y=0.5,
                bgcolor='rgba(0,0,0,0.8)',
                bordercolor='rgba(255,255,255,0.3)',
                borderwidth=1,
                font=dict(color='white')
            ),
            paper_bgcolor='rgba(0,0,0,0.9)',
            plot_bgcolor='rgba(0,0,0,0.9)'
        )
        
        logger.info("3D visualization created successfully")
        return fig
        
    except Exception as e:
        logger.error(f"Error in create_advanced_3d_figure: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Return a simple test figure
        fig = go.Figure(data=go.Scatter3d(
            x=[1, 2, 3, 4, 5],
            y=[1, 2, 3, 4, 5],
            z=[1, 2, 3, 4, 5],
            mode='markers',
            marker=dict(
                size=10,
                color=[1, 2, 3, 4, 5],
                colorscale=[
                    [0, 'rgba(255,255,255,0.3)'],  # White for low intensity
                    [0.5, 'rgba(255,100,100,0.7)'],  # Light red
                    [1, 'rgba(255,0,0,1)']  # Bright red for high intensity
                ]
            )
        ))
        fig.update_layout(
            title="Test 3D Visualization",
            scene=dict(
                xaxis=dict(title="X"),
                yaxis=dict(title="Y"),
                zaxis=dict(title="Z")
            ),
            height=700
        )
        return fig

def create_simple_3d_figure(compound_matrix, projection_type, thickness, gap, max_projection):
    """
    Creates an optimized simple 3D visualization using scatter plots
    """
    try:
        slices, rows, cols = compound_matrix.shape
        logger.info(f"Matrix shape: {compound_matrix.shape}")
        
        # Performance optimization: Downsample data if too large
        max_dimension = 60  # Limit maximum dimension for performance
        if any(dim > max_dimension for dim in compound_matrix.shape):
            # Calculate downsampling factors
            factors = [max_dimension / dim if dim > max_dimension else 1 for dim in compound_matrix.shape]
            compound_matrix = zoom(compound_matrix, factors, order=1)  # Use order=1 for speed
            slices, rows, cols = compound_matrix.shape
            logger.info(f"Downsampled data to shape: {compound_matrix.shape}")
        
        # Simple threshold
        threshold = 0.1
        if projection_type == 'maximum':
            threshold = np.percentile(compound_matrix, max_projection) / 100
        
        # Normalize data
        max_val = np.max(compound_matrix)
        if max_val > 0:
            normalized_data = compound_matrix / max_val
        else:
            normalized_data = compound_matrix
        
        # Sample points for visualization with performance optimization
        points_x = []
        points_y = []
        points_z = []
        point_values = []
        
        # Sample every 4th slice for better performance
        slice_step = max(1, slices // 15)  # Limit to 15 slices maximum
        
        for z_idx in range(0, slices, slice_step):
            slice_data = normalized_data[z_idx]
            mask = slice_data > threshold
            y_coords, x_coords = np.nonzero(mask)
            values = slice_data[mask]
            
            if len(x_coords) > 0:
                # Limit points per slice for performance
                max_points_per_slice = 200  # Reduced from 300
                if len(x_coords) > max_points_per_slice:
                    indices = np.random.choice(len(x_coords), max_points_per_slice, replace=False)
                    x_coords = x_coords[indices]
                    y_coords = y_coords[indices]
                    values = values[indices]
                
                points_x.extend(x_coords)
                points_y.extend(y_coords)
                points_z.extend([z_idx] * len(x_coords))
                point_values.extend(values)
        
        # Convert to arrays
        points_x = np.array(points_x)
        points_y = np.array(points_y)
        points_z = np.array(points_z)
        point_values = np.array(point_values)
        
        logger.info(f"Total points: {len(points_x)}")
        
        if len(points_x) == 0:
            # Create a simple test visualization if no points
            x = np.linspace(0, cols, 30)  # Reduced from 50
            y = np.linspace(0, rows, 30)  # Reduced from 50
            z = np.linspace(0, slices, 15)  # Reduced from 50
            X, Y, Z = np.meshgrid(x, y, z)
            points_x = X.flatten()
            points_y = Y.flatten()
            points_z = Z.flatten()
            point_values = np.random.random(len(points_x))
        
        # Create the figure with bright red and white color scheme
        fig = go.Figure(data=go.Scatter3d(
            x=points_x, y=points_y, z=points_z,
            mode='markers',
            marker=dict(
                size=3,
                color=point_values,
                colorscale=[
                    [0, 'rgba(255,255,255,0.3)'],  # White for low intensity
                    [0.5, 'rgba(255,100,100,0.7)'],  # Light red
                    [1, 'rgba(255,0,0,1)']  # Bright red for high intensity
                ],
                opacity=0.8,
                showscale=True,
                colorbar=dict(
                    title=dict(text="Intensity", font=dict(size=14, color='white')),
                    thickness=15,
                    len=0.5,
                    x=1.1,
                    tickfont=dict(size=12, color='white')
                )
            ),
            hoverinfo='x+y+z+text',
            hovertemplate='<b>Position:</b><br>' +
                         'X: %{x}<br>' +
                         'Y: %{y}<br>' +
                         'Z: %{z}<br>' +
                         '<b>Intensity:</b> %{marker.color:.3f}<br>' +
                         '<extra></extra>'
        ))
        
        # Enhanced layout with bright red and white theme
        fig.update_layout(
            title=dict(
                text="3D Brain Visualization (Simple)",
                font=dict(size=20, color='white'),
                x=0.5
            ),
            scene=dict(
                xaxis=dict(
                    title=dict(text="X (Columns)", font=dict(size=14, color='white')),
                    tickfont=dict(size=10, color='white'),
                    gridcolor='rgba(255,255,255,0.1)',
                    zerolinecolor='rgba(255,255,255,0.2)'
                ),
                yaxis=dict(
                    title=dict(text="Y (Rows)", font=dict(size=14, color='white')),
                    tickfont=dict(size=10, color='white'),
                    gridcolor='rgba(255,255,255,0.1)',
                    zerolinecolor='rgba(255,255,255,0.2)'
                ),
                zaxis=dict(
                    title=dict(text="Z (Slices)", font=dict(size=14, color='white')),
                    tickfont=dict(size=10, color='white'),
                    gridcolor='rgba(255,255,255,0.1)',
                    zerolinecolor='rgba(255,255,255,0.2)'
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                ),
                bgcolor='rgba(0,0,0,0.9)'
            ),
            height=700,
            margin=dict(r=100, l=50, b=50, t=50),
            paper_bgcolor='rgba(0,0,0,0.9)',
            plot_bgcolor='rgba(0,0,0,0.9)'
        )
        
        logger.info("Simple 3D visualization created successfully")
        return fig
        
    except Exception as e:
        logger.error(f"Error in create_simple_3d_figure: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Return a simple test figure
        fig = go.Figure(data=go.Scatter3d(
            x=[1, 2, 3, 4, 5],
            y=[1, 2, 3, 4, 5],
            z=[1, 2, 3, 4, 5],
            mode='markers',
            marker=dict(
                size=10,
                color=[1, 2, 3, 4, 5],
                colorscale=[
                    [0, 'rgba(255,255,255,0.3)'],  # White for low intensity
                    [0.5, 'rgba(255,100,100,0.7)'],  # Light red
                    [1, 'rgba(255,0,0,1)']  # Bright red for high intensity
                ]
            )
        ))
        fig.update_layout(
            title="Test 3D Visualization",
            scene=dict(
                xaxis=dict(title="X"),
                yaxis=dict(title="Y"),
                zaxis=dict(title="Z")
            ),
            height=700
        )
        return fig

def create_slice_based_3d_figure(compound_matrix, projection_type, thickness, gap, max_projection):
    """
    Creates a 3D visualization using stacked 2D slices with transparency
    """
    try:
        slices, rows, cols = compound_matrix.shape
        logger.info(f"Creating slice-based 3D visualization for shape: {compound_matrix.shape}")
        
        # Data preprocessing
        if projection_type == 'maximum':
            threshold = np.percentile(compound_matrix, max_projection) / 100
            data = np.maximum(compound_matrix, threshold)
        else:
            data = compound_matrix.copy()
        
        # Normalize data
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max > data_min:
            normalized_data = (data - data_min) / (data_max - data_min)
        else:
            normalized_data = data
        
        traces = []
        
        # Create surface plots for each slice
        slice_step = max(1, slices // 20)  # Show every 20th slice or at least every slice
        
        for z_idx in range(0, slices, slice_step):
            slice_data = normalized_data[z_idx]
            
            # Skip empty slices
            if np.max(slice_data) < 0.01:
                continue
            
            # Create meshgrid for the slice
            x = np.arange(cols)
            y = np.arange(rows)
            X, Y = np.meshgrid(x, y)
            Z = np.full_like(X, z_idx)
            
            # Create surface plot for this slice
            surface_trace = go.Surface(
                x=X, y=Y, z=Z,
                surfacecolor=slice_data,
                colorscale='Viridis',
                opacity=0.3,
                showscale=False,
                name=f'Slice {z_idx}',
                hoverinfo='skip'
            )
            traces.append(surface_trace)
        
        # Add a colorbar trace
        colorbar_trace = go.Surface(
            x=[[0, 1], [0, 1]], y=[[0, 0], [1, 1]], z=[[0, 0], [0, 0]],
            surfacecolor=[[0, 1], [0, 1]],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title="Intensity",
                thickness=15,
                len=0.5,
                x=1.1
            ),
            name='Colorbar'
        )
        traces.append(colorbar_trace)
        
        # Create the figure
        fig = go.Figure(data=traces)
        
        # Layout
        fig.update_layout(
            title=dict(
                text="3D Slice-Based Visualization",
                font=dict(size=24, color='white'),
                x=0.5
            ),
            scene=dict(
                xaxis=dict(
                    title=dict(text="X (Columns)", font=dict(size=16, color='white')),
                    tickfont=dict(size=12, color='white'),
                    gridcolor='rgba(100,100,100,0.3)'
                ),
                yaxis=dict(
                    title=dict(text="Y (Rows)", font=dict(size=16, color='white')),
                    tickfont=dict(size=12, color='white'),
                    gridcolor='rgba(100,100,100,0.3)'
                ),
                zaxis=dict(
                    title=dict(text="Z (Slices)", font=dict(size=16, color='white')),
                    tickfont=dict(size=12, color='white'),
                    gridcolor='rgba(100,100,100,0.3)'
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                ),
                aspectmode='data',
                bgcolor='rgba(0,0,0,0.9)'
            ),
            height=800,
            margin=dict(r=100, l=50, b=50, t=80),
            paper_bgcolor='rgba(0,0,0,0.9)',
            plot_bgcolor='rgba(0,0,0,0.9)'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error in create_slice_based_3d_figure: {str(e)}")
        # Return fallback
        return create_advanced_3d_figure(compound_matrix, projection_type, thickness, gap, max_projection)

def register_3d_callback(app, cache):
    # Callback to handle slide selection and update 2D slice view
    @app.callback(
        Output("threed-2d-slice-graph", "figure"),
        Input("slide-selector-3d", "value"),
        prevent_initial_call=True
    )
    def update_2d_slice_view(selected_slide):
        if selected_slide is None:
            raise PreventUpdate
        
        try:
            session_id = session.get('session_id', 'default')
            compound_matrix = cache.get(f"{session_id}:compound_matrix")
            
            if compound_matrix is None:
                raise PreventUpdate
            
            # Get the selected slice data
            slice_data = compound_matrix[selected_slide]
            
            # Rainbow colormap for intensity-based coloring
            rainbow_colorscale = [
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
            
            # Create 2D slice figure with Rainbow colormap
            fig = go.Figure(
                data=go.Heatmap(
                    z=slice_data,
                    colorscale=rainbow_colorscale,
                    colorbar=dict(
                        title=dict(text='Intensity', font=dict(size=14, color='white')),
                        x=1.02, 
                        len=0.8,
                        thickness=15,
                        tickfont=dict(size=12, color='white'),
                        tickcolor='white'
                    ),
                    zsmooth='fast',
                    connectgaps=True,
                    hovertemplate='<b>Brain Slice</b><br>X: %{x}<br>Y: %{y}<br>Intensity: %{z:.3f}<extra></extra>'
                ),
                layout=go.Layout(
                    margin=dict(l=5, r=40, t=50, b=5),
                    title=dict(
                        text=f"Slide {selected_slide+1} of {compound_matrix.shape[0]}",
                        font=dict(size=16, color='white'),
                        x=0.5
                    ),
                    xaxis=dict(
                        showticklabels=False, 
                        showgrid=False, 
                        zeroline=False,
                        showspikes=True,
                        spikecolor='white',
                        spikethickness=1
                    ),
                    yaxis=dict(
                        showticklabels=False, 
                        scaleanchor="x", 
                        scaleratio=1, 
                        showgrid=False,
                        zeroline=False,
                        autorange='reversed',
                        showspikes=True,
                        spikecolor='white',
                        spikethickness=1
                    ),
                    plot_bgcolor='black',
                    paper_bgcolor='black',
                    font=dict(color='white')
                )
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error updating 2D slice view: {str(e)}")
            raise PreventUpdate

    # Callback to handle pixel selection from 2D graph
    @app.callback(
        Output("selected-pixel-coords", "data"),
        Input("threed-2d-slice-graph", "clickData"),
        State("slide-selector-3d", "value"),
        prevent_initial_call=True
    )
    def handle_pixel_selection(click_data, current_slide):
        """Handle pixel selection from 2D heatmap"""
        if click_data is None or current_slide is None:
            raise PreventUpdate
        
        try:
            # Extract pixel coordinates from click data
            point = click_data['points'][0]
            x_coord = int(point['x'])
            y_coord = int(point['y'])
            intensity = point['z'] if 'z' in point else 0
            
            logger.info(f"Pixel selected: x={x_coord}, y={y_coord}, slide={current_slide}, intensity={intensity}")
            
            return {
                "x": x_coord,
                "y": y_coord, 
                "slide": current_slide,
                "intensity": intensity
            }
            
        except Exception as e:
            logger.error(f"Error handling pixel selection: {str(e)}")
            raise PreventUpdate

    # Callback to update pixel info display
    @app.callback(
        Output("pixel-info", "children"),
        Input("selected-pixel-coords", "data"),
        prevent_initial_call=True
    )
    def update_pixel_info(pixel_data):
        """Update the pixel information display"""
        if pixel_data is None or pixel_data.get("x") is None:
            return "No pixel selected"
        
        return f"Selected: X={pixel_data['x']}, Y={pixel_data['y']}, Slide={pixel_data['slide']+1}, Intensity={pixel_data.get('intensity', 0):.3f}"

    # Callback to handle Run button for 3D point analysis
    @app.callback(
        Output("threed-3d-graph", "figure"),
        Input("run-pixel-analysis", "n_clicks"),
        State("selected-pixel-coords", "data"),
        State("slide-selector-3d", "value"),
        prevent_initial_call=True
    )
    def run_pixel_analysis(n_clicks, pixel_data, current_slide):
        """Generate 3D visualization focused on selected pixel"""
        if n_clicks == 0 or pixel_data is None or pixel_data.get("x") is None:
            raise PreventUpdate
        
        try:
            session_id = session.get('session_id', 'default')
            compound_matrix = cache.get(f"{session_id}:compound_matrix")
            
            if compound_matrix is None:
                logger.error("No compound matrix available for 3D analysis")
                raise PreventUpdate
            
            # Create 3D visualization focused on the selected pixel
            fig = create_pixel_focused_3d_figure(compound_matrix, pixel_data)
            return fig
            
        except Exception as e:
            logger.error(f"Error in pixel analysis: {str(e)}")
            raise PreventUpdate

    # Callback to navigate to 3D visualization page
    @app.callback(
        Output("selected-page", "data", allow_duplicate=True),
        Input("generate-viz-btn", "n_clicks"),
        State("visualization-type", "value"),
        prevent_initial_call=True,
        allow_duplicate=True
    )
    def navigate_to_3d_page(generate_clicks, viz_type):
        """Navigate to 3D visualization page when generate is clicked"""
        if generate_clicks and viz_type == "3dimage":
            logger.info("[3D Navigation] Navigating to 3D visualization page")
            return "3dvisualization"
        raise PreventUpdate

    # Callback to handle 3D graph click (only when on visualization page)
    @app.callback(
        Output("selected-page", "data", allow_duplicate=True),
        Input("threed-3d-graph", "clickData"),
        prevent_initial_call=True,
        allow_duplicate=True
    )
    def handle_3d_graph_click(graph_click):
        """Navigate to 3D page when 3D graph is clicked"""
        if graph_click:
            return "3dvisualization"
        raise PreventUpdate

    # Callback to return from 3D page




    # Callback to populate slide selector with available slides
    @app.callback(
        Output("slide-selector-3d", "options"),
        Output("slide-selector-3d", "value"),
        Output("slide-info-3d", "children"),
        Input("visualization-type", "value"),
        Input("slide-selector-3d", "value"),
        prevent_initial_call=True,
        allow_duplicate=True
    )
    def update_3d_slide_selector_and_info(viz_type, selected_slide):
        if viz_type != "3dimage":
            return [], None, "Please select '3D Image' visualization type."
        try:
            session_id = session.get('session_id', 'default')
            compound_matrix = cache.get(f"{session_id}:compound_matrix")
            logger.info(f"[3D] compound_matrix type: {type(compound_matrix)}, shape: {getattr(compound_matrix, 'shape', None)}")
            if compound_matrix is None:
                logger.warning("[3D] compound_matrix is missing in cache. User must generate visualization first.")
                return [], None, (
                    html.Div([
                        html.I(className="fas fa-exclamation-triangle", style={"color": "#ff9800", "fontSize": "2rem"}),
                        html.H3("No 3D Data Available", style={"color": "#ff9800", "marginTop": "16px"}),
                        html.P("No slides found. Please select a molecule and generate a visualization before using 3D view.", style={"color": "#b0bec5", "marginTop": "8px"})
                    ], style={"textAlign": "center", "padding": "40px"})
                )
            num_slices = compound_matrix.shape[0]
            if num_slices == 0:
                logger.warning("[3D] compound_matrix exists but has zero slices.")
                return [], None, (
                    html.Div([
                        html.I(className="fas fa-exclamation-triangle", style={"color": "#ff9800", "fontSize": "2rem"}),
                        html.H3("No Slides Found", style={"color": "#ff9800", "marginTop": "16px"}),
                        html.P("The processed data contains no slides. Please check your input data.", style={"color": "#b0bec5", "marginTop": "8px"})
                    ], style={"textAlign": "center", "padding": "40px"})
                )
            options = [
                {'label': f'Slide {i+1}', 'value': i} for i in range(num_slices)
            ]
            logger.info(f"[3D] Populated {num_slices} slides for dropdown.")
            # Determine which slide to show info for
            if selected_slide is None:
                slide_info = f"{num_slices} slides available. Select one to view in 3D."
                value = 0
            else:
                try:
                    slice_data = compound_matrix[selected_slide]
                    non_zero_pixels = np.count_nonzero(slice_data)
                    total_pixels = slice_data.size
                    max_intensity = np.max(slice_data)
                    min_intensity = np.min(slice_data)
                    slide_info = f"Slide {selected_slide+1}: {non_zero_pixels:,} active pixels, Intensity range: {min_intensity:.3f} - {max_intensity:.3f}"
                    value = selected_slide
                except Exception as e:
                    logger.error(f"Error updating slide info: {str(e)}")
                    slide_info = "Error loading slide information."
                    value = selected_slide
            return options, value, slide_info
        except Exception as e:
            logger.error(f"Error populating slide selector: {str(e)}")
            return [], None, (
                html.Div([
                    html.I(className="fas fa-exclamation-triangle", style={"color": "#f44336", "fontSize": "2rem"}),
                    html.H3("Error Loading Slides", style={"color": "#f44336", "marginTop": "16px"}),
                    html.P(f"{str(e)}", style={"color": "#b0bec5", "marginTop": "8px"})
                ], style={"textAlign": "center", "padding": "40px"})
            )
    
    # Callback to show slide information
    # @app.callback(
    #     Output("slide-info-3d", "children"),
    #     Input("slide-selector-3d", "value"),
    #     prevent_initial_call=True
    # )
    # def update_slide_info(selected_slide):
    #     if selected_slide is None:
    #         return "Please select a slide to visualize."
        
    #     try:
    #         session_id = session.get('session_id', 'default')
    #         compound_matrix = cache.get(f"{session_id}:compound_matrix")
            
    #         if compound_matrix is None:
    #             return "No data available. Please process data first."
            
    #         slice_data = compound_matrix[selected_slide]
    #         non_zero_pixels = np.count_nonzero(slice_data)
    #         total_pixels = slice_data.size
    #         max_intensity = np.max(slice_data)
    #         min_intensity = np.min(slice_data)
            
    #         return f"Slide {selected_slide+1}: {non_zero_pixels:,} active pixels, Intensity range: {min_intensity:.3f} - {max_intensity:.3f}"
            
    #     except Exception as e:
    #         logger.error(f"Error updating slide info: {str(e)}")
    #         return "Error loading slide information."
    
    # Callback to show/hide the max projection dropdown based on projection type selection
    @app.callback(
        Output("max-projection-container", "style"),
        Input("projection-type", "value")
    )
    def toggle_max_projection(projection_type):
        if projection_type == 'maximum':
            return {'display': 'block'}
        return {'display': 'none'}
    
    # Ensure 3D visualization container stays hidden for 3D visualization
    @app.callback(
        Output("3d-visualization-container", "style"),
        Input("visualization-type", "value"),
        prevent_initial_call=True
    )
    def ensure_3d_container_hidden(viz_type):
        # Always keep 3D container hidden on main page - only show on dedicated 3D page
        return {"display": "none"}
    
    # Callback to view the 3D visualization
    @app.callback(
        Output("3d-visualization-container", "children"),
        Input("visualization-type", "value"),
        State("thickness-value", "value"),
        State("gap-value", "value"),
        State("max-projection-value", "value"),
        State("projection-type", "value"),
        State("3d-visualization-type", "value"),
        Input('force-3d-update', 'n_intervals'),
        prevent_initial_call=True
    )
    def view_all_3d_slides(viz_type, thickness, gap, max_projection, projection_type, viz_type_3d, force_update):
        # Don't show any 3D visualization on main page - only on dedicated 3D page
        # Return empty div to keep container completely hidden
        return html.Div(style={"display": "none"})

    # Fixed callback to download 3D image
    @app.callback(
        Output("download-3d-image", "data"),
        Input("save-3d-button", "n_clicks"),
        State("thickness-value", "value"),
        State("gap-value", "value"),
        State("max-projection-value", "value"),
        State("projection-type", "value"),
        State("3d-visualization-type", "value"),
        State("slide-selector-3d", "value"),
        prevent_initial_call=True
    )
    def download_3d_image(n_clicks, thickness, gap, max_projection, projection_type, viz_type, selected_slide):
        if not n_clicks:
            raise PreventUpdate
        
        logger.info(f"Download 3D button clicked. n_clicks: {n_clicks}")
        
        try:
            session_id = session.get('session_id', 'default')
            cache_key = f"{session_id}:compound_matrix"
            compound_matrix = cache.get(cache_key)
            
            if compound_matrix is None:
                logger.warning("No compound matrix found for download")
                return None
            
            logger.info(f"Creating 3D figure for download with shape: {compound_matrix.shape}")
            
            # Create 3D figure for download based on visualization type
            if viz_type == 'advanced':
                fig = create_advanced_3d_figure(compound_matrix, projection_type, thickness, gap, max_projection)
            elif viz_type == 'slice-based':
                fig = create_slice_based_3d_figure(compound_matrix, projection_type, thickness, gap, max_projection)
            else:  # simple
                fig = create_simple_3d_figure(compound_matrix, projection_type, thickness, gap, max_projection)
            
            # Update layout for better download quality
            fig.update_layout(
                title=dict(
                    text="3D Brain Visualization",
                    font=dict(size=24, color='black'),
                    x=0.5
                ),
                scene=dict(
                    xaxis=dict(title="X Axis", titlefont=dict(size=16), tickfont=dict(size=12)),
                    yaxis=dict(title="Y Axis", titlefont=dict(size=16), tickfont=dict(size=12)),
                    zaxis=dict(title="Z Axis", titlefont=dict(size=16), tickfont=dict(size=12)),
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.2)
                    )
                ),
                height=800,
                width=1200,
                margin=dict(r=100, l=50, b=50, t=80)
            )
            
            # Generate timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"3D_Brain_Visualization_{projection_type}_{thickness}_{gap}_{timestamp}.nii.gz"
            
            logger.info(f"Generating NIfTI file: {filename}")
            
            # Create NIfTI file from compound matrix
            try:
                import nibabel as nib
                
                # Create NIfTI image from compound matrix
                # Reshape matrix from (z,y,x) to (x,y,z) for NIfTI format
                output_matrix = np.transpose(compound_matrix, (2, 1, 0))
                nii_img = nib.Nifti1Image(output_matrix, affine=np.eye(4))
                
                # Save to buffer using to_bytes()
                nii_buffer = io.BytesIO()
                nii_buffer.write(nii_img.to_bytes())
                nii_buffer.seek(0)
                
                logger.info(f"NIfTI file generated successfully, size: {len(nii_buffer.getvalue())} bytes")
                
                return dcc.send_bytes(
                    nii_buffer.getvalue(),
                    filename=filename
                )
                
            except Exception as nii_error:
                logger.error(f"Error generating NIfTI file: {str(nii_error)}")
                # Fallback: export as numpy array
                np_buffer = io.BytesIO()
                np.save(np_buffer, compound_matrix)
                np_buffer.seek(0)
                
                fallback_filename = f"3D_Brain_Visualization_{projection_type}_{thickness}_{gap}_{timestamp}.npy"
                return dcc.send_bytes(
                    np_buffer.getvalue(),
                    filename=fallback_filename
                )
        
        except Exception as e:
            logger.error(f"Error downloading 3D image: {str(e)}")
            import traceback
            logger.error(f"Download traceback: {traceback.format_exc()}")
            return None

    # Fixed callback to show download status
    @app.callback(
        Output("save-3d-button", "children"),
        Input("save-3d-button", "n_clicks"),
        Input("download-3d-image", "data"),
        prevent_initial_call=True
    )
    def update_download_button(n_clicks, download_data):
        ctx = callback_context
        if not ctx.triggered:
            return [
                html.I(className="fas fa-download"),
                " Download 3D NIfTI File"
            ]
        
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if triggered_id == "save-3d-button":
            return [
                html.I(className="fas fa-spinner fa-spin"),
                " Generating NIfTI File..."
            ]
        elif triggered_id == "download-3d-image" and download_data:
            return [
                html.I(className="fas fa-check"),
                " Download Complete!"
            ]
        
        return [
            html.I(className="fas fa-download"),
            " Download 3D NIfTI File"
        ]