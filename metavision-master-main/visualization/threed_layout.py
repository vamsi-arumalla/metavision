from dash import html, dcc, Input, Output
import numpy as np
from flask import session
import plotly.graph_objects as go
import os
import nibabel as nib

def generate_nifti_files(compound_matrix, session_id):
    """
    Generate NIfTI files for all slides and save them in the terminal
    """
    try:
        # Create nifti directory if it doesn't exist
        nifti_dir = "nifti_files"
        if not os.path.exists(nifti_dir):
            os.makedirs(nifti_dir)
        
        print(f"[NIfTI] Generating NIfTI files for session {session_id}")
        print(f"[NIfTI] Compound matrix shape: {compound_matrix.shape}")
        
        # Generate NIfTI file for each slice
        for i in range(compound_matrix.shape[0]):
            slice_data = compound_matrix[i]
            
            # Create NIfTI image
            nii_img = nib.Nifti1Image(slice_data, np.eye(4))
            
            # Save as .nii.gz
            filename = f"slide{i+1}.nii.gz"
            filepath = os.path.join(nifti_dir, filename)
            nib.save(nii_img, filepath)
            
            print(f"[NIfTI] Generated: {filepath}")
        
        print(f"[NIfTI] Successfully generated {compound_matrix.shape[0]} NIfTI files")
        return True
        
    except Exception as e:
        print(f"[NIfTI] Error generating NIfTI files: {e}")
        return False

# COMMENT OUT get_3d_image_layout and all its internal helper functions and usages
# def get_3d_image_layout(cache):
#     try:
#         print("DEBUG: get_3d_image_layout CALLED (Plotly 3D)")
#         session_id = session.get('session_id', 'default')
#         compound_matrix = cache.get(f"{session_id}:compound_matrix")
#         if compound_matrix is None:
#             # Use demo data if no real data is available
#             compound_matrix = np.zeros((18, 30, 30))  # Fixed to 18 slices
            
#             # Create a brain-like structure
#             for z in range(18):
#                 for y in range(30):
#                     for x in range(30):
#                         # Create a spherical brain-like shape
#                         center_x, center_y, center_z = 15, 15, 9
#                         distance = np.sqrt((x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2)
                        
#                         if distance < 12:  # Brain volume
#                             # Add some variation and structure
#                             intensity = 50 + 50 * np.exp(-distance/8) + np.random.normal(0, 10)
#                             compound_matrix[z, y, x] = max(0, intensity)
            
#             print("DEBUG: Using demo data for 3D visualization")
        
#         print(f"DEBUG: compound_matrix shape: {compound_matrix.shape}, dtype: {compound_matrix.dtype}")
#         print(f"DEBUG: compound_matrix min: {np.nanmin(compound_matrix)}, max: {np.nanmax(compound_matrix)}")
#         print(f"DEBUG: compound_matrix has NaN: {np.isnan(compound_matrix).any()}, has inf: {np.isinf(compound_matrix).any()}")
        
#         # Downsample for performance if needed
#         max_z, max_y, max_x = 32, 64, 64  # Smaller for better performance
#         z, y, x = compound_matrix.shape
#         dz = max(1, z // max_z) if z > max_z else 1
#         dy = max(1, y // max_y) if y > max_y else 1
#         dx = max(1, x // max_x) if x > max_x else 1
#         arr_ds = compound_matrix[::dz, ::dy, ::dx]
#         print(f"DEBUG: Using shape for 3D: {arr_ds.shape}")
        
#         # Ensure float32 for better performance
#         arr_ds = arr_ds.astype(np.float32)
        
#         # 2D slice viewer setup
#         n_slices = arr_ds.shape[0]
#         default_slice = n_slices // 2
        
#         # Create initial 2D slice figure with Rainbow colormap
#         def get_2d_slice_fig(arr, idx):
#             # Use original resolution without excessive smoothing for sharper image
#             slice_data = arr[idx]
            
#             # Apply light smoothing to create solid areas instead of scattered points
#             try:
#                 from scipy.ndimage import gaussian_filter
#                 slice_data = gaussian_filter(slice_data, sigma=0.5)
#                 print(f"[DEBUG] Applied Gaussian smoothing to 2D slice {idx}")
#             except ImportError:
#                 print(f"[DEBUG] scipy not available, using original data for 2D slice {idx}")
#             except Exception as e:
#                 print(f"[DEBUG] Error applying smoothing to 2D slice {idx}: {e}")
            
#             # Normalize data for better visualization
#             if np.max(slice_data) > 0:
#                 slice_data = slice_data / np.max(slice_data)
            
#             print(f"[DEBUG] 2D slice {idx} - shape: {slice_data.shape}, min: {np.min(slice_data)}, max: {np.max(slice_data)}")
            
#             # Rainbow colormap for intensity-based coloring
#             rainbow_colorscale = [
#                 [0.0, '#ffffff'],  # White for zero values
#                 [0.1, '#ff0000'],  # Red
#                 [0.2, '#ff8000'],  # Orange
#                 [0.3, '#ffff00'],  # Yellow
#                 [0.4, '#80ff00'],  # Lime
#                 [0.5, '#00ff00'],  # Green
#                 [0.6, '#00ff80'],  # Light green
#                 [0.7, '#00ffff'],  # Cyan
#                 [0.8, '#0080ff'],  # Light blue
#                 [0.9, '#0000ff'],  # Blue
#                 [1.0, '#8000ff']   # Purple
#             ]
            
#             fig = go.Figure(
#                 data=go.Heatmap(
#                     z=slice_data,
#                     colorscale=rainbow_colorscale,
#                     colorbar=dict(
#                         title=dict(text='Intensity', font=dict(size=14, color='white')),
#                         x=1.02, 
#                         len=0.8,
#                         thickness=15,
#                         tickfont=dict(size=12, color='white'),
#                         tickcolor='white'
#                     ),
#                     zsmooth='best',  # Best interpolation for solid areas
#                     connectgaps=True,
#                     hoverongaps=False,
#                     hovertemplate='<b>Brain Slice</b><br>X: %{x}<br>Y: %{y}<br>Intensity: %{z:.3f}<extra></extra>',
#                     # Force solid rendering
#                     autocolorscale=False,
#                     reversescale=False
#                 ),
#                 layout=go.Layout(
#                     margin=dict(l=5, r=40, t=30, b=5),
#                     title=dict(
#                         text=f"Slice {idx+1} of {arr.shape[0]}",
#                         font=dict(size=12, color='white'),
#                         x=0.5
#                     ),
#                     xaxis=dict(
#                         showticklabels=False, 
#                         showgrid=False, 
#                         zeroline=False,
#                         showspikes=False,
#                         range=[0, slice_data.shape[1]-1]
#                     ),
#                     yaxis=dict(
#                         showticklabels=False, 
#                         scaleanchor="x", 
#                         scaleratio=1, 
#                         showgrid=False,
#                         zeroline=False,
#                         autorange='reversed',  # Proper image orientation
#                         showspikes=False,
#                         range=[0, slice_data.shape[0]-1]
#                     ),
#                     plot_bgcolor='black',
#                     paper_bgcolor='black',
#                     font=dict(color='white'),
#                     width=450,
#                     height=400
#                 )
#             )
#             return fig
        
#         # Create 3D visualization focused on a specific pixel
#         def create_pixel_focused_3d_figure(arr, pixel_data):
#             """Create 3D visualization focused on a selected pixel"""
#             try:
#                 print(f"DEBUG: Creating pixel-focused 3D visualization for pixel {pixel_data}")
                
#                 from skimage import measure
#                 from scipy.ndimage import gaussian_filter
                
#                 # Get pixel coordinates
#                 x_coord = pixel_data['x']
#                 y_coord = pixel_data['y']
#                 slide_idx = pixel_data['slide']
                
#                 # Create a focused region around the selected pixel
#                 # Extract a small region around the pixel
#                 radius = 5  # pixels around the selected point
                
#                 # Ensure coordinates are within bounds
#                 x_start = max(0, x_coord - radius)
#                 x_end = min(arr.shape[2], x_coord + radius + 1)
#                 y_start = max(0, y_coord - radius)
#                 y_end = min(arr.shape[1], y_coord + radius + 1)
#                 z_start = max(0, slide_idx - 2)
#                 z_end = min(arr.shape[0], slide_idx + 3)
                
#                 # Extract the region of interest
#                 roi = arr[z_start:z_end, y_start:y_end, x_start:x_end]
                
#                 if roi.size == 0:
#                     print("DEBUG: ROI is empty, creating fallback visualization")
#                     return create_3d_figure(arr)  # Fallback to full visualization
                
#                 # Smooth the ROI for better visualization
#                 smoothed_roi = gaussian_filter(roi, sigma=0.5)
                
#                 # Create isosurface for the focused region
#                 try:
#                     from skimage import measure
                    
#                     # Use a threshold based on the selected pixel's intensity
#                     threshold = pixel_data.get('intensity', 0) * 0.5
#                     if threshold <= 0:
#                         threshold = np.percentile(smoothed_roi[smoothed_roi > 0], 50) if np.any(smoothed_roi > 0) else 0.1
                    
#                     verts, faces, _, _ = measure.marching_cubes(
#                         smoothed_roi, 
#                         level=threshold,
#                         spacing=(1, 1, 1)
#                     )
                    
#                     if len(verts) > 0 and len(faces) > 0:
#                         # Adjust coordinates to match original space
#                         x, y, z = verts.T
#                         x += x_start
#                         y += y_start
#                         z += z_start
#                         i, j, k = faces.T
                        
#                         # Create focused mesh
#                         mesh = go.Mesh3d(
#                             x=x, y=y, z=z,
#                             i=i, j=j, k=k,
#                             color='red',
#                             opacity=0.8,
#                             lighting=dict(
#                                 ambient=0.4,
#                                 diffuse=0.8,
#                                 fresnel=0.1,
#                                 specular=0.05,
#                                 roughness=0.1
#                             ),
#                             lightposition=dict(x=100, y=100, z=1000),
#                             name='Focused Pixel Region',
#                             showlegend=False,
#                             flatshading=False
#                         )
                        
#                         # Create the figure
#                         fig = go.Figure(data=[mesh])
                        
#                         # Configure layout for focused visualization
#                         fig.update_layout(
#                             title=dict(
#                                 text=f"3D Focus: Pixel ({x_coord}, {y_coord}) on Slide {slide_idx+1}",
#                                 font=dict(color='white', size=16),
#                                 x=0.5
#                             ),
#                             scene=dict(
#                                 xaxis=dict(title="", showgrid=False, showticklabels=False, showbackground=False, showspikes=False, zeroline=False, visible=False),
#                                 yaxis=dict(title="", showgrid=False, showticklabels=False, showbackground=False, showspikes=False, zeroline=False, visible=False),
#                                 zaxis=dict(title="", showgrid=False, showticklabels=False, showbackground=False, showspikes=False, zeroline=False, visible=False),
#                                 camera=dict(
#                                     eye=dict(x=1.5, y=1.5, z=1.2),
#                                     center=dict(x=x_coord, y=y_coord, z=slide_idx)
#                                 ),
#                                 bgcolor='black',
#                                 aspectmode='data'
#                             ),
#                             margin=dict(l=0, r=0, t=60, b=0),
#                             paper_bgcolor='black',
#                             plot_bgcolor='black',
#                             height=500,
#                             showlegend=False,
#                             font=dict(color='white')
#                         )
                        
#                         print(f"DEBUG: Created focused 3D visualization with {len(verts)} vertices")
#                         return fig
                    
#                 except Exception as e:
#                     print(f"DEBUG: Error creating focused mesh: {e}")
                
#                 # Fallback to simple point visualization
#                 fig = go.Figure()
#                 fig.add_trace(go.Scatter3d(
#                     x=[x_coord], y=[y_coord], z=[slide_idx],
#                     mode='markers',
#                     marker=dict(size=10, color='red'),
#                     name='Selected Pixel'
#                 ))
                
#                 fig.update_layout(
#                     title=dict(text=f"Selected Pixel: ({x_coord}, {y_coord}, {slide_idx+1})", font=dict(color='white')),
#                     scene=dict(bgcolor='black'),
#                     paper_bgcolor='black',
#                     height=500
#                 )
                
#                 return fig
                
#             except Exception as e:
#                 print(f"ERROR: Failed to create pixel-focused 3D visualization: {e}")
#                 return create_3d_figure(arr)  # Fallback to full visualization

#         # Create advanced 3D visualization with different rendering modes
#         def create_advanced_3d_figure(arr, rendering_mode='surface', color_scheme='rainbow'):
#             """Create 3D visualization with different rendering modes and color schemes"""
#             try:
#                 print(f"DEBUG: Creating advanced 3D brain with mode: {rendering_mode}, colors: {color_scheme}")
                
#                 from skimage import measure
#                 from scipy.ndimage import gaussian_filter
                
#                 # Smooth the data for better surface generation
#                 smoothed_data = gaussian_filter(arr, sigma=1.0)
                
#                 # Normalize data
#                 data_min = np.nanmin(smoothed_data)
#                 data_max = np.nanmax(smoothed_data)
                
#                 if data_max <= data_min:
#                     print("DEBUG: No data variation found")
#                     return create_fallback_3d_figure()
                
#                 # Define color schemes
#                 color_schemes = {
#                     'rainbow': ['red', 'orange', 'yellow', 'green', 'blue', 'purple'],
#                     'redblue': ['blue', 'cyan', 'white', 'yellow', 'red'],
#                     'viridis': ['#440154', '#31688e', '#35b779', '#fde725'],
#                     'plasma': ['#0d0887', '#7e03a8', '#cc4778', '#f89441', '#f0f921']
#                 }
                
#                 colors = color_schemes.get(color_scheme, color_schemes['rainbow'])
                
#                 traces = []
                
#                 if rendering_mode == 'surface':
#                     # Create multiple isosurface levels for solid brain appearance
#                     percentiles = [20, 40, 60, 80]
#                     opacities = [0.2, 0.3, 0.4, 0.6]
                    
#                     for i, (percentile, opacity) in enumerate(zip(percentiles, opacities)):
#                         try:
#                             threshold = np.percentile(smoothed_data[smoothed_data > 0], percentile)
                            
#                             if threshold > data_min:
#                                 verts, faces, _, _ = measure.marching_cubes(
#                                     smoothed_data, 
#                                     level=threshold, 
#                                     spacing=(1, 1, 1),
#                                     step_size=1,
#                                     allow_degenerate=False
#                                 )
                                
#                                 if len(verts) > 0 and len(faces) > 0:
#                                     x, y, z = verts.T
#                                     i, j, k = faces.T
                                    
#                                     color = colors[i % len(colors)]
#                                     mesh = go.Mesh3d(
#                                         x=x, y=y, z=z,
#                                         i=i, j=j, k=k,
#                                         color=color,
#                                         opacity=opacity,
#                                         lighting=dict(
#                                             ambient=0.4,
#                                             diffuse=0.8,
#                                             fresnel=0.1,
#                                             specular=0.05,
#                                             roughness=0.1
#                                         ),
#                                         lightposition=dict(x=100, y=100, z=1000),
#                                         name=f'Brain Layer {percentile}%',
#                                         showlegend=False,
#                                         flatshading=False
#                                     )
#                                     traces.append(mesh)
                        
#                         except Exception as e:
#                             print(f"WARNING: Could not create isosurface at {percentile}%: {e}")
#                             continue
                
#                 elif rendering_mode == 'volume':
#                     # Volume rendering using scatter3d with opacity
#                     x_coords, y_coords, z_coords, intensities = [], [], [], []
                    
#                     # Sample points for volume rendering
#                     step = max(1, min(arr.shape) // 20)  # Adaptive sampling
                    
#                     for z in range(0, arr.shape[0], step):
#                         for y in range(0, arr.shape[1], step):
#                             for x in range(0, arr.shape[2], step):
#                                 intensity = arr[z, y, x]
#                                 if intensity > data_min:
#                                     x_coords.append(x)
#                                     y_coords.append(y)
#                                     z_coords.append(z)
#                                     intensities.append(intensity)
                    
#                     if x_coords:
#                         # Normalize intensities for color mapping
#                         max_int = max(intensities)
#                         normalized_intensities = [i/max_int for i in intensities]
                        
#                         scatter = go.Scatter3d(
#                             x=x_coords, y=y_coords, z=z_coords,
#                             mode='markers',
#                             marker=dict(
#                                 size=2,
#                                 color=normalized_intensities,
#                                 colorscale=color_scheme,
#                                 opacity=0.6,
#                                 showscale=True
#                             ),
#                             name='Volume Rendering',
#                             showlegend=False
#                         )
#                         traces.append(scatter)
                
#                 elif rendering_mode == 'points':
#                     # Point cloud rendering
#                     x_coords, y_coords, z_coords, intensities = [], [], [], []
                    
#                     # Sample high-intensity points
#                     threshold = np.percentile(arr, 70)
#                     step = max(1, min(arr.shape) // 30)
                    
#                     for z in range(0, arr.shape[0], step):
#                         for y in range(0, arr.shape[1], step):
#                             for x in range(0, arr.shape[2], step):
#                                 intensity = arr[z, y, x]
#                                 if intensity > threshold:
#                                     x_coords.append(x)
#                                     y_coords.append(y)
#                                     z_coords.append(z)
#                                     intensities.append(intensity)
                    
#                     if x_coords:
#                         max_int = max(intensities)
#                         normalized_intensities = [i/max_int for i in intensities]
                        
#                         scatter = go.Scatter3d(
#                             x=x_coords, y=y_coords, z=z_coords,
#                             mode='markers',
#                             marker=dict(
#                                 size=3,
#                                 color=normalized_intensities,
#                                 colorscale=color_scheme,
#                                 opacity=0.8
#                             ),
#                             name='Point Cloud',
#                             showlegend=False
#                         )
#                         traces.append(scatter)
                
#                 # Create the figure
#                 fig = go.Figure(data=traces)
                
#                 # Configure layout for full-page visualization
#                 fig.update_layout(
#                     title=dict(
#                         text=f"3D Brain Visualization - {rendering_mode.title()} Mode",
#                         font=dict(color='white', size=20),
#                         x=0.5
#                     ),
#                     scene=dict(
#                         xaxis=dict(title="", showgrid=False, showticklabels=False, showbackground=False, showspikes=False, zeroline=False, visible=False),
#                         yaxis=dict(title="", showgrid=False, showticklabels=False, showbackground=False, showspikes=False, zeroline=False, visible=False),
#                         zaxis=dict(title="", showgrid=False, showticklabels=False, showbackground=False, showspikes=False, zeroline=False, visible=False),
#                         camera=dict(
#                             eye=dict(x=1.8, y=1.8, z=1.5),
#                             center=dict(x=0, y=0, z=0)
#                         ),
#                         bgcolor='black',
#                         aspectmode='data'
#                     ),
#                     margin=dict(l=0, r=0, t=60, b=0),
#                     paper_bgcolor='black',
#                     plot_bgcolor='black',
#                     height=800,
#                     showlegend=False,
#                     font=dict(color='white')
#                 )
                
#                 print(f"DEBUG: Created advanced 3D brain with {len(traces)} traces")
#                 return fig
                
#             except Exception as e:
#                 print(f"ERROR: Failed to create advanced 3D brain: {e}")
#                 return create_fallback_3d_figure()

#         def create_fallback_3d_figure():
#             """Create a fallback 3D figure when data is not available"""
#             fig = go.Figure()
#             fig.add_trace(go.Scatter3d(
#                 x=[10, 20, 30],
#                 y=[10, 20, 30],
#                 z=[10, 20, 30],
#                 mode='markers',
#                 marker=dict(size=15, color='red'),
#                 name='Fallback'
#             ))
            
#             fig.update_layout(
#                 title=dict(text="3D Visualization Error", font=dict(color='white')),
#                 scene=dict(bgcolor='black'),
#                 paper_bgcolor='black',
#                 height=800
#             )
#             return fig

#         # Create 3D visualization using volume rendering approach
#         def create_3d_figure(arr):
#             # Create solid 3D brain visualization using smooth surfaces (no dots)
#             try:
#                 print(f"DEBUG: Creating solid 3D brain with array shape: {arr.shape}")
#                 print(f"DEBUG: Data range: {np.nanmin(arr)} to {np.nanmax(arr)}")
                
#                 from skimage import measure
#                 from scipy.ndimage import gaussian_filter
                
#                 # Smooth the data for better surface generation
#                 smoothed_data = gaussian_filter(arr, sigma=1.0)
                
#                 # Normalize data
#                 data_min = np.nanmin(smoothed_data)
#                 data_max = np.nanmax(smoothed_data)
                
#                 if data_max <= data_min:
#                     print("DEBUG: No data variation found")
#                     # Fallback visualization
#                     fig = go.Figure()
#                     fig.add_trace(go.Mesh3d(
#                         x=[0, 1, 1, 0, 0, 1, 1, 0],
#                         y=[0, 0, 1, 1, 0, 0, 1, 1],
#                         z=[0, 0, 0, 0, 1, 1, 1, 1],
#                         i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
#                         j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
#                         k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
#                         color='red',
#                         opacity=0.7
#                     ))
#                     return fig
                
#                 # Create multiple isosurface levels for solid brain appearance
#                 traces = []
                
#                 # Use multiple threshold levels to create a solid-looking brain
#                 percentiles = [20, 40, 60, 80]  # Multiple layers
#                 opacities = [0.2, 0.3, 0.4, 0.6]  # Increasing opacity
#                 colors = ['rgba(139,0,0,{})', 'rgba(178,34,34,{})', 'rgba(220,20,60,{})', 'rgba(255,0,0,{})']
                
#                 for i, (percentile, opacity) in enumerate(zip(percentiles, opacities)):
#                     try:
#                         # Calculate threshold for this percentile
#                         threshold = np.percentile(smoothed_data[smoothed_data > 0], percentile)
#                         print(f"DEBUG: Creating isosurface at {percentile}% (threshold: {threshold})")
                        
#                         if threshold > data_min:
#                             # Generate mesh using marching cubes
#                             verts, faces, _, _ = measure.marching_cubes(
#                                 smoothed_data, 
#                                 level=threshold, 
#                                 spacing=(1, 1, 1),
#                                 step_size=1,
#                                 allow_degenerate=False
#                             )
                            
#                             if len(verts) > 0 and len(faces) > 0:
#                                 # Create mesh surface
#                                 x, y, z = verts.T
#                                 i, j, k = faces.T
                                
#                                 # Create solid mesh
#                                 mesh = go.Mesh3d(
#                                     x=x, y=y, z=z,
#                                     i=i, j=j, k=k,
#                                     color=colors[i].format(opacity),
#                                     opacity=opacity,
#                                     lighting=dict(
#                                         ambient=0.4,
#                                         diffuse=0.8,
#                                         fresnel=0.1,
#                                         specular=0.05,
#                                         roughness=0.1
#                                     ),
#                                     lightposition=dict(x=100, y=100, z=1000),
#                                     name=f'Brain Layer {percentile}%',
#                                     showlegend=False,
#                                     flatshading=False,  # Smooth shading
#                                     alphahull=0  # Solid surface
#                                 )
#                                 traces.append(mesh)
#                                 print(f"DEBUG: Created mesh with {len(verts)} vertices, {len(faces)} faces")
                    
#                     except Exception as e:
#                         print(f"WARNING: Could not create isosurface at {percentile}%: {e}")
#                         continue
                
#                 # If no surfaces were created, try a different approach
#                 if not traces:
#                     print("DEBUG: No isosurfaces created, trying alternative method")
                    
#                     # Try with lower threshold
#                     threshold = np.percentile(smoothed_data, 10)
#                     try:
#                         verts, faces, _, _ = measure.marching_cubes(
#                             smoothed_data, 
#                             level=threshold,
#                             spacing=(1, 1, 1)
#                         )
                        
#                         if len(verts) > 0 and len(faces) > 0:
#                             x, y, z = verts.T
#                             i, j, k = faces.T
                            
#                             mesh = go.Mesh3d(
#                                 x=x, y=y, z=z,
#                                 i=i, j=j, k=k,
#                                 color='red',
#                                 opacity=0.7,
#                                 lighting=dict(
#                                     ambient=0.5,
#                                     diffuse=0.8,
#                                     specular=0.1
#                                 ),
#                                 showlegend=False
#                             )
#                             traces.append(mesh)
#                             print(f"DEBUG: Created fallback mesh with {len(verts)} vertices")
                    
#                     except Exception as e:
#                         print(f"ERROR: Fallback mesh creation failed: {e}")
                        
#                         # Final fallback - create a simple brain-like shape
#                         from numpy import pi, sin, cos
                        
#                         # Create a brain-like ellipsoid
#                         u = np.linspace(0, 2 * pi, 50)
#                         v = np.linspace(0, pi, 50)
#                         x = 10 * np.outer(cos(u), sin(v)).flatten()
#                         y = 8 * np.outer(sin(u), sin(v)).flatten()
#                         z = 6 * np.outer(np.ones(np.size(u)), cos(v)).flatten()
                        
#                         # Create triangulation for the ellipsoid
#                         from scipy.spatial import SphericalVoronoi, geometric_slerp
                        
#                         mesh = go.Mesh3d(
#                             x=x + arr.shape[2]/2,
#                             y=y + arr.shape[1]/2,
#                             z=z + arr.shape[0]/2,
#                             color='red',
#                             opacity=0.7,
#                             lighting=dict(ambient=0.5, diffuse=0.8),
#                             showlegend=False
#                         )
#                         traces.append(mesh)
#                         print("DEBUG: Created geometric fallback brain shape")
                
#                 # Create the figure with all surfaces
#                 fig = go.Figure(data=traces)
                
#                 # Configure layout for medical visualization
#                 fig.update_layout(
#                     title=dict(
#                         text="3D Brain Volume",
#                         font=dict(color='white', size=20),
#                         x=0.5
#                     ),
#                     scene=dict(
#                         xaxis=dict(
#                             title="", 
#                             showgrid=False, 
#                             showticklabels=False,
#                             showbackground=False,
#                             showspikes=False,
#                             zeroline=False,
#                             visible=False
#                         ),
#                         yaxis=dict(
#                             title="", 
#                             showgrid=False, 
#                             showticklabels=False,
#                             showbackground=False,
#                             showspikes=False,
#                             zeroline=False,
#                             visible=False
#                         ),
#                         zaxis=dict(
#                             title="", 
#                             showgrid=False, 
#                             showticklabels=False,
#                             showbackground=False,
#                             showspikes=False,
#                             zeroline=False,
#                             visible=False
#                         ),
#                         camera=dict(
#                             eye=dict(x=1.8, y=1.8, z=1.5),
#                             center=dict(x=0, y=0, z=0)
#                         ),
#                         bgcolor='black',
#                         aspectmode='data'
#                     ),
#                     margin=dict(l=0, r=0, t=60, b=0),
#                                 paper_bgcolor='black',
#             plot_bgcolor='black',
#                     height=500,
#                     showlegend=False,
#                     font=dict(color='white')
#                 )
                
#                 print(f"DEBUG: Created 3D brain with {len(traces)} surface layers")
#                 return fig
                
#             except Exception as e:
#                 print(f"ERROR: Failed to create 3D brain: {e}")
#                 import traceback
#                 traceback.print_exc()
                
#                 # Emergency fallback
#                 fig = go.Figure()
#                 fig.add_trace(go.Scatter3d(
#                     x=[10, 20, 30],
#                     y=[10, 20, 30],
#                     z=[10, 20, 30],
#                     mode='markers',
#                     marker=dict(size=15, color='red'),
#                     name='Fallback'
#                 ))
                
#                 fig.update_layout(
#                     title=dict(text="3D Visualization Error", font=dict(color='white')),
#                                 scene=dict(bgcolor='black'),
#             paper_bgcolor='black',
#                     height=500
#                 )
#                 return fig
        
#         # Generate NIfTI files automatically
#         generate_nifti_files(compound_matrix, session_id)
        
#         # Create slide options for dropdown
#         slide_options = [{'label': f'slide{i+1}.nii.gz', 'value': i} for i in range(compound_matrix.shape[0])]
        
#         # Simplified layout with full-space visualization
#         return html.Div([
#             # Hidden stores for pixel selection
#             dcc.Store(id="selected-pixel-coords", data={"x": None, "y": None, "slide": None}),
            
#             # Simple header
#             html.H1("3D Brain Visualization with Rainbow Colormap", 
#                    style={"textAlign": "center", "color": "#ffffff", "marginBottom": "20px", 
#                          "fontSize": "32px", "fontWeight": "bold"}),
            
#             # Simple slide selector (no extra boxes)
#             html.Div([
#                 html.Label("Select Slide:", 
#                           style={"color": "#ffffff", "fontSize": "18px", "fontWeight": "600", "marginRight": "15px"}),
#                 dcc.Dropdown(
#                     id="slide-selector-3d",
#                     options=slide_options,
#                     value=0,  # Default to first slide
#                     placeholder="Select a slide...",
#                     style={
#                         "backgroundColor": "#2d2d2d",
#                         "color": "#ffffff",
#                         "border": "2px solid #9c27b0",
#                         "borderRadius": "8px",
#                         "minHeight": "50px",
#                         "fontSize": "16px",
#                         "fontWeight": "600",
#                         "width": "300px"
#                     }
#                 )
#             ], style={"display": "flex", "alignItems": "center", "justifyContent": "center", "marginBottom": "20px"}),
            
#             # Simple status text (no box)
#             html.Div([
#                 html.P("✅ NIfTI files generated successfully!", 
#                       style={"color": "#4caf50", "fontSize": "14px", "textAlign": "center", "marginBottom": "5px"}),
#                 html.P(f"📁 Files saved in: nifti_files/ (slide1.nii.gz to slide{compound_matrix.shape[0]}.nii.gz)", 
#                       style={"color": "#cccccc", "fontSize": "12px", "textAlign": "center", "marginBottom": "10px"}),
#                 html.Div(id="slide-info-3d", style={
#                     "color": "#ffffff",
#                     "fontSize": "14px",
#                     "textAlign": "center"
#                 })
#             ], style={"marginBottom": "20px"}),
            
#             # Main visualization area - 2D and 3D side by side with exactly equal sized boxes
#             html.Div([
#                 # Left side - 3D visualization (exactly 50% width)
#                 html.Div([
#                     html.H3("3D Brain Volume", 
#                            style={"textAlign": "center", "color": "#ffffff", "marginBottom": "8px", 
#                                  "fontSize": "16px", "fontWeight": "bold"}),
#                     dcc.Graph(
#                         id="threed-3d-graph",
#                         figure=create_3d_figure(arr_ds),
#                         style={"height": "50vh", "width": "95%", "margin": "0 auto", "backgroundColor": "black", "cursor": "pointer"},
#                         config={
#                             'displayModeBar': True,
#                             'displaylogo': False,
#                             'modeBarButtonsToRemove': [],
#                             'responsive': True,
#                             'toImageButtonOptions': {
#                                 'format': 'png',
#                                 'filename': '3d_brain_rainbow',
#                                 'height': 500,
#                                 'width': 500,
#                                 'scale': 2
#                             }
#                         }
#                     ),
#                     # Simple click indicator for 3D graph
#                     html.Div([
#                         html.P("🖱️ Click 3D image to open full-page viewer", 
#                               style={"color": "#4caf50", "fontSize": "12px", "textAlign": "center", "marginTop": "10px"})
#                     ])
#                 ], style={"width": "50%", "marginRight": "5px", "height": "65vh", "backgroundColor": "black", 
#                          "border": "1px solid #333", "borderRadius": "6px", "padding": "10px"}),
                
#                 # Right side - 2D slice visualization (exactly 50% width)
#                 html.Div([
#                     html.H3("2D Slice View", 
#                            style={"textAlign": "center", "color": "#ffffff", "marginBottom": "8px", 
#                                  "fontSize": "16px", "fontWeight": "bold"}),
#                     dcc.Graph(
#                         id="threed-2d-slice-graph",
#                         figure=get_2d_slice_fig(arr_ds, 0),
#                         style={"height": "50vh", "width": "95%", "margin": "0 auto", "backgroundColor": "black"},
#                         config={
#                             'displayModeBar': True,
#                             'displaylogo': False,
#                             'modeBarButtonsToRemove': [],
#                             'responsive': True,
#                             'toImageButtonOptions': {
#                                 'format': 'png',
#                                 'filename': '2d_slice',
#                                 'height': 500,
#                                 'width': 500,
#                                 'scale': 2
#                             }
#                         }
#                     ),
#                     # Pixel selection and Run button
#                     html.Div([
#                         html.P("🎯 Pixel Selection:", 
#                               style={"color": "#4caf50", "fontSize": "12px", "fontWeight": "bold", "marginBottom": "5px"}),
#                         html.P("Click on a pixel in 2D image, then click Run", 
#                               style={"color": "#cccccc", "fontSize": "11px", "marginBottom": "8px"}),
#                         html.Button(
#                             "Run 3D Point Analysis",
#                             id="run-pixel-analysis",
#                             n_clicks=0,
#                             style={
#                                 "backgroundColor": "#9c27b0",
#                                 "color": "white",
#                                 "border": "none",
#                                 "padding": "8px 16px",
#                                 "borderRadius": "4px",
#                                 "fontSize": "12px",
#                                 "fontWeight": "bold",
#                                 "cursor": "pointer"
#                             }
#                         ),
#                         html.Div(id="pixel-info", style={
#                             "color": "#cccccc",
#                             "fontSize": "10px",
#                             "marginTop": "5px",
#                             "textAlign": "center"
#                         })
#                     ], style={"textAlign": "center", "marginTop": "10px", "padding": "8px", "backgroundColor": "#2a2a2a", "borderRadius": "4px"})
#                 ], style={"width": "50%", "marginLeft": "5px", "height": "65vh", "backgroundColor": "black", 
#                          "border": "1px solid #333", "borderRadius": "6px", "padding": "10px"})
                
#             ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start", 
#                      "width": "100%", "height": "65vh", "marginBottom": "20px", "backgroundColor": "black"}),
            
#             # Simple controls info (no box)
#             html.Div([
#                 html.P("🖱️ Rotate: Click & Drag | 🔍 Zoom: Mouse Wheel | 📐 Pan: Shift + Drag | 📷 Export: Use toolbar", 
#                       style={"color": "#cccccc", "fontSize": "14px", "textAlign": "center", "margin": "10px 0"})
#             ])
            
#         ], style={"background": "linear-gradient(135deg, #1a1a1a 0%, #2d1b2e 50%, #1a1a1a 100%)", 
#                  "minHeight": "100vh", "padding": "30px", "color": "#ffffff"})
        
#         # Note: Custom CSS for dropdown styling is added to assets/visualization.css
        
#     except Exception as e:
#         print(f"ERROR in get_3d_image_layout: {e}")
#         import traceback
#         traceback.print_exc()
#         return html.Div([
#             html.H3(f"Error creating 3D visualization: {str(e)}", 
#                    style={"color": "#f44336", "textAlign": "center"})
#         ])