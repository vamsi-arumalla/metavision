import nibabel as nib
import plotly.graph_objects as go
import numpy as np
from skimage import measure
import plotly.express as px

def load_nifti(file_path):
    img = nib.load(file_path)
    data = img.get_fdata()
    return data

def render_nifti_isosurface(file_path, iso_val=0.1, opacity=0.1):
    """Alternative rendering using isosurface extraction"""
    # Load the data
    data = load_nifti(file_path)
    print(f"Data shape: {data.shape}, Min: {data.min()}, Max: {data.max()}")
    
    # Normalize data
    data_norm = (data - data.min()) / (data.max() - data.min() + 1e-8)
    
    # Extract isosurface using marching cubes
    verts, faces, _, _ = measure.marching_cubes(data_norm, level=iso_val)
    
    x, y, z = verts.T
    i, j, k = faces.T
    
    # Create the mesh
    fig = go.Figure(data=[
        go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            opacity=opacity,
            colorscale='Gray',
            intensity=z,
            showscale=True
        )
    ])
    
    fig.update_layout(
        title=f'NIfTI Isosurface (level={iso_val})',
        width=800, height=800,
        scene=dict(
            aspectmode='data'
        )
    )
    
    return fig

fig2 = render_nifti_isosurface('original.nii', iso_val=0.1) 
fig2.show()
