from dash import Input, Output, State, html, ALL, dcc, no_update
from visualization.slides_layout import get_slides_layout
from visualization.norm_layout import get_norm_layout
from visualization.animation_layout import get_animation_layout
from form.form_callback import normalize, align, impute_mat, interpolate_mat
from flask import session
import dash
import logging
import os
import numpy as np
import base64
import io
import json
from datetime import datetime
import plotly.graph_objects as go
import dash_vtk

logger = logging.getLogger('metavision')

def register_visualization_callback(app, cache):
    """
    Register callbacks for enhanced visualization interactions.
    """
    
    # Callback to handle visualization type card selection
    @app.callback(
        Output("visualization-type", "value"),
        Output("column-selector", "value"),
        [
            Input("vis-slides", "n_clicks"),
            Input("vis-normplot", "n_clicks"),
            Input("vis-animation", "n_clicks"),
            Input("vis-3dimage", "n_clicks"),
            Input({"type": "quick-select-btn", "index": ALL}, "n_clicks"),
            Input("store-last-viz-type", "data"),
        ],
        [State("visualization-type", "value"), State("column-selector", "value")],
        prevent_initial_call=True,
    )
    def unified_selection_callback(
        slides_clicks, normplot_clicks, animation_clicks, threed_clicks,
        quick_select_clicks, last_viz_type,
        current_viz_type, current_molecule
    ):
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update, dash.no_update

        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

        # Handle visualization type card selection
        if triggered_id == "vis-slides":
            return "slides", current_molecule
        elif triggered_id == "vis-normplot":
            return "normplot", current_molecule
        elif triggered_id == "vis-animation":
            return "animation", current_molecule
        elif triggered_id == "vis-3dimage":
            return "3dimage", current_molecule

        # Handle quick select
        if "quick-select-btn" in triggered_id:
            print(f"[DEBUG] Quick select button triggered: {triggered_id}")
            print(f"[DEBUG] Quick select clicks: {quick_select_clicks}")
            
            # Find which button was clicked by comparing click counts
            if quick_select_clicks:
                for i, n in enumerate(quick_select_clicks):
                    if n and n > 0:
                        # Fetch molecule list from cache
                        molecule_list = cache.get(f"{session['session_id']}:molecules_list")
                        print(f"[DEBUG] Molecule list for quick select: {molecule_list}")
                        
                        if molecule_list and i < len(molecule_list):
                            selected_molecule = molecule_list[i]
                            print(f"[DEBUG] Selected molecule from quick select: {selected_molecule}")
                            
                            # Force compound matrix regeneration for new molecule
                            cache.delete(f"{session['session_id']}:compound_matrix")
                            print(f"[DEBUG] Cleared compound matrix cache for molecule switch")
                            
                            return current_viz_type, selected_molecule
            
            print("[DEBUG] No valid quick select button found")
            return current_viz_type, current_molecule

        # Handle restore last selection
        if triggered_id == "store-last-viz-type":
            return last_viz_type, current_molecule

        # Always return two values
        return dash.no_update, dash.no_update
    
    # Debug callback to show current selection
    @app.callback(
        Output("debug-selection", "children"),
        [Input("visualization-type", "value"),
         Input("column-selector", "value")],
        prevent_initial_call=False
    )
    def show_current_selection(viz_type, molecule):
        if viz_type and molecule:
            return f"Selected: {viz_type.upper()} visualization for molecule '{molecule}'"
        elif viz_type:
            return f"Selected: {viz_type.upper()} visualization (no molecule selected)"
        elif molecule:
            return f"Selected molecule: '{molecule}' (no visualization type selected)"
        else:
            return "Please select a visualization type and molecule"
    
    # Callback to update card selection styling - with debugging
    @app.callback(
        [
            Output("slides-card", "className"),
            Output("normplot-card", "className"),
            Output("animation-card", "className"),
            Output("3dimage-card", "className"),
        ],
        Input("visualization-type", "value"),
        prevent_initial_call=True
    )
    def update_card_selection(selected_type):
        base_class = "vis-card"
        selected_class = "vis-card selected"
        
        logger.info(f"=== CARD SELECTION CALLBACK ===")
        logger.info(f"Selected type: {selected_type}")
        logger.info(f"Type of selected_type: {type(selected_type)}")
        
        slides_class = selected_class if selected_type == "slides" else base_class
        normplot_class = selected_class if selected_type == "normplot" else base_class
        animation_class = selected_class if selected_type == "animation" else base_class
        threed_class = selected_class if selected_type == "3dimage" else base_class
        
        logger.info(f"Card classes:")
        logger.info(f"  slides: {slides_class}")
        logger.info(f"  normplot: {normplot_class}")
        logger.info(f"  animation: {animation_class}")
        logger.info(f"  3dimage: {threed_class}")
        logger.info(f"=== END CARD SELECTION ===")
        
        return slides_class, normplot_class, animation_class, threed_class
    
    # Callback to update quick select button styling
    @app.callback(
        Output({"type": "quick-select-btn", "index": ALL}, "className"),
        Input("column-selector", "value"),
        prevent_initial_call=True
    )
    def update_quick_select_styling(selected_molecule):
        try:
            # Get the molecule list from cache first
            session_id = session.get('session_id', 'default')
            molecule_list = cache.get(f"{session_id}:molecules_list")
            
            print(f"[DEBUG] Quick select styling - selected_molecule: {selected_molecule}")
            print(f"[DEBUG] Molecule list: {molecule_list}")
            
            # If no molecule list available, return no_update to prevent errors
            if not molecule_list:
                print("[DEBUG] No molecule list available, returning no_update")
                return dash.no_update
            
            # Get the first 5 molecules for quick select buttons (same as layout)
            quick_select_molecules = molecule_list[:5]
            
            # Check if we have any molecules to display
            if len(quick_select_molecules) == 0:
                print("[DEBUG] No molecules to display, returning no_update")
                return dash.no_update
            
            # Check if the callback context has any outputs (buttons exist)
            ctx = dash.callback_context
            if not ctx.outputs_list:
                print("[DEBUG] No outputs in callback context, returning no_update")
                return dash.no_update
            
            classes = []
            
            # Generate classes for each quick select button (up to 5)
            for molecule in quick_select_molecules:
                if molecule == selected_molecule:
                    classes.append("quick-select-btn selected")
                else:
                    classes.append("quick-select-btn")
            
            print(f"[DEBUG] Generated {len(classes)} classes for {len(quick_select_molecules)} molecules: {classes}")
            print(f"[DEBUG] Number of outputs expected: {len(ctx.outputs_list)}")
            
            # Return exactly the number of classes we have (should match the number of buttons)
            return classes
            
        except Exception as e:
            print(f"Error in update_quick_select_styling: {e}")
            # In case of any error, return no_update to prevent callback crashes
            return dash.no_update
    
    # Enhanced molecule selection tracker with compound matrix regeneration
    @app.callback(
        Output("store-last-molecule", "data"),
        Input("column-selector", "value"),
        prevent_initial_call=True
    )
    def track_molecule_selection_and_regenerate(selected_molecule):
        """Track molecule selection and trigger compound matrix regeneration"""
        if selected_molecule:
            logger.info(f"MOLECULE SELECTION DETECTED: {selected_molecule}")
            
            # Get current molecule from cache to compare
            session_id = session.get('session_id', 'default')
            current_molecule = cache.get(f"{session_id}:current_molecule")
            
            # If molecule changed, clear compound matrix to force regeneration
            if current_molecule != selected_molecule:
                print(f"[DEBUG] Molecule changed from {current_molecule} to {selected_molecule}")
                cache.delete(f"{session_id}:compound_matrix")
                cache.set(f"{session_id}:current_molecule", selected_molecule)
                
                # Track the molecule change
                import time
                timestamp = time.time()
                molecule_data = {"molecule": selected_molecule, "timestamp": timestamp}
                return molecule_data
            
            # No change - return current data
            import time
            timestamp = time.time()
            return {"molecule": selected_molecule, "timestamp": timestamp}
            
        return dash.no_update
    
    # Combined callback to handle visualization output
    @app.callback(
        Output("visualization-output", "children"),
        [
            Input("generate-viz-btn", "n_clicks"),
            Input("visualization-type", "value"),
            Input("column-selector", "value"),
        ],
        prevent_initial_call=True,
    )
    def handle_visualization_output(generate_clicks, viz_type, molecule):
        # Check if we have compound matrix data available
        session_id = session.get('session_id', 'default')
        compound_matrix = cache.get(f"{session_id}:compound_matrix")
        
        logger.info(f"Visualization callback triggered - viz_type: {viz_type}, molecule: {molecule}, matrix available: {compound_matrix is not None}")
        
        # Get the actual selected molecule from the tracking or direct selection
        selected_molecule = molecule
        
        # Check if we have the required data
        if not viz_type or not selected_molecule:
            return html.Div([
                html.Div([
                    html.I(className="fas fa-info-circle", style={"color": "#2196f3", "marginRight": "8px"}),
                    html.Span("Please select both a visualization type and molecule to continue.")
                ], style={"color": "#2196f3", "textAlign": "center", "marginTop": "20px"})
            ])
        
        if compound_matrix is None:
            # Check if we have the basic data needed to generate the compound matrix
            df = cache.get(f"{session_id}:df")
            molecules_list = cache.get(f"{session_id}:molecules_list")
            warp_matrix = cache.get(f"{session_id}:warp_matrix")
            
            if df is not None and molecules_list is not None and warp_matrix is not None:
                # Data is available - compound matrix should be generated automatically
                return html.Div([
                    html.Div([
                        html.I(className="fas fa-cog fa-spin", style={"color": "#00bcd4", "marginRight": "8px"}),
                        html.Span(f"Generating compound matrix for {selected_molecule}... Please wait.")
                    ], style={"color": "#00bcd4", "textAlign": "center", "marginTop": "20px"}),
                    html.Div([
                        html.Span("This may take a few moments for large datasets.", style={"fontSize": "0.9rem", "color": "#b0bec5"})
                    ], style={"textAlign": "center", "marginTop": "8px"})
                ])
            else:
                # Missing basic data
                missing_items = []
                if df is None: missing_items.append("data")
                if molecules_list is None: missing_items.append("molecules")
                if warp_matrix is None: missing_items.append("warp matrix")
                
            return html.Div([
                html.Div([
                    html.I(className="fas fa-exclamation-triangle", style={"color": "#ff9800", "marginRight": "8px"}),
                        html.Span(f"Missing required data: {', '.join(missing_items)}. Please run the processing pipeline first.")
                    ], style={"color": "#ff9800", "textAlign": "center", "marginTop": "20px"}),
                    html.Div([
                        html.Button([
                            html.I(className="fas fa-arrow-left", style={"marginRight": "8px"}),
                            "Go Back to Upload"
                        ], id="back-to-form-btn", style={
                            "marginTop": "16px", "padding": "8px 16px", "background": "#00bcd4",
                            "color": "white", "border": "none", "borderRadius": "4px", "cursor": "pointer"
                        })
                    ], style={"textAlign": "center", "marginTop": "12px"})
        ])
        
        logger.info(f"Generating {viz_type} visualization for molecule {selected_molecule}")
        
        try:
            if viz_type == "3dimage":
                # For 3D visualization, don't show anything on main page - it will be handled by navigation
                return html.Div([
                    html.Div([
                        html.I(className="fas fa-cog fa-spin", style={"color": "#00bcd4", "marginRight": "8px"}),
                        html.Span("Preparing 3D visualization...")
                    ], style={"color": "#00bcd4", "textAlign": "center", "marginTop": "20px"}),
                    html.Div([
                        html.Span("Redirecting to 3D visualization page...", style={"fontSize": "0.9rem", "color": "#b0bec5"})
                    ], style={"textAlign": "center", "marginTop": "8px"})
                ])
            elif viz_type == "slides":
                return get_slides_layout(cache)
            elif viz_type == "normplot":
                return get_norm_layout(cache)
            elif viz_type == "animation":
                return get_animation_layout(cache)
            else:
                return html.Div([
                    html.Div([
                        html.I(className="fas fa-exclamation-circle", style={"color": "#f44336", "marginRight": "8px"}),
                        html.Span(f"Unknown visualization type: {viz_type}")
                    ], style={"color": "#f44336", "textAlign": "center", "marginTop": "20px"})
                ])
        except Exception as e:
            logger.error(f"Error generating {viz_type} visualization: {e}")
            return html.Div([
                html.Div([
                    html.I(className="fas fa-exclamation-circle", style={"color": "#f44336", "marginRight": "8px"}),
                    html.Span(f"Error generating visualization: {str(e)}")
                ], style={"color": "#f44336", "textAlign": "center", "marginTop": "20px"})
            ])
    
    # Combined callback to handle export visualization button
    @app.callback(
        [Output("export-viz-btn", "children"),
         Output("download-data", "data")],
        [Input("export-viz-btn", "n_clicks"),
         Input("download-data", "data")],
        [State("visualization-type", "value"),
         State("column-selector", "value")],
        prevent_initial_call=True
    )
    def handle_export_visualization(n_clicks, download_data, viz_type, molecule):
        ctx = dash.callback_context
        triggered_id = ctx.triggered_id
        
        # Handle export button click
        if triggered_id == "export-viz-btn":
            if not n_clicks:
                return dash.no_update, dash.no_update
            
            try:
                # Check if we have the required data
                if not viz_type or not molecule:
                    return [
                        html.I(className="fas fa-exclamation-circle"),
                        " Export Failed - Select Type & Molecule"
                    ], dash.no_update
                
                # Get the compound matrix from cache
                compound_matrix = cache.get(f"{session['session_id']}:compound_matrix")
                if compound_matrix is None:
                    return [
                        html.I(className="fas fa-exclamation-circle"),
                        " Export Failed - No Data Available"
                    ], dash.no_update
                
                # Create export data
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"metavision_{viz_type}_{molecule}_{timestamp}"
                
                if viz_type == "slides":
                    # Export as NIfTI (.nii.gz) file for 3D data
                    import nibabel as nib
                    
                    # Create NIfTI image from compound matrix
                    # Reshape matrix from (z,y,x) to (x,y,z) for NIfTI format
                    output_matrix = np.transpose(compound_matrix, (2, 1, 0))
                    nii_img = nib.Nifti1Image(output_matrix, affine=np.eye(4))
                    
                    # Save to buffer using to_bytes()
                    nii_buffer = io.BytesIO()
                    nii_buffer.write(nii_img.to_bytes())
                    nii_buffer.seek(0)
                    
                    return [
                        html.I(className="fas fa-check"),
                        " Export Complete"
                    ], dcc.send_bytes(nii_buffer.getvalue(), f"{filename}.nii.gz")
                
                elif viz_type == "normplot":
                    # Export as CSV with statistics
                    import pandas as pd
                    
                    # Calculate statistics for each slice
                    stats_data = []
                    for i in range(compound_matrix.shape[0]):
                        slice_data = compound_matrix[i]
                        stats = {
                            'slice': i + 1,
                            'mean': float(np.mean(slice_data)),
                            'std': float(np.std(slice_data)),
                            'min': float(np.min(slice_data)),
                            'max': float(np.max(slice_data)),
                            'median': float(np.median(slice_data))
                        }
                        stats_data.append(stats)
                    
                    df_stats = pd.DataFrame(stats_data)
                    csv_string = df_stats.to_csv(index=False)
                    
                    return [
                        html.I(className="fas fa-check"),
                        " Export Complete"
                    ], dcc.send_string(csv_string, f"{filename}.csv")
                
                elif viz_type == "animation":
                    # Export as NIfTI (.nii.gz) file for 3D data
                    import nibabel as nib
                    
                    # Create NIfTI image from compound matrix
                    # Reshape matrix from (z,y,x) to (x,y,z) for NIfTI format
                    output_matrix = np.transpose(compound_matrix, (2, 1, 0))
                    nii_img = nib.Nifti1Image(output_matrix, affine=np.eye(4))
                    
                    # Save to buffer using to_bytes()
                    nii_buffer = io.BytesIO()
                    nii_buffer.write(nii_img.to_bytes())
                    nii_buffer.seek(0)
                    
                    return [
                        html.I(className="fas fa-check"),
                        " Export Complete"
                    ], dcc.send_bytes(nii_buffer.getvalue(), f"{filename}.nii.gz")
                
                elif viz_type == "3dimage":
                    # Export as NIfTI (.nii.gz) file
                    import nibabel as nib
                    
                    # Create NIfTI image from compound matrix
                    # Reshape matrix from (z,y,x) to (x,y,z) for NIfTI format
                    output_matrix = np.transpose(compound_matrix, (2, 1, 0))
                    nii_img = nib.Nifti1Image(output_matrix, affine=np.eye(4))
                    
                    # Save to buffer using to_bytes()
                    nii_buffer = io.BytesIO()
                    nii_buffer.write(nii_img.to_bytes())
                    nii_buffer.seek(0)
                    
                    return [
                        html.I(className="fas fa-check"),
                        " Export Complete"
                    ], dcc.send_bytes(nii_buffer.getvalue(), f"{filename}.nii.gz")
                
                else:
                    return [
                        html.I(className="fas fa-exclamation-circle"),
                        " Export Failed - Unknown Type"
                    ], dash.no_update
                    
            except Exception as e:
                logger.error(f"Export error: {e}")
                return [
                    html.I(className="fas fa-exclamation-circle"),
                    f" Export Failed - {str(e)}"
                ], dash.no_update
        
        # Handle download completion (reset button)
        elif triggered_id == "download-data":
            if download_data:
                # Reset button after successful export
                return [
                    html.I(className="fas fa-download"),
                    " Export Visualization"
                ], dash.no_update
        
        return dash.no_update, dash.no_update

    # Remove all 2D slice callbacks since 2D Brain Slice has been removed
    # The colormap-radio-animation callback is already handled in animation_callback.py