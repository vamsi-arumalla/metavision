etavision-master % python main.py
Traceback (most recent call last):
  File "/Users/somisettysaiharsha/Downloads/MetaVision_Project/metavision-master/main.py", line 30, in <module>
    from visualization.visualization_callback import register_visualization_callback
  File "/Users/somisettysaiharsha/Downloads/MetaVision_Project/metavision-master/visualization/visualization_callback.py", line 69
    for i, n in enumerate(quick_select_clicks):
    ^
from dash import Output, Input, State, html
import logging
import os
import pandas as pd
import io
import base64
from flask import session
from metavision.MetaAlign3D import create_compound_matrix, calculate_warp_matrix
from metavision.MetaAlign3D import MetaAlign3D
from metavision.MetaInterp3D import MetaInterp3D
import os
import numpy as np
from metavision.utils import get_ref_compound
from metavision.MetaNorm3D import MetaNorm3D
from flask import session
from metavision.MetaImpute3D import MetaImpute3D
import dash
from dash import ctx

logger = logging.getLogger("metavision")


def normalize(molecule, df, molecule_list, filename, cache):
    logger.info("Normalizing molecule data...")
    meta_norm = MetaNorm3D(df, molecule_list[0])
    logger.info("MetaNorm3D is completed")
    norm_df = meta_norm.totalsum_norm()
    cache.set(f"{session['session_id']}:norm_df", norm_df)
    logger.info(f"totalsum norm df head: {norm_df.head}")
    data = meta_norm.section_norm()
    cache.set(f"{session['session_id']}:df", data)
    logger.info(f"Calling MetaAlign3D in start_normalization process")
    session_id = session["session_id"]
    warp_matrix = cache.get(f"{session_id}:warp_matrix")
    meta_normalize = MetaAlign3D(
        filename, df, molecule, molecule_list[0], warp_matrix, reverse=True
    )
    logger.info("MetaAlign3D is completed in start_normalization process")
    compound_matrix = meta_normalize.create_compound_matrix()
    # cache.set(f"{session_id}:compound_matrix", compound_matrix)
    logger.info(f"Sum of compound_matrix: {compound_matrix.sum()}")
    return data, compound_matrix


def align(molecule, df, molecule_list, filename, warp_matrix, cache):
    logger.info("Aligning molecule data...")
    meta_align = MetaAlign3D(
        filename,
        df,
        molecule,
        molecule_list[0],
        warp_matrix,
        reverse=True,
    )
    logger.info("MetaAlign3D is completed in start_alignment process")
    compound_matrix = meta_align.create_compound_matrix()
    compound_matrix = meta_align.seq_align()
    logger.info(f"Sum of compound_matrix: {compound_matrix.sum()}")
    return compound_matrix

def impute_mat(radius, compound_matrix):
    logger.info("Imputing molecule data...")
    meta_impute = MetaImpute3D(compound_matrix, radius)
    logger.info("MetaImpute3D is completed")
    compound_matrix = meta_impute.seq_impute()
    logger.info(f"Sum of compound_matrix: {compound_matrix.sum()}")
    return compound_matrix

def interpolate_mat(slices, compound_matrix):
    logger.info("Interpolating molecule data...")
    meta_interp = MetaInterp3D(compound_matrix, slices)
    logger.info("MetaInterp3D is completed")
    compound_matrix = meta_interp.interp()
    logger.info(f"Sum of compound_matrix: {compound_matrix.sum()}")
    return compound_matrix

def register_form_callback(app, cache):
    """
    Register the callback for the form submission.
    """

    @app.callback(
        Output("selected-page", "data"),
        [
            Input("landing-go-btn", "n_clicks"),
            Input("go-visualization-btn", "n_clicks"),
            Input("go-export-btn", "n_clicks"),
            Input("back-to-dashboard-btn", "n_clicks"),
            Input("processed", "data"),
        ],
        prevent_initial_call=True,
    )
    def handle_navigation_and_processing(n_landing, n_vis, n_exp, n_back, processed):
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update
        
        btn_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        # Handle navigation button clicks
        if btn_id == "landing-go-btn":
            return "dashboard"
        elif btn_id == "back-to-dashboard-btn":
            return "dashboard"
        # Handle processing completion
        elif btn_id == "processed" and processed is True:
            return "postprocessing"
        
        return dash.no_update

    @app.callback(
        [Output("standard-upload-area", "style", allow_duplicate=True),
         Output("large-file-upload-area", "style", allow_duplicate=True),
         Output("standard-upload-btn", "style", allow_duplicate=True),
         Output("chunked-upload-btn", "style", allow_duplicate=True)],
        [Input("standard-upload-btn", "n_clicks"),
         Input("chunked-upload-btn", "n_clicks")],
        prevent_initial_call=True
    )
    def handle_upload_method_selection(standard_clicks, chunked_clicks):
        """Handle switching between standard and chunked upload methods"""
        ctx = dash.callback_context
        if not ctx.triggered:
            return {"display": "block"}, {"display": "none"}, {
                "background": "#00bcd4", "color": "#fff", "border": "none",
                "padding": "12px 24px", "borderRadius": "8px 0 0 8px",
                "fontSize": "1rem", "cursor": "pointer", "flex": "1"
            }, {
                "background": "#37474f", "color": "#fff", "border": "none",
                "padding": "12px 24px", "borderRadius": "0 8px 8px 0",
                "fontSize": "1rem", "cursor": "pointer", "flex": "1"
            }
        
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        if triggered_id == "standard-upload-btn":
            # Show standard upload, hide chunked upload
            return {"display": "block"}, {"display": "none"}, {
                "background": "#00bcd4", "color": "#fff", "border": "none",
                "padding": "12px 24px", "borderRadius": "8px 0 0 8px",
                "fontSize": "1rem", "cursor": "pointer", "flex": "1"
            }, {
                "background": "#37474f", "color": "#fff", "border": "none",
                "padding": "12px 24px", "borderRadius": "0 8px 8px 0",
                "fontSize": "1rem", "cursor": "pointer", "flex": "1"
            }
        elif triggered_id == "chunked-upload-btn":
            # Show chunked upload, hide standard upload
            return {"display": "none"}, {"display": "block"}, {
                "background": "#37474f", "color": "#fff", "border": "none",
                "padding": "12px 24px", "borderRadius": "8px 0 0 8px",
                "fontSize": "1rem", "cursor": "pointer", "flex": "1"
            }, {
                "background": "#00bcd4", "color": "#fff", "border": "none",
                "padding": "12px 24px", "borderRadius": "0 8px 8px 0",
                "fontSize": "1rem", "cursor": "pointer", "flex": "1"
            }
        
        return {"display": "block"}, {"display": "none"}, {
            "background": "#00bcd4", "color": "#fff", "border": "none",
            "padding": "12px 24px", "borderRadius": "8px 0 0 8px",
            "fontSize": "1rem", "cursor": "pointer", "flex": "1"
        }, {
            "background": "#37474f", "color": "#fff", "border": "none",
            "padding": "12px 24px", "borderRadius": "0 8px 8px 0",
            "fontSize": "1rem", "cursor": "pointer", "flex": "1"
        }

    @app.callback(
        [Output("chunked-upload-status", "children", allow_duplicate=True),
         Output("chunked-upload-progress", "children", allow_duplicate=True),
         Output("large-file-display", "children", allow_duplicate=True),
         Output("large-file-display", "style", allow_duplicate=True),
         Output("large-file-progress-bar", "style", allow_duplicate=True),
         Output("large-file-progress-inner", "style", allow_duplicate=True),
         Output("large-file-progress-text", "children", allow_duplicate=True),
         Output("chunked-upload-processed", "data", allow_duplicate=True)],
        [Input("chunked-upload-btn", "n_clicks"),
         Input("remove-large-file-btn", "n_clicks")],
        prevent_initial_call=True
    )
    def handle_large_file_selection(select_clicks, remove_clicks):
        import os
        ctx = dash.callback_context
        if not ctx.triggered:
            return "", "", "", {"display": "none"}, {"display": "none"}, {"width": "0%"}, "0%", False
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if triggered_id == "chunked-upload-btn":
            # Reset UI so a new file can be selected
            return (
                html.Div([
                    html.I(className="fas fa-info-circle", style={"marginRight": "8px"}),
                    html.Span("Large file upload selected. Click the upload area to select your file.", style={"color": "#2196f3"})
                ], className="status-indicator info"),
                html.Div([
                    html.I(className="fas fa-clock", style={"marginRight": "8px"}),
                    html.Span("After upload, use 'Run Processing Pipeline' button", style={"color": "#ff9800"})
                ], className="status-indicator warning"),
                "Ready for file upload",  # Show ready message
                {"display": "block", "color": "#b0bec5"},  # Show file display area
                {"display": "block"},  # Show progress bar
                {"width": "0%"},
                "0%",
                False  # Don't set chunked upload as processed until upload completes
            )
        elif triggered_id == "remove-large-file-btn":
            # Delete all CSV files in the workspaces directory
            workspace_dir = "workspaces"
            for f in os.listdir(workspace_dir):
                if f.endswith(".csv"):
                    try:
                        os.remove(os.path.join(workspace_dir, f))
                    except Exception:
                        pass
            return (
                "",
                "",
                "",
                {"display": "none"},
                {"display": "none"},
                {"width": "0%"},
                "0%",
                False  # Reset chunked upload processed state
            )
        return "", "", True, "", {"display": "none"}, {"display": "none"}, {"width": "0%"}, "0%", False

    @app.callback(
        [Output("large-file-progress-inner", "style", allow_duplicate=True),
         Output("large-file-progress-text", "children", allow_duplicate=True)],
        [Input("chunked-upload-btn", "n_clicks")],
        prevent_initial_call=True
    )
    def simulate_upload_progress(n_clicks):
        """Simulate upload progress for large files"""
        if n_clicks:
            # Simulate progress from 0% to 100%
            import time
            time.sleep(0.1)  # Small delay for visual effect
            return {"width": "100%", "height": "100%", "background": "linear-gradient(90deg, #00bcd4, #00acc1)", "borderRadius": "8px", "transition": "width 0.3s ease"}, "100%"
        return {"width": "0%"}, "0%"

    # Test progress callback for debugging
    @app.callback(
        [Output("large-file-display", "children", allow_duplicate=True),
         Output("large-file-display", "style", allow_duplicate=True),
         Output("large-file-progress-bar", "style", allow_duplicate=True),
         Output("large-file-progress-inner", "style", allow_duplicate=True),
         Output("large-file-progress-text", "children", allow_duplicate=True),
         Output("chunked-upload-status", "children", allow_duplicate=True)],
        Input("test-progress-btn", "n_clicks"),
        prevent_initial_call=True
    )
    def test_progress_display(n_clicks):
        if not n_clicks:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
            
        return (
            "🧪 Test File - sample_data.csv (850.5 MB)",
            {"display": "block", "color": "#4caf50", "fontWeight": "500", "textAlign": "center", "marginTop": "16px", "padding": "12px", "background": "rgba(76, 175, 80, 0.1)", "border": "1px solid rgba(76, 175, 80, 0.3)", "borderRadius": "8px"},
            {"display": "block", "position": "relative", "width": "100%", "height": "28px", "background": "rgba(0,0,0,0.1)", "borderRadius": "14px", "marginTop": "16px", "border": "1px solid rgba(0,0,0,0.1)"},
            {"width": "75%", "height": "100%", "background": "linear-gradient(90deg, #00bcd4, #00acc1)", "borderRadius": "8px", "transition": "width 0.5s ease", "boxShadow": "0 2px 8px rgba(0, 188, 212, 0.4)"},
            "75%",
                html.Div([
                html.I(className="fas fa-test-tube", style={"marginRight": "8px", "color": "#00bcd4"}),
                html.Span("Test mode: Progress bar and file display working!")
            ])
        )
            
    # Removed process_chunked_file callback - now handled by main processing pipeline

    @app.callback(
        [Output("unified-upload-output", "style", allow_duplicate=True), Output("unified-upload-progress-bar", "style", allow_duplicate=True)],
        Input("chunked-upload-processed", "data"),
        prevent_initial_call=True
    )
    def hide_standard_upload(chunked_processed):
        if chunked_processed:
            return {"display": "none"}, {"display": "none"}
        return {"display": "block"}, {"display": "block"}

    # Removed duplicate callback - functionality merged into handle_unified_file_upload

    # Removed redundant callback - functionality merged into handle_unified_file_upload

    @app.callback(
        [Output("standard-upload-area", "style", allow_duplicate=True),
         Output("unified-upload-output", "children", allow_duplicate=True),
         Output("unified-upload-progress-bar", "style", allow_duplicate=True),
         Output("unified-upload-progress-inner", "style", allow_duplicate=True),
         Output("unified-upload-progress-text", "children", allow_duplicate=True)],
        [Input("unified-upload-data", "contents"),
         Input("unified-upload-data", "filename")],
        prevent_initial_call=True
    )
    def handle_unified_file_upload(contents, filename):
        import os
        import base64
        import io
        from flask import session
        if not contents or not filename:
            # No file uploaded: show upload area, hide progress bar and filename
            return {"display": "block"}, "", {"display": "none"}, {"width": "0%"}, "0%"
        else:
            # Save uploaded file to workspaces
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            workspace_dir = "workspaces"
            file_path = os.path.join(workspace_dir, filename)
            with open(file_path, "wb") as f:
                f.write(decoded)
            # Store the uploaded filename in cache for this session
            session_id = session["session_id"]
            cache.set(f"{session_id}:large_filename", filename)
            # Show filename and progress bar at 100%
            file_display = (
                html.Div([
                    html.I(className="fas fa-file-csv", style={"color": "#00bcd4", "marginRight": "8px", "fontSize": "1.2rem"}),
                    html.Span(filename, style={"color": "#fff", "fontWeight": "500", "fontSize": "1.1rem"})
                ], className="file-name")
            )
            return {"display": "block"}, file_display, {"display": "block"}, {"width": "100%"}, "100%"

    # Removed redundant callback - functionality merged into handle_unified_file_upload

    @app.callback(
        [Output("processed", "data", allow_duplicate=True),
         Output("molecule", "data", allow_duplicate=True),
         Output("loading-output", "children"),
         Output("selected-page", "data", allow_duplicate=True)],
        [
            Input("run-button", "n_clicks"),
        ],
        [
            State("unified-upload-data", "contents"),
            State("unified-upload-data", "filename"),
            State("interpolate-checkbox", "value"),
            State("slices-input", "value"),
            State("impute-checkbox", "value"),
            State("radius-input", "value"),
            State("selected-page", "data"),
            State("processed", "data"),
        ],
        prevent_initial_call=True,
    )


    def handle_processing(
        n_run,
        contents, filename, interpolate, slices, impute, radius,
        selected_page, processed
    ):
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update, dash.no_update, dash.no_update
        btn_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        if btn_id == "run-button":
            # Check for both standard upload and chunked upload
            chunked_file = None
            if contents is None:
                # Check for chunked uploaded files in workspaces directory
                workspace_dir = "workspaces"
                if os.path.exists(workspace_dir):
                    csv_files = [f for f in os.listdir(workspace_dir) if f.endswith(".csv")]
                    if csv_files:
                        # Use the most recent CSV file
                        chunked_file = max(csv_files, key=lambda f: os.path.getmtime(os.path.join(workspace_dir, f)))
                        filename = chunked_file
                        logger.info(f"Found chunked uploaded file: {chunked_file}")
                
                if not chunked_file:
                    logger.warning("No file uploaded.")
                    return False, "", html.Div([
                        html.I(className="fas fa-exclamation-circle error-icon"),
                        html.Span("Please upload a file first using either Standard Upload or Large File Upload.")
                    ], className="error-message"), dash.no_update
            
            # Show initial processing status with progress steps
            initial_processing_status = html.Div([
                html.Div([
                    html.I(className="fas fa-spinner fa-spin", style={"color": "#00bcd4", "marginRight": "8px"}),
                    html.Span("Processing file... This may take a few minutes.")
                ], style={"color": "#00bcd4", "marginBottom": "8px"}),
                html.Div([
                    html.Div([
                        html.Span("Step 1: Loading Data", style={"color": "#4caf50", "fontWeight": "500"}),
                        html.Span(" (25%)", style={"color": "#b0bec5"})
                    ], style={"marginBottom": "4px"}),
                    html.Div([
                        html.Span("Step 2: Normalization", style={"color": "#ff9800", "fontWeight": "500"}),
                        html.Span(" (50%)", style={"color": "#b0bec5"})
                    ], style={"marginBottom": "4px"}),
                    html.Div([
                        html.Span("Step 3: 3D Alignment", style={"color": "#2196f3", "fontWeight": "500"}),
                        html.Span(" (75%)", style={"color": "#b0bec5"})
                    ], style={"marginBottom": "4px"}),
                    html.Div([
                        html.Span("Step 4: Processing", style={"color": "#9c27b0", "fontWeight": "500"}),
                        html.Span(" (100%)", style={"color": "#b0bec5"})
                    ])
                ], style={"background": "rgba(0,0,0,0.1)", "padding": "12px", "borderRadius": "8px", "marginTop": "12px"})
            ])
            
            try:
                # Handle both standard and chunked uploads
                logger.info(f"Processing uploaded file: {filename}")
                cache.set(f"{session['session_id']}:filename", filename)
                
                if chunked_file:
                    # File is already saved to disk from chunked upload
                    logger.info(f"Processing chunked uploaded file: {chunked_file}")
                    df = pd.read_csv(os.path.join("workspaces", chunked_file))
                    
                else:
                    # Standard upload - decode contents
                    logger.info(f"Processing standard upload - Contents type: {type(contents)}, length: {len(contents) if contents else 'None'}")
                    if contents:
                        logger.info(f"Contents preview (first 100 chars): {contents[:100]}")
                
                    # Standard upload processing - decode contents
                    if contents is None:
                        logger.error("File contents are None")
                        return False, "", html.Div([
                            html.I(className="fas fa-exclamation-circle error-icon"),
                            html.Span("No file content received. Please try uploading again.")
                        ], className="error-message")
                    
                    # Handle different upload formats
                    try:
                        # Check if contents contains base64 prefix
                        if "," in contents:
                            # Try standard format: "data:type;base64,content"
                            split_contents = contents.split(",", 1)
                            if len(split_contents) == 2:
                                content_type, content_string = split_contents
                                logger.info(f"Found standard format with content type: {content_type}")
                            else:
                                logger.warning("Multiple commas found in contents, using everything after first comma")
                                content_string = ",".join(split_contents[1:])
                        else:
                            # Contents might be just base64 without prefix
                            logger.info("No comma found, treating entire contents as base64")
                            content_string = contents
                        
                        # Try to decode base64
                        decoded = base64.b64decode(content_string)
                        
                    except Exception as e:
                        logger.error(f"Error processing file contents: {str(e)}")
                        # Try treating contents as plain text (fallback)
                        try:
                            decoded = contents.encode('utf-8')
                            logger.info("Fallback: treating contents as plain text")
                        except Exception as fallback_e:
                            logger.error(f"Fallback failed: {str(fallback_e)}")
                            return False, "", html.Div([
                                html.I(className="fas fa-exclamation-circle error-icon"),
                                html.Span(f"Could not process file contents: {str(e)}")
                            ], className="error-message")
                    
                    # Robust CSV reading: use chunked reading for large files
                    try:
                        # Try to decode the content as UTF-8
                        try:
                            decoded_text = decoded.decode("utf-8")
                        except UnicodeDecodeError:
                            # If UTF-8 fails, try other encodings
                            try:
                                decoded_text = decoded.decode("latin1")
                                logger.info("Used latin1 encoding for decoding")
                            except UnicodeDecodeError:
                                decoded_text = decoded.decode("cp1252")
                                logger.info("Used cp1252 encoding for decoding")
                        
                        logger.info(f"Decoded text length: {len(decoded_text)}")
                        logger.info(f"Decoded text preview (first 200 chars): {decoded_text[:200]}")
                        
                        if len(decoded) > 10 * 1024 * 1024:  # >10MB, use chunked reading
                            chunk_iter = pd.read_csv(io.StringIO(decoded_text), on_bad_lines='skip', chunksize=100000)
                            df = pd.concat(chunk_iter, ignore_index=True)
                            logger.info("Used chunked reading for large file")
                        else:
                            df = pd.read_csv(io.StringIO(decoded_text), on_bad_lines='skip')
                            logger.info("Used standard reading for file")
                            
                    except Exception as e:
                        logger.error(f"Error reading CSV: {str(e)}")
                        logger.error(f"Decoded content type: {type(decoded)}, length: {len(decoded) if decoded else 'None'}")
                        return False, "", html.Div([
                            html.I(className="fas fa-exclamation-circle error-icon"),
                            html.Span(f"Error reading CSV: {str(e)}")
                        ], className="error-message")

                # Defensive: Remove any rows or columns that are all-NaN or empty
                df = df.dropna(how='all')
                df = df.dropna(axis=1, how='all')
                df = df.loc[:, df.notna().any()]

                # Defensive: Print shape and sample for debugging
                print("DF shape after cleaning:", df.shape)
                print("First 5 rows after cleaning:", df.head())

                if "region" in df.columns:
                    df = df.rename(columns={"region": "tissue_id"})
                if "tissue_id" not in df.columns:
                    logger.error("CSV file must contain a 'tissue_id' or 'region' column.")
                    return False, "", html.Div([
                        html.I(className="fas fa-exclamation-circle error-icon"),
                        html.Span("CSV file must contain a 'tissue_id' or 'region' column.")
                    ], className="error-message")
                df = df[df["tissue_id"].notna()]

                # Defensive: If you do any row unpacking later, always check length
                # For example, if you do:
                # for row in df.values:
                #     if len(row) != len(df.columns):
                #         print("Skipping malformed row:", row)
                #         continue
                #     ...unpack...

                tissue_ids = df["tissue_id"].unique().tolist()
                cache.set(f"{session['session_id']}:tissue_ids", tissue_ids)
                columns = df.columns[1:].tolist()
                metadata_columns = [
                    "spotId", "raster", "x", "y", "z", "Date", "Class", "tissue_id", "roi",
                ]
                molecules_list = [col for col in columns if col not in metadata_columns]
                cache.set(f"{session['session_id']}:molecules_list", molecules_list)
                cache.set(f"{session['session_id']}:df", df)
                logger.info(
                    f"DataFrame loaded with {len(df)} rows and {len(df.columns)} columns."
                )
                logger.info(f"Found {len(molecules_list)} molecule columns.")
                logger.debug(f"Molecule columns: {molecules_list}")
                ref_coumpound = get_ref_compound(df, molecules_list[0]) if molecules_list else None
                if not ref_coumpound:
                    logger.error("No valid molecule columns found in CSV.")
                    return False, "", html.Div([
                        html.I(className="fas fa-exclamation-circle error-icon"),
                        html.Span("No valid molecule columns found in CSV.")
                    ], className="error-message")
                if os.path.exists(
                    os.path.join("workspaces", filename.replace(".csv", ".npy"))
                ):
                    logger.info(f"Warp Matrix already created for this file. Loading from disk...")
                    warp_matrix = np.load(
                        os.path.join("workspaces", filename.replace(".csv", ".npy"))
                    )
                    # CRITICAL: Store the loaded warp matrix in cache for molecule switching
                    cache.set(f"{session['session_id']}:warp_matrix", warp_matrix)
                    logger.info("Warp matrix loaded from disk and stored in cache")
                else:
                    logger.info(f"Most prevalent compound: {ref_coumpound}")
                    try:
                        logger.info("Creating reference compound matrix...")
                        ref_compound_matrix = create_compound_matrix(
                            df, ref_coumpound, reverse=True
                        )
                        logger.info("Creating warp matrix...")
                        warp_matrix = calculate_warp_matrix(ref_compound_matrix)
                        cache.set(f"{session['session_id']}:warp_matrix", warp_matrix)
                        np.save(
                            os.path.join("workspaces", filename.replace(".csv", ".npy")),
                            warp_matrix,
                        )
                    except Exception as e:
                        logger.error(f"Error creating warp matrix: {str(e)}")
                        return False, "", html.Div([
                            html.I(className="fas fa-exclamation-circle error-icon"),
                            html.Span(f"Error creating warp matrix: {str(e)}")
                        ], className="error-message")
                logger.info("Starting to load DataFrame into NumPy arrays...")
                try:
                    df, compound_matrix = normalize(
                        ref_coumpound, df, molecules_list, filename, cache
                    )
                    compound_matrix = align(
                        ref_coumpound, df, molecules_list, filename, warp_matrix, cache
                    )
                except Exception as e:
                    logger.error(f"Error in normalization/alignment: {str(e)}")
                    return False, "", html.Div([
                        html.I(className="fas fa-exclamation-circle error-icon"),
                        html.Span(f"Error in normalization/alignment: {str(e)}")
                    ], className="error-message")
                if interpolate == [] and impute == []:
                    logger.info("No interpolation or imputation selected.")
                    cache.set(f"{session['session_id']}:compound_matrix", compound_matrix)
                else:
                    if impute != []:
                        logger.info("Imputation selected.")
                        try:
                            compound_matrix = impute_mat(radius, compound_matrix)
                        except Exception as e:
                            logger.error(f"Error in imputation: {str(e)}")
                            return False, "", html.Div([
                                html.I(className="fas fa-exclamation-circle error-icon"),
                                html.Span(f"Error in imputation: {str(e)}")
                            ], className="error-message")
                    if interpolate != []:
                        logger.info("Interpolation selected.")
                        try:
                            compound_matrix = interpolate_mat(slices, compound_matrix)
                        except Exception as e:
                            logger.error(f"Error in interpolation: {str(e)}")
                            return False, "", html.Div([
                                html.I(className="fas fa-exclamation-circle error-icon"),
                                html.Span(f"Error in interpolation: {str(e)}")
                            ], className="error-message")
                    cache.set(f"{session['session_id']}:compound_matrix", compound_matrix)
                cache.set(f"{session['session_id']}:interpolate", interpolate)
                cache.set(f"{session['session_id']}:impute", impute)
                cache.set(f"{session['session_id']}:slices", slices)
                cache.set(f"{session['session_id']}:radius", radius)
                cache.set(f"{session['session_id']}:ref_compound", ref_coumpound)
                
                success_message = html.Div([
                    html.Div([
                        html.I(className="fas fa-check-circle", style={"color": "#4caf50", "marginRight": "8px"}),
                        html.Span(f"Successfully processed {filename}! (100%)")
                    ], style={"color": "#4caf50", "marginBottom": "8px"}),
                    html.Div([
                        html.I(className="fas fa-file-alt", style={"color": "#2196f3", "marginRight": "8px"}),
                        html.Span(f"Dataset: {filename} ({len(df)} rows)")
                    ], style={"color": "#2196f3", "marginBottom": "8px"}),
                    html.Div([
                        html.I(className="fas fa-cube", style={"color": "#00bcd4", "marginRight": "8px"}),
                        html.Span(f"3D matrix shape: {compound_matrix.shape}")
                    ], style={"color": "#00bcd4", "marginBottom": "8px"}),
                    html.Div([
                        html.I(className="fas fa-arrow-right", style={"color": "#ff9800", "marginRight": "8px"}),
                        html.Span("Redirecting to visualization...")
                    ], style={"color": "#ff9800", "marginBottom": "8px"}),
                    html.Div([
                        html.Div([
                            html.Span("✓ Step 1: Data Loading", style={"color": "#4caf50", "fontWeight": "500"}),
                            html.Span(" (25%)", style={"color": "#b0bec5"})
                        ], style={"marginBottom": "4px"}),
                        html.Div([
                            html.Span("✓ Step 2: Normalization", style={"color": "#4caf50", "fontWeight": "500"}),
                            html.Span(" (50%)", style={"color": "#b0bec5"})
                        ], style={"marginBottom": "4px"}),
                        html.Div([
                            html.Span("✓ Step 3: 3D Alignment", style={"color": "#4caf50", "fontWeight": "500"}),
                            html.Span(" (75%)", style={"color": "#b0bec5"})
                        ], style={"marginBottom": "4px"}),
                        html.Div([
                            html.Span("✓ Step 4: Processing", style={"color": "#4caf50", "fontWeight": "500"}),
                            html.Span(" (100%)", style={"color": "#b0bec5"})
                        ])
                    ], style={"background": "rgba(76,175,80,0.1)", "padding": "12px", "borderRadius": "8px", "marginTop": "12px", "border": "1px solid rgba(76,175,80,0.3)"})
                ], className="success-message")
                
                # Navigate to visualization after successful processing
                return True, molecules_list[0], success_message, "visualization"
                
            except Exception as e:
                logger.error(f"Error processing uploaded file: {str(e)}")
                return False, "", html.Div([
                    html.I(className="fas fa-exclamation-triangle", style={"color": "#f44336", "marginRight": "8px"}),
                    html.Span(f"Error: {str(e)}")
                ], className="error-message"), dash.no_update
        
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    @app.callback(
        Output("postprocessing-content", "children"),
        [
            Input("go-visualization-btn", "n_clicks"),
            Input("go-export-btn", "n_clicks"),
        ],
        prevent_initial_call=True,
    )
    def handle_postprocessing_content(n_vis, n_exp):
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update
        
        btn_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        if btn_id == "go-visualization-btn":
            # Return visualization layout
            from visualization.visualization_layout import get_visualization_layout
            return get_visualization_layout(cache)
        elif btn_id == "go-export-btn":
            # Return export layout
            from export.export_layout import get_export_layout
            return get_export_layout(cache)
        
        return dash.no_update

    @app.callback(
        Output("molecule", "data", allow_duplicate=True),
        Input("column-selector", "value"),
        prevent_initial_call=True
    )
    def update_compound_matrix_for_molecule(selected_molecule):
        """Generate molecule-specific compound matrix when a molecule is selected"""
        if not selected_molecule:
            logger.info("No molecule selected, returning no_update")
            return dash.no_update
            
        try:
            session_id = session.get("session_id")
            if not session_id:
                logger.warning("No session_id found")
                return dash.no_update
                
            logger.info(f"Processing molecule selection: {selected_molecule}")
            
            # Check if we have the necessary data
            df = cache.get(f"{session_id}:df")
            molecules_list = cache.get(f"{session_id}:molecules_list")
            warp_matrix = cache.get(f"{session_id}:warp_matrix")
            
            # Check both possible filename keys (for compatibility)
            filename = cache.get(f"{session_id}:filename") or cache.get(f"{session_id}:large_filename")
            
            logger.info(f"Available data - df: {df is not None}, molecules: {molecules_list}, warp: {warp_matrix is not None}, filename: {filename}")
            
            if df is None or molecules_list is None or warp_matrix is None:
                logger.warning(f"Missing data for molecule compound matrix generation: df={df is not None}, molecules={molecules_list is not None}, warp={warp_matrix is not None}")
                
                # If warp matrix is missing, try to reload it from disk
                if warp_matrix is None and filename:
                    warp_file_path = os.path.join("workspaces", filename.replace(".csv", ".npy"))
                    if os.path.exists(warp_file_path):
                        logger.info(f"Attempting to reload warp matrix from disk: {warp_file_path}")
                        try:
                            warp_matrix = np.load(warp_file_path)
                            cache.set(f"{session_id}:warp_matrix", warp_matrix)
                            logger.info("Successfully reloaded warp matrix from disk")
                        except Exception as e:
                            logger.error(f"Failed to reload warp matrix: {e}")
                            return dash.no_update
                    else:
                        logger.error(f"Warp matrix file not found: {warp_file_path}")
                        return dash.no_update
                else:
                    return dash.no_update
                
            if selected_molecule not in molecules_list:
                logger.warning(f"Selected molecule {selected_molecule} not in molecules list {molecules_list}")
                return dash.no_update
                
            logger.info(f"Generating compound matrix for molecule: {selected_molecule}")
            logger.info(f"Using reference molecule: {molecules_list[0]}")
            logger.info(f"Data shape: {df.shape if df is not None else 'None'}")
            
            # Check if compound matrix already exists for this molecule
            existing_matrix = cache.get(f"{session_id}:compound_matrix")
            current_cached_molecule = cache.get(f"{session_id}:current_molecule")
            
            if existing_matrix is not None and current_cached_molecule == selected_molecule:
                print(f"[DEBUG] Using existing compound matrix for molecule: {selected_molecule}")
                return dash.no_update
            
            print(f"[DEBUG] Generating new compound matrix for molecule: {selected_molecule}")
            
            # Generate compound matrix for the selected molecule
            meta_align = MetaAlign3D(filename, df, selected_molecule, molecules_list[0], warp_matrix, reverse=True)
            
            logger.info("Created MetaAlign3D instance, creating compound matrix...")
            compound_matrix = meta_align.create_compound_matrix()
            
            logger.info(f"Compound matrix created with shape: {compound_matrix.shape}")
            logger.info("Applying sequential alignment...")
            compound_matrix = meta_align.seq_align()
            
            # Cache the compound matrix with molecule-specific key
            cache.set(f"{session_id}:compound_matrix", compound_matrix)
            cache.set(f"{session_id}:current_molecule", selected_molecule)
            
            logger.info(f"Successfully generated compound matrix for {selected_molecule}")
            logger.info(f"Compound matrix shape: {compound_matrix.shape}")
            logger.info(f"Compound matrix sum: {compound_matrix.sum()}")
            logger.info(f"Compound matrix min/max: {np.min(compound_matrix)} / {np.max(compound_matrix)}")
            
            return selected_molecule
            
        except Exception as e:
            logger.error(f"Error generating compound matrix for molecule {selected_molecule}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return dash.no_update
