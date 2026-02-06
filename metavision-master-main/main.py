# Fix matplotlib backend before any imports that might use it
import matplotlib
matplotlib.use('Agg')

from dash import Dash, html, dcc, State, Input, Output, ClientsideFunction
from form.form_layout import get_layout
import logging
import dash
import uuid
import os
import secrets
from flask import session, g, request, send_from_directory
from flask_caching import Cache
from utils import janitor
import threading
from form.form_callback import register_form_callback
from visualization.visualization_layout import get_visualization_layout
from visualization.slides_callback import register_slides_callback
from visualization.threed_callback import register_3d_callback
from visualization.animation_callback import register_animation_callback
from export.export_layout import get_export_layout
from export.export_callback import register_export_callback
from config import setup_logger
from visualization.slides_layout import register_slides_colormap_callbacks
from visualization.threed_page import get_3d_visualization_page
from visualization.threed_page_callback import register_3d_page_callbacks

# Initialize the root logger once
setup_logger(log_level=logging.INFO, log_dir="logs", name="metavision")
# Get a reference to that logger
logger = logging.getLogger("metavision")
from visualization.visualization_callback import register_visualization_callback

app = Dash(
    external_stylesheets=[
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css",
        "/assets/upload_styles.css?v=2"
    ],
    external_scripts=[
        "https://cdn.jsdelivr.net/npm/resumablejs@1.1.0/resumable.min.js",
        "/assets/chunked_upload.js?v=2"
    ]
)
server = app.server
server.secret_key = secrets.token_hex(16)

# Allow uploads up to 2GB
server.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB

app.config.suppress_callback_exceptions = True
app.title = "MetaVision Data Processing"
threading.Thread(target=janitor, daemon=True).start()

CACHE_CONFIG = {
    "CACHE_TYPE": "simple",
    "CACHE_DEFAULT_TIMEOUT": 60 * 60,
}

cache = Cache(app.server, config=CACHE_CONFIG)

app.layout = html.Div([
    dcc.Location(id="url"),
    dcc.Store(id="selected-page", data="landing"),
    dcc.Store(id="processed", data=False),
    dcc.Store(id="molecule"),
    dcc.Download(id="download-data"),
    dcc.Download(id="download-3d-image"),
    html.Div(id="root-content")
])

@server.before_request
def ensure_user_workspace():
    """Ensure that the user workspace exists."""
    if "session_id" not in session:
        session["session_id"] = uuid.uuid4().hex
    os.mkdir("workspaces") if not os.path.exists("workspaces") else None

@app.callback(
    Output("root-content", "children"),
    [Input("selected-page", "data"), State("processed", "data")],
)
def render_page(page, processed):
    # Determine which button should be visible
    show_landing = page == "landing"
    show_post = page == "postprocessing"
    show_vis = page == "visualization"
    show_exp = page == "export"
    show_dashboard = page == "dashboard"

    # Always render all navigation buttons, but only one is visible at a time
    buttons = html.Div([
        html.Button("Let's Go!", id="landing-go-btn", n_clicks=0, className="lets-go-btn hidden"),
        html.Button("Visualization", id="go-visualization-btn", n_clicks=0, className="lets-go-btn" if show_post else "lets-go-btn hidden", style={"marginRight": "32px"}),
        html.Button("Export", id="go-export-btn", n_clicks=0, className="lets-go-btn" if show_post else "lets-go-btn hidden"),
        html.Button("Back", id="back-to-dashboard-btn", n_clicks=0, className="lets-go-btn" if (show_vis or show_exp) else "lets-go-btn hidden", style={"margin": "32px 0 24px 32px"}),
    ], style={"display": "flex", "justifyContent": "center", "marginTop": "48px"})

    landing_style = {
        "height": "100vh",
        "width": "100vw",
        "display": "flex",
        "flexDirection": "row",
        "justifyContent": "center",
        "alignItems": "center",
        "position": "relative",
        "overflow": "hidden",
        "backgroundImage": "url('https://wallpapercave.com/wp/wp6947571.jpg'), linear-gradient(135deg, #121212cc 60%, #00bcd4cc 100%)",
        "backgroundBlendMode": "darken, normal",
        "backgroundSize": "cover",
        "backgroundPosition": "center",
        "backgroundRepeat": "no-repeat",
    }
    sidebar_animation_style = {
        "width": "340px",
        "height": "70vh",
        "background": "rgba(30,30,30,0.98)",
        "borderRadius": "24px",
        "boxShadow": "0 8px 32px rgba(0,0,0,0.25)",
        "display": "flex",
        "flexDirection": "column",
        "alignItems": "center",
        "justifyContent": "center",
        "marginLeft": "5vw",
        "animation": "sidebarFadeIn 1.2s cubic-bezier(.4,2,.6,1)",
        "zIndex": 10
    }
    if page == "landing":
        landing_content = html.Div([
            html.Div(style={
                "position": "absolute",
                "top": 0,
                "left": 0,
                "width": "100%",
                "height": "100%",
                "background": "rgba(18,18,18,0.82)",
                "zIndex": 1
            }),
            html.Div([
                html.H1("MetaVision", style={
                    "fontSize": "2.8rem",
                    "color": "#00bcd4",
                    "fontWeight": 700,
                    "textAlign": "center",
                    "margin": "0 0 12px 0",
                    "letterSpacing": "1.5px",
                    "zIndex": 2
                }),
                html.P("Biomolecule Data Processing Dashboard", style={
                    "textAlign": "center",
                    "color": "#b0bec5",
                    "fontSize": "1.15rem",
                    "margin": "0 0 32px 0",
                    "fontWeight": 400,
                    "zIndex": 2
                }),
                html.Button("Let's Go!", id="landing-go-btn", n_clicks=0, className="lets-go-btn", style={"marginTop": "32px"})
            ], style=sidebar_animation_style)
        ], style=landing_style)
        # Include all navigation buttons for callback compatibility, but only show the landing button
        return html.Div([
            landing_content,
            html.Div([
                html.Button("Visualization", id="go-visualization-btn", n_clicks=0, className="lets-go-btn hidden"),
                html.Button("Export", id="go-export-btn", n_clicks=0, className="lets-go-btn hidden"),
                html.Button("Back", id="back-to-dashboard-btn", n_clicks=0, className="lets-go-btn hidden"),
            ], style={"display": "none"})
        ])
    elif page == "3dvisualization":
        # Use the dedicated 3D visualization page
        return get_3d_visualization_page(cache)
    elif page == "dashboard":
        # Show upload/pre-processing form with enhanced pipeline explanation
        return html.Div([
            # Header with title and description
            html.Div([
                html.H1("MetaVision", className="app-title", style={"marginTop": "32px", "textAlign": "center", "color": "#00bcd4"}),
                html.P("Advanced Biomolecule Data Processing & 3D Visualization Platform", 
                       style={"textAlign": "center", "color": "#b0bec5", "fontSize": "1.2rem", "marginBottom": "48px"})
            ]),
            
            # Pipeline Overview Section
            html.Div([
                html.H2("Data Processing Pipeline", style={
                    "color": "#00bcd4", 
                    "textAlign": "center", 
                    "marginBottom": "32px",
                    "fontSize": "2rem"
                }),
                
                # Pipeline Flow Diagram
                html.Div([
                    # Step 1: Data Upload
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-upload", style={"fontSize": "2rem", "color": "#00bcd4"}),
                            html.H4("1. Data Upload", style={"color": "#fff", "margin": "16px 0 8px 0"}),
                            html.P("Upload CSV files containing MALDI-MS data with spatial coordinates", 
                                   style={"color": "#b0bec5", "fontSize": "0.9rem", "textAlign": "center"})
                        ], style={"textAlign": "center", "padding": "24px"})
                    ], style={
                        "background": "linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%)",
                        "borderRadius": "16px",
                        "border": "2px solid #00bcd4",
                        "flex": "1",
                        "margin": "0 8px"
                    }),
                    
                    # Arrow
                    html.Div([
                        html.I(className="fas fa-arrow-right", style={"fontSize": "1.5rem", "color": "#00bcd4"})
                    ], style={"display": "flex", "alignItems": "center", "margin": "0 16px"}),
                    
                    # Step 2: Normalization
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-balance-scale", style={"fontSize": "2rem", "color": "#00bcd4"}),
                            html.H4("2. Normalization", style={"color": "#fff", "margin": "16px 0 8px 0"}),
                            html.P("Total sum and section normalization for data standardization", 
                                   style={"color": "#b0bec5", "fontSize": "0.9rem", "textAlign": "center"})
                        ], style={"textAlign": "center", "padding": "24px"})
                    ], style={
                        "background": "linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%)",
                        "borderRadius": "16px",
                        "border": "2px solid #00bcd4",
                        "flex": "1",
                        "margin": "0 8px"
                    }),
                    
                    # Arrow
                    html.Div([
                        html.I(className="fas fa-arrow-right", style={"fontSize": "1.5rem", "color": "#00bcd4"})
                    ], style={"display": "flex", "alignItems": "center", "margin": "0 16px"}),
                    
                    # Step 3: 3D Alignment
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-cube", style={"fontSize": "2rem", "color": "#00bcd4"}),
                            html.H4("3. 3D Alignment", style={"color": "#fff", "margin": "16px 0 8px 0"}),
                            html.P("Motion correction and spatial alignment of tissue sections", 
                                   style={"color": "#b0bec5", "fontSize": "0.9rem", "textAlign": "center"})
                        ], style={"textAlign": "center", "padding": "24px"})
                    ], style={
                        "background": "linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%)",
                        "borderRadius": "16px",
                        "border": "2px solid #00bcd4",
                        "flex": "1",
                        "margin": "0 8px"
                    }),
                    
                    # Arrow
                    html.Div([
                        html.I(className="fas fa-arrow-right", style={"fontSize": "1.5rem", "color": "#00bcd4"})
                    ], style={"display": "flex", "alignItems": "center", "margin": "0 16px"}),
                    
                    # Step 4: Processing
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-cogs", style={"fontSize": "2rem", "color": "#00bcd4"}),
                            html.H4("4. Processing", style={"color": "#fff", "margin": "16px 0 8px 0"}),
                            html.P("Interpolation and imputation for data enhancement", 
                                   style={"color": "#b0bec5", "fontSize": "0.9rem", "textAlign": "center"})
                        ], style={"textAlign": "center", "padding": "24px"})
                    ], style={
                        "background": "linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%)",
                        "borderRadius": "16px",
                        "border": "2px solid #00bcd4",
                        "flex": "1",
                        "margin": "0 8px"
                    }),
                    
                    # Arrow
                    html.Div([
                        html.I(className="fas fa-arrow-right", style={"fontSize": "1.5rem", "color": "#00bcd4"})
                    ], style={"display": "flex", "alignItems": "center", "margin": "0 16px"}),
                    
                    # Step 5: Visualization
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-chart-3d", style={"fontSize": "2rem", "color": "#00bcd4"}),
                            html.H4("5. Visualization", style={"color": "#fff", "margin": "16px 0 8px 0"}),
                            html.P("3D visualization and analysis of processed data", 
                                   style={"color": "#b0bec5", "fontSize": "0.9rem", "textAlign": "center"})
                        ], style={"textAlign": "center", "padding": "24px"})
                    ], style={
                        "background": "linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%)",
                        "borderRadius": "16px",
                        "border": "2px solid #00bcd4",
                        "flex": "1",
                        "margin": "0 8px"
                    })
                ], style={
                    "display": "flex",
                    "justifyContent": "center",
                    "alignItems": "center",
                    "marginBottom": "48px",
                    "flexWrap": "wrap",
                    "gap": "16px"
                }),
                
                # Pipeline Details
                html.Div([
                    html.H3("Pipeline Details", style={"color": "#00bcd4", "marginBottom": "24px", "textAlign": "center"}),
                    
                    # Two-column layout for details
                    html.Div([
                        # Left column - Technical Details
                        html.Div([
                            html.H4("Technical Implementation", style={"color": "#fff", "marginBottom": "16px"}),
                            html.Ul([
                                html.Li("MetaNorm3D: Total sum and section normalization", style={"color": "#b0bec5", "marginBottom": "8px"}),
                                html.Li("MetaAlign3D: Motion correction using ECC algorithm", style={"color": "#b0bec5", "marginBottom": "8px"}),
                                html.Li("MetaInterp3D: Spatial interpolation for missing data", style={"color": "#b0bec5", "marginBottom": "8px"}),
                                html.Li("MetaImpute3D: Neighbor-based imputation", style={"color": "#b0bec5", "marginBottom": "8px"}),
                                html.Li("MetaAtlas3D: NIfTI format export for external tools", style={"color": "#b0bec5", "marginBottom": "8px"})
                            ], style={"listStyle": "none", "padding": "0"})
                        ], style={"flex": "1", "padding": "24px", "background": "rgba(0,188,212,0.1)", "borderRadius": "12px", "marginRight": "16px"}),
                        
                        # Right column - Features
                        html.Div([
                            html.H4("Key Features", style={"color": "#fff", "marginBottom": "16px"}),
                            html.Ul([
                                html.Li("Large file support with chunked upload", style={"color": "#b0bec5", "marginBottom": "8px"}),
                                html.Li("Real-time 3D visualization", style={"color": "#b0bec5", "marginBottom": "8px"}),
                                html.Li("Interactive data exploration", style={"color": "#b0bec5", "marginBottom": "8px"}),
                                html.Li("Multiple export formats", style={"color": "#b0bec5", "marginBottom": "8px"}),
                                html.Li("Session-based data management", style={"color": "#b0bec5", "marginBottom": "8px"})
                            ], style={"listStyle": "none", "padding": "0"})
                        ], style={"flex": "1", "padding": "24px", "background": "rgba(0,188,212,0.1)", "borderRadius": "12px", "marginLeft": "16px"})
                    ], style={"display": "flex", "marginBottom": "32px"})
                ], style={"background": "var(--dark-surface-2)", "borderRadius": "16px", "padding": "32px", "border": "1px solid #333"})
            ], style={"marginBottom": "48px"}),
            
            # Upload Section
            html.Div([
                html.H3("Start Processing", style={"color": "#00bcd4", "textAlign": "center", "marginBottom": "32px"}),
                get_layout()
            ], style={"background": "var(--dark-surface-2)", "borderRadius": "16px", "padding": "32px", "border": "1px solid #333"}),
            
            buttons
        ], style={"minHeight": "100vh", "background": "var(--dark-surface)", "paddingBottom": "32px", "overflow": "auto"})
    elif page == "postprocessing":
        # Show sidebar layout with Visualization and Export buttons
        return html.Div([
            html.H1("MetaVision", className="app-title", style={"marginTop": "32px", "textAlign": "center"}),
            
            # Main container with sidebar and content
            html.Div([
                # Sidebar with navigation buttons
                html.Div([
                    html.H3("Post-Processing", style={
                        "color": "#00bcd4",
                        "marginBottom": "24px",
                        "textAlign": "center",
                        "fontSize": "1.4rem"
                    }),
                    html.Button("Visualization", id="go-visualization-btn", n_clicks=0, className="sidebar-btn", style={
                        "width": "100%",
                        "marginBottom": "16px",
                        "padding": "16px 24px",
                        "fontSize": "1.1rem",
                        "fontWeight": "600"
                    }),
                    html.Button("Export", id="go-export-btn", n_clicks=0, className="sidebar-btn", style={
                        "width": "100%",
                        "padding": "16px 24px",
                        "fontSize": "1.1rem",
                        "fontWeight": "600"
                    })
                ], style={
                    "width": "280px",
                    "background": "var(--dark-surface-2)",
                    "padding": "32px 24px",
                    "borderRadius": "16px",
                    "border": "1px solid #333",
                    "height": "fit-content",
                    "position": "sticky",
                    "top": "32px"
                }),
                
                # Main content area
                html.Div(id="postprocessing-content", style={
                    "flex": "1",
                    "marginLeft": "32px",
                    "minHeight": "600px",
                    "background": "var(--dark-surface-2)",
                    "borderRadius": "16px",
                    "border": "1px solid #333",
                    "padding": "32px",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center"
                }, children=[
                    html.Div([
                        html.I(className="fas fa-chart-line", style={
                            "fontSize": "4rem",
                            "color": "#00bcd4",
                            "marginBottom": "24px"
                        }),
                        html.H3("Select an Option", style={
                            "color": "#b0bec5",
                            "marginBottom": "16px",
                            "fontSize": "1.5rem"
                        }),
                        html.P("Choose Visualization or Export from the sidebar to get started", style={
                            "color": "#888",
                            "textAlign": "center",
                            "fontSize": "1.1rem"
                        })
                    ], style={"textAlign": "center"})
                ])
            ], style={
                "display": "flex",
                "marginTop": "32px",
                "gap": "32px",
                "alignItems": "flex-start"
            }),
            
            # Back button at the bottom
            html.Div([
                html.Button("Back to Dashboard", id="back-to-dashboard-btn", n_clicks=0, className="back-btn", style={
                    "marginTop": "48px",
                    "padding": "12px 32px",
                    "fontSize": "1.1rem"
                })
            ], style={"textAlign": "center", "marginTop": "32px"}),
            
            # Include hidden buttons for callback compatibility
            html.Div([
                html.Button("Let's Go!", id="landing-go-btn", n_clicks=0, className="lets-go-btn hidden"),
            ], style={"display": "none"})
        ], style={"minHeight": "100vh", "background": "var(--dark-surface)", "paddingBottom": "32px", "overflow": "auto"})
    elif page == "visualization":
        return html.Div([
            buttons,
            get_visualization_layout(cache)
        ], style={"minHeight": "100vh", "background": "var(--dark-surface)", "overflow": "auto"})
    elif page == "export":
        return html.Div([
            buttons,
            get_export_layout(cache)
        ], style={"minHeight": "100vh", "background": "var(--dark-surface)", "overflow": "auto"})
    else:
        return html.Div([
            html.H3("Error", className="error-text"),
            html.P("Invalid page selected.", className="error-message"),
            buttons
        ], style={"overflow": "auto"})

@app.callback(
    Output("page-content-inner", "children"),
    [Input("selected-page", "data"), State("processed", "data")],
    prevent_initial_call=True
)
def render_main_content(page, processed):
    if page == "dashboard":
        return get_layout()
    return None

# Registering the callback for the tabs
register_form_callback(app, cache)
# Registering the callback for visualization
register_visualization_callback(app, cache)
register_export_callback(app, cache)
register_slides_callback(app, cache)
register_3d_callback(app, cache)
register_animation_callback(app, cache)
register_slides_colormap_callbacks(app, cache)
register_3d_page_callbacks(app, cache)

app.clientside_callback(
    """
    function(page) {
        if (page === 'landing') {
            document.body.style.overflow = 'hidden';
        } else {
            document.body.style.overflow = 'auto';
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output('root-content', 'children', allow_duplicate=True),
    Input('selected-page', 'data'),
    prevent_initial_call=True
)

# Add chunked upload endpoint for Resumable.js
# Serve chunked upload JavaScript
@app.server.route('/assets/<path:filename>')
def serve_assets(filename):
    return send_from_directory('assets', filename)

assemble_locks = {}
@app.server.route('/chunk-upload', methods=['POST'])
def chunk_upload():
    print("[DEBUG] request.files:", request.files)
    print("[DEBUG] request.form:", request.form)
    print("[DEBUG] request.data:", request.data)
    print("[DEBUG] request.files.keys():", list(request.files.keys()))
    resumableIdentifier = request.form.get('resumableIdentifier')
    resumableFilename = request.form.get('resumableFilename')
    resumableChunkNumber = request.form.get('resumableChunkNumber')
    total_chunks = request.form.get('resumableTotalChunks')

    if not request.files:
        print("[ERROR] No files in request.files!")
        return "No file uploaded", 400

    # Try to get the first file, regardless of field name
    file_field = next(iter(request.files), None)
    if not file_field:
        print("[ERROR] No file field found in request.files!")
        return "No file uploaded", 400

    chunk_data = request.files[file_field]
    print(f"[DEBUG] Using file field: {file_field}, filename: {chunk_data.filename}")

    upload_dir = 'workspaces'
    os.makedirs(upload_dir, exist_ok=True)
    chunk_name = f"{resumableFilename}.part{resumableChunkNumber}"
    chunk_path = os.path.join(upload_dir, chunk_name)

    try:
        chunk_data.save(chunk_path)
        print(f"[INFO] Saved chunk {chunk_name} to {chunk_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save chunk: {e}")
        return f"Failed to save chunk: {e}", 500

    try:
        total_chunks = int(total_chunks)
    except Exception as e:
        print(f"[ERROR] Invalid total_chunks: {e}")
        return "Invalid total_chunks", 400

    # Only one thread should try to assemble at a time per file
    lock = assemble_locks.setdefault(resumableFilename, threading.Lock())
    with lock:
        all_chunks = [
            os.path.join(upload_dir, f"{resumableFilename}.part{i}")
            for i in range(1, total_chunks + 1)
        ]
        if all(os.path.exists(p) for p in all_chunks):
            assembled_path = os.path.join(upload_dir, resumableFilename)
            try:
                with open(assembled_path, 'wb') as f:
                    for part_path in all_chunks:
                        with open(part_path, 'rb') as part_file:
                            f.write(part_file.read())
                        os.remove(part_path)
                print(f"[INFO] All chunks uploaded and file assembled at {assembled_path}")
                return 'All chunks uploaded and file assembled', 200
            except Exception as e:
                print(f"[ERROR] Failed to assemble file: {e}")
                return f"Failed to assemble file: {e}", 500

    return 'Chunk upload successful', 200

# Add clientside callback for chunked upload initialization
app.clientside_callback(
    '''
    function(upload_area_style) {
        // Initialize chunked upload when large file upload area is shown
        if (upload_area_style && upload_area_style.display !== 'none') {
            console.log('Large file upload area is visible, initializing chunked upload...');
            setTimeout(function() {
                if (window.setupChunkedUpload) {
                    window.setupChunkedUpload();
                }
            }, 500);
        }
        return window.dash_clientside.no_update;
    }
    ''',
    Output('large-file-upload-area', 'title'),  # Dummy output
    Input('large-file-upload-area', 'style'),
    prevent_initial_call=True
)

# Add clientside callback for mouse wheel slice navigation on 2D image in 3D visualization
app.clientside_callback(
    '''
    function(max_slice) {
        // Attach mouse wheel listener to 2D image
        if (!window._threed2dWheelAttached) {
            setTimeout(function() {
                var graph = document.getElementById('threed-2d-image-graph');
                if (graph) {
                    graph.addEventListener('wheel', function(e) {
                        e.preventDefault();
                        var slider = document.getElementById('threed-2d-slice-slider');
                        if (slider && slider.value !== undefined) {
                            var current = parseInt(slider.value) || 0;
                            var delta = e.deltaY > 0 ? 1 : -1;
                            var next = Math.max(0, Math.min(current + delta, max_slice));
                            if (next !== current) {
                                slider.value = next;
                                slider.dispatchEvent(new Event('input', { bubbles: true }));
                            }
                        }
                    }, { passive: false });
                    window._threed2dWheelAttached = true;
                }
            }, 1000);
        }
        return window.dash_clientside.no_update;
    }
    ''',
    Output('threed-2d-slice-slider', 'id'),  # Dummy output
    Input('threed-2d-slice-slider', 'max'),
    prevent_initial_call=True
)

if __name__ == "__main__":
    app.run(debug=True)