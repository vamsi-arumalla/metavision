from dash import html, dcc
from flask import session
import logging
logger = logging.getLogger("metavision")

def get_visualization_layout(cache):
    """
    Returns the layout for the Visualization sub-tab with enhanced interactive controls.
    """
    molecule_list = cache.get(f"{session['session_id']}:molecules_list")
    ref_compound = cache.get(f"{session['session_id']}:ref_compound")
    logger.info(f"ref_compound: {ref_compound}")
    
    return html.Div([
        # Header section
        html.Div([
            html.H2("Data Visualization", className="visualization-title"),
            html.P("Select your preferred visualization type and molecule to explore your data", 
                   className="visualization-subtitle")
        ], className="visualization-header"),
        # Store for last selected visualization type
        dcc.Store(id="store-last-viz-type"),
        # Store for last selected molecule
        dcc.Store(id="store-last-molecule"),
        
        # Main controls section
        html.Div([
            # Visualization Type Selection with enhanced cards
            html.Div([
                html.H3("Choose Visualization Type", className="section-title"),
                html.P("Select how you'd like to visualize your data", className="section-description"),
                
                # Interactive visualization type cards
                html.Div([
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-images", style={"fontSize": "2rem", "color": "#00bcd4"}),
                            html.H4("Slides", className="card-title"),
                            html.P("Browse through data slices as individual images", className="card-description")
                        ], className="card-content"),
                        # Make the entire card clickable
                        html.Button("", id="vis-slides", className="card-click-area", n_clicks=0)
                    ], className="vis-card", id="slides-card"),
                    
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-chart-line", style={"fontSize": "2rem", "color": "#43a047"}),
                            html.H4("Norm Plot", className="card-title"),
                            html.P("Statistical plots showing data distribution", className="card-description")
                        ], className="card-content"),
                        # Make the entire card clickable
                        html.Button("", id="vis-normplot", className="card-click-area", n_clicks=0)
                    ], className="vis-card", id="normplot-card"),
                    
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-play-circle", style={"fontSize": "2rem", "color": "#ec407a"}),
                            html.H4("Animation", className="card-title"),
                            html.P("Animated sequence through data slices", className="card-description")
                        ], className="card-content"),
                        # Make the entire card clickable
                        html.Button("", id="vis-animation", className="card-click-area", n_clicks=0)
                    ], className="vis-card", id="animation-card"),
                    
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-cube", style={"fontSize": "2rem", "color": "#ff9800"}),
                            html.H4("3D Image", className="card-title"),
                            html.P("Three-dimensional visualization of your data", className="card-description")
                        ], className="card-content"),
                        # Make the entire card clickable
                        html.Button("", id="vis-3dimage", className="card-click-area", n_clicks=0)
                    ], className="vis-card", id="3dimage-card"),
                ], className="vis-cards-grid"),
                
                # Hidden dropdown for callback compatibility
                dcc.Dropdown(
                    id='visualization-type',
                    options=[
                        {'label': 'Slides', 'value': 'slides'},
                        {'label': 'Norm Plot', 'value': 'normplot'},
                        {'label': 'Animation', 'value': 'animation'},
                        {'label': '3D Image', 'value': '3dimage'},
                    ],
                    value=None,  # No default selection
                    style={'display': 'none'}
                )
            ], className="visualization-type-section"),
            
            html.Hr(className="section-divider"),
            
            # Molecule Selection with enhanced interface
            html.Div([
                html.H3("Select Molecule", className="section-title"),
                html.P("Choose which molecule to visualize", className="section-description"),
                
                # Enhanced molecule selector
                html.Div([
                    html.Div([
                        html.Label("Primary Molecule:", className="molecule-label"),
                        html.Div([
                            dcc.Dropdown(
                                id='column-selector',
                                options=[
                                    {'label': molecule, 'value': molecule} for molecule in molecule_list
                                ],
                                value=None,  # No default selection
                                placeholder="Select primary molecule",
                                className="molecule-dropdown"
                            ),
                            html.Div([
                                html.I(className="fas fa-info-circle"),
                                html.Span("This will be the main molecule displayed in your visualization", 
                                         className="molecule-hint")
                            ], className="molecule-hint-container")
                        ], className="molecule-selector-container")
                    ], className="molecule-section"),
                    
                    # Quick selection buttons for common molecules
                    html.Div([
                        html.Label("Quick Select:", className="quick-select-label"),
                        html.Div([
                            html.Button(
                                molecule,
                                id={"type": "quick-select-btn", "index": i},
                                className="quick-select-btn",
                                n_clicks=0
                            ) for i, molecule in enumerate(molecule_list[:5] if molecule_list else [])  # Show first 5 molecules
                        ], className="quick-select-buttons")
                    ], className="quick-select-section") if molecule_list else html.Div(style={"display": "none"})
                ], className="molecule-selection-container")
            ], className="molecule-section"),
            
            # Action buttons
            html.Div([
                html.Button(
                    [html.I(className="fas fa-eye"), " Generate Visualization"],
                    id="generate-viz-btn",
                    className="generate-viz-btn"
                ),
                html.Button(
                    [html.I(className="fas fa-download"), " Export Visualization"],
                    id="export-viz-btn",
                    className="export-viz-btn"
                )
            ], className="viz-action-buttons"),
            
            # Debug info - show current selection
            html.Div([
                html.Div(id="debug-selection", style={
                    "textAlign": "center", 
                    "marginTop": "16px", 
                    "padding": "8px", 
                    "background": "rgba(0,188,212,0.1)", 
                    "borderRadius": "8px",
                    "fontSize": "0.9rem",
                    "color": "#00bcd4"
                })
            ]),
            
            # Loading and output container
            dcc.Loading(
                id="loading-visualization",
                type="circle",
                color="#00BCD4",
                children=html.Div(id='visualization-output', className="visualization-container"),
                overlay_style={"visibility": "visible", "filter": "blur(2px)", "background": "rgba(0,0,0,0.3)"}
            ),
            # Always include a hidden placeholder for vtk-2d-view so Dash registers the callback
            html.Div(id="vtk-2d-view", style={"display": "none"}),
            # Always include the 3d-visualization-container (hidden by default)
            html.Div(id="3d-visualization-container", style={"display": "none"}),
            # Always include hidden 3D controls so Dash callbacks never error
            html.Div([
                dcc.Dropdown(
                    id='thickness-value',
                    options=[{'label': str(i), 'value': i} for i in range(1, 11)],
                    value=1,
                    clearable=False,
                    className="compact-dropdown"
                ),
                dcc.Dropdown(
                    id='gap-value',
                    options=[{'label': str(i), 'value': i} for i in range(10)],
                    value=0,
                    clearable=False,
                    className="compact-dropdown"
                ),
                dcc.Dropdown(
                    id='max-projection-value',
                    options=[{'label': str(i), 'value': i} for i in range(100, 0, -1)],
                    value=99,
                    clearable=False,
                    className="compact-dropdown"
                ),
                dcc.RadioItems(
                    id='projection-type',
                    options=[
                        {'label': 'Original', 'value': 'original'},
                        {'label': 'Maximum', 'value': 'maximum'}
                    ],
                    value='original',
                    inline=True,
                    className="radio-group"
                ),
                dcc.RadioItems(
                    id='3d-visualization-type',
                    options=[
                        {'label': 'Advanced (Isosurface + Volume)', 'value': 'advanced'},
                        {'label': 'Slice-Based', 'value': 'slice-based'},
                        {'label': 'Simple Scatter', 'value': 'simple'}
                    ],
                    value='advanced',
                    inline=True,
                    className="radio-group"
                ),
                dcc.Interval(id='force-3d-update', interval=500, n_intervals=0, max_intervals=1)
            ], style={"display": "none"})
        ], className="visualization-controls")
    ], className="visualization-layout")