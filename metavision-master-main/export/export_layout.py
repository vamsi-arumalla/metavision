from dash import html, dcc
from flask import session
import logging
logger = logging.getLogger("metavision")

def get_export_layout(cache):
    """
    Returns the enhanced layout for the Export sub-tab with better interactivity.
    """
    logger.debug("Creating Enhanced Export Layout")
    molecule_list = cache.get(f"{session['session_id']}:molecules_list")
    if not molecule_list:
        logger.error("No molecules found in cache.")
        raise ValueError("No molecules found in cache.")
    logger.debug(f"molecule_list: {molecule_list}")
    tissue_list = cache.get(f"{session['session_id']}:tissue_ids")
    if not tissue_list:
        logger.error("No tissues found in cache.")
        raise ValueError("No tissues found in cache.")
    
    return html.Div([
        # Header Section
        html.Div([
            html.H1("DATA EXPORT", 
                   style={
                       "color": "#00BCD4", 
                       "textAlign": "center", 
                       "fontSize": "36px",
                       "fontWeight": "bold",
                       "marginBottom": "10px",
                       "textShadow": "2px 2px 4px rgba(0,0,0,0.5)",
                       "letterSpacing": "2px"
                   }),
            html.P("Export your processed data in various formats", 
                  style={
                      "color": "#B0BEC5", 
                      "textAlign": "center", 
                      "fontSize": "16px",
                      "marginBottom": "30px"
                  })
        ], style={
            "padding": "30px",
            "backgroundColor": "rgba(0,188,212,0.1)",
            "borderRadius": "15px",
            "border": "2px solid #00BCD4",
            "marginBottom": "30px"
        }),
        
        # Normalization Selection with enhanced styling
        html.Div([
            html.Div([
                html.Label("Select Normalization:", className="export-label-enhanced"),
                html.Div([
                    dcc.RadioItems(
                        id='normalization-select',
                        options=[
                            {'label': 'None', 'value': 'none'},
                            {'label': 'Total Sum Normalization', 'value': 'total_sum_norm'}
                        ],
                        value='none',
                        inline=True,
                        inputStyle={"marginRight": "8px"},
                        labelStyle={"marginRight": "30px", "color": "#E0E0E0"},
                        className="normalization-radio-enhanced"
                    )
                ], className="radio-container-enhanced")
            ], className="normalization-controls-enhanced")
        ], className="form-section-enhanced normalization-section-enhanced"),
        
        html.Hr(style={"border": "1px solid #444", "margin": "30px 0"}),
        
        # Enhanced Two Columns for Selection
        html.Div([
            # First Column (Molecules)
            html.Div([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-atom", style={"color": "#00BCD4", "marginRight": "10px"}),
                        html.Span("MOLECULES", style={"color": "#00BCD4", "fontWeight": "bold", "fontSize": "18px"})
                    ], style={"display": "flex", "alignItems": "center"}),
                    html.Div([
                        html.Span(f"Total: {len(molecule_list)}", className="count-label-enhanced"),
                        html.Span("Selected: ", className="count-label-enhanced"),
                        html.Span("0", id="molecules-selected-count", className="count-value-enhanced"),
                        html.Span(" ", className="spacer"),
                        dcc.Checklist(
                            id='select-all-molecules',
                            options=[{'label': 'Select All', 'value': 'select-all'}],
                            value=[],
                            className="select-all-checkbox-enhanced"
                        )
                    ], style={"marginTop": "10px"})
                ], className="column-header-enhanced"),
                
                # Enhanced scrollable list of molecules
                html.Div([
                    dcc.Checklist(
                        id='molecule-checklist',
                        options=[
                            {'label': i, 'value': i} 
                            for i in molecule_list
                        ],
                        value=[],
                        className="item-checklist-enhanced"
                    )
                ], className="column-content-enhanced checklist-container-enhanced")
            ], className="selection-column-enhanced"),
            
            # Second Column Tissues
            html.Div([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-layer-group", style={"color": "#00BCD4", "marginRight": "10px"}),
                        html.Span("TISSUES", style={"color": "#00BCD4", "fontWeight": "bold", "fontSize": "18px"})
                    ], style={"display": "flex", "alignItems": "center"}),
                    html.Div([
                        html.Span(f"Total: {len(tissue_list)}", className="count-label-enhanced"),
                        html.Span("Selected: ", className="count-label-enhanced"),
                        html.Span("0", id="tissues-selected-count", className="count-value-enhanced"),
                        html.Span(" ", className="spacer"),
                        dcc.Checklist(
                            id='select-all-tissues',
                            options=[{'label': 'Select All', 'value': 'select-all'}],
                            value=[],
                            className="select-all-checkbox-enhanced"
                        )
                    ], style={"marginTop": "10px"})
                ], className="column-header-enhanced"),
                
                # Enhanced scrollable list of tissues
                html.Div([
                    dcc.Checklist(
                        id='tissues-checklist',
                        options=[
                            {'label': i, 'value': i} 
                            for i in tissue_list
                        ],
                        value=[],
                        className="item-checklist-enhanced"
                    )
                ], className="column-content-enhanced checklist-container-enhanced")
            ], className="selection-column-enhanced"),
        ], className="selection-columns-container-enhanced"),
        
        html.Hr(style={"border": "1px solid #444", "margin": "30px 0"}),
        
        # Enhanced Export Options with BIG interactive elements
        html.Div([
            # BIG Export Format Selection
            html.Div([
                html.Label("Export Format:", className="export-label-big"),
                dcc.RadioItems(
                    id='export-format',
                    options=[
                        {'label': 'Single CSV', 'value': 'single'},
                        {'label': 'Multiple CSVs (ZIP)', 'value': 'multiple'}
                    ],
                    value='single',
                    inline=True,
                    inputStyle={"marginRight": "18px", "transform": "scale(1.7)"},
                    labelStyle={"marginRight": "40px", "color": "#2196F3", "fontSize": "1.5rem", "fontWeight": "700", "padding": "18px 32px", "borderRadius": "12px", "background": "rgba(33,150,243,0.08)", "transition": "all 0.3s"},
                    className="export-format-radio-big"
                ),
            ], className="export-format-section-big"),
            
            # BIG Directory Selection and Export buttons
            html.Div([
                html.Label("Output Directory:", className="export-label-big"),
                dcc.Input(
                    id='output-directory',
                    type='text',
                    placeholder="Select output directory...",
                    className="directory-input-big"
                ),
                html.Button(
                    [
                        html.I(className="fas fa-folder-open", style={"marginRight": "14px", "fontSize": "1.5rem"}),
                        "BROWSE"
                    ],
                    id="browse-button",
                    className="browse-button-big"
                ),
                html.Button(
                    [
                        html.I(className="fas fa-download", style={"marginRight": "14px", "fontSize": "1.5rem"}),
                        "EXPORT DATA"
                    ],
                    id="export-button",
                    className="export-button-big"
                ),
            ], className="directory-controls-big", style={"marginTop": "32px", "marginBottom": "32px"}),
            
            # Enhanced Status message area
            html.Div(id="export-status", className="export-status-enhanced")
        ], className="form-section-enhanced export-options-enhanced"),
        
        # Export-specific download component and interval for button reset
        dcc.Download(id="export-download-data"),
        dcc.Interval(id="export-reset-interval", interval=1200, n_intervals=0, disabled=True),
        
    ], className="export-layout-enhanced")