from dash import Dash, html, dcc

def get_layout():
    """
    Function to create the layout of the Dash app.
    """
    return html.Div([
        dcc.Store(id="chunked-upload-processed", data=False),
        
        # File Upload Section (Unified)
        html.Div([
            html.H3([
                html.I(className="fas fa-upload", style={"marginRight": "12px", "color": "#00bcd4"}),
                "Upload Data (CSV)"
            ], style={
                "color": "#fff",
                "fontSize": "1.5rem",
                "marginBottom": "24px",
                "textAlign": "center"
            }),
            
            # Upload Method Selection
            html.Div([
                html.Div([
                    html.Button([
                        html.I(className="fas fa-file", style={"marginRight": "8px"}),
                        "Standard Upload"
                    ], id="standard-upload-btn", className="upload-method-btn", style={
                        "background": "#00bcd4",
                        "color": "#fff",
                        "border": "none",
                        "padding": "12px 24px",
                        "borderRadius": "8px 0 0 8px",
                        "fontSize": "1rem",
                        "cursor": "pointer",
                        "flex": "1"
                    }),
                    html.Button([
                        html.I(className="fas fa-cloud-upload-alt", style={"marginRight": "8px"}),
                        "Large File Upload"
                    ], id="chunked-upload-btn", className="upload-method-btn", style={
                        "background": "#37474f",
                        "color": "#fff",
                        "border": "none",
                        "padding": "12px 24px",
                        "borderRadius": "0 8px 8px 0",
                        "fontSize": "1rem",
                        "cursor": "pointer",
                        "flex": "1"
                    })
                ], style={
                    "display": "flex",
                    "marginBottom": "24px",
                    "maxWidth": "400px",
                    "margin": "0 auto 24px auto"
                }),
                
                # Info text
                html.Div([
                    html.I(className="fas fa-info-circle", style={"color": "#2196f3", "marginRight": "8px"}),
                    html.Span("Use Standard Upload for files < 100MB, Large File Upload for files > 100MB (up to 2GB)", 
                             style={"color": "#b0bec5", "fontSize": "0.9rem"})
                ], style={
                    "textAlign": "center",
                    "marginBottom": "16px"
                })
            ]),
            # Standard Upload Area
            html.Div(id="standard-upload-area", style={"display": "block"}, children=[
                dcc.Upload(
                    id='unified-upload-data',
                    children=html.Div([
                        html.Div([
                            html.I(className="fas fa-cloud-upload-alt", style={
                                "fontSize": "3rem",
                                "color": "#00bcd4",
                                "marginBottom": "16px"
                            }),
                            html.H4("Drag and Drop CSV File", style={
                                "color": "#fff",
                                "marginBottom": "8px"
                            }),
                            html.P("or click to browse (files < 100MB)", style={
                                "color": "#b0bec5",
                                "marginBottom": "16px"
                            }),
                            html.Button('Select File', style={
                                "background": "#00bcd4",
                                "color": "#fff",
                                "border": "none",
                                "padding": "12px 24px",
                                "borderRadius": "8px",
                                "fontSize": "1rem",
                                "cursor": "pointer",
                                "transition": "all 0.3s ease"
                            })
                        ], style={"textAlign": "center"})
                    ]),
                    multiple=False,
                    accept='.csv',
                    style={
                        'border': '2px dashed #00bcd4',
                        'borderRadius': '16px',
                        'padding': '40px 20px',
                        'textAlign': 'center',
                        'background': 'rgba(0,188,212,0.05)',
                        'transition': 'all 0.3s ease',
                        'cursor': 'pointer'
                    }
                ),
                html.Button([
                    html.I(className="fas fa-times", style={"marginRight": "8px"}),
                    "Remove File"
                ], id="remove-standard-file-btn", style={
                    "background": "#f44336",
                    "color": "#fff",
                    "border": "none",
                    "padding": "8px 16px",
                    "borderRadius": "6px",
                    "fontSize": "0.9rem",
                    "cursor": "pointer",
                    "marginTop": "12px",
                    "display": "none"
                })
            ]),
            
            # Large File Upload Area
            html.Div(id="large-file-upload-area", style={"display": "none"}, children=[
                html.Div([
                    html.I(className="fas fa-database", style={
                        "fontSize": "3rem",
                        "color": "#37474f",
                        "marginBottom": "16px"
                    }),
                    html.H4("Large File Upload (up to 2GB)", style={
                        "color": "#fff",
                        "marginBottom": "8px"
                    }),
                    html.P("Click anywhere in this area to select your large CSV file", style={
                        "color": "#b0bec5",
                        "marginBottom": "16px"
                    }),
                ], style={
                    "textAlign": "center",
                    "border": "2px dashed #37474f",
                    "borderRadius": "16px",
                    "padding": "40px 20px",
                    "background": "rgba(55,71,79,0.05)",
                    "cursor": "pointer",
                    "transition": "all 0.3s ease"
                }, id="large-file-dropzone"),
                
                # File information display
                html.Div(id="large-file-display", style={
                    "display": "none", 
                    "marginTop": "16px",
                    "padding": "12px",
                    "background": "rgba(76, 175, 80, 0.1)",
                    "border": "1px solid rgba(76, 175, 80, 0.3)",
                    "borderRadius": "8px",
                    "textAlign": "center",
                    "color": "#4caf50",
                    "fontWeight": "500"
                }),
                
                # Chunked upload status and progress
                html.Div(id="chunked-upload-status", style={
                    "marginTop": "16px", 
                    "textAlign": "center",
                    "fontSize": "1.1rem",
                    "fontWeight": "500"
                }),
                html.Div(id="chunked-upload-progress", style={
                    "marginTop": "8px", 
                    "textAlign": "center",
                    "fontSize": "1rem",
                    "color": "#00bcd4"
                }),
                html.Div([
                    html.Div(id="large-file-progress-inner", style={
                        "width": "0%",
                        "height": "100%",
                        "background": "linear-gradient(90deg, #00bcd4, #00acc1)",
                        "borderRadius": "8px",
                        "transition": "width 0.5s ease",
                        "boxShadow": "0 2px 8px rgba(0, 188, 212, 0.4)"
                    }),
                    html.Span("0%", id="large-file-progress-text", style={
                        "position": "absolute",
                        "left": "50%",
                        "top": "50%",
                        "transform": "translate(-50%, -50%)",
                        "color": "#fff",
                        "fontWeight": "700",
                        "fontSize": "0.9rem",
                        "textShadow": "0 1px 2px rgba(0,0,0,0.5)"
                    })
                ], id="large-file-progress-bar", style={
                    "position": "relative",
                    "width": "100%",
                    "height": "28px",
                    "background": "rgba(0,0,0,0.1)",
                    "borderRadius": "14px",
                    "marginTop": "16px",
                    "display": "none",
                    "border": "1px solid rgba(0,0,0,0.1)"
                }),
                
                # Control buttons for large files
                html.Div([
                    html.Button([
                        html.I(className="fas fa-test-tube", style={"marginRight": "8px"}),
                        "Test Progress"
                    ], id="test-progress-btn", style={
                        "background": "#00bcd4",
                        "color": "#fff",
                        "border": "none",
                        "padding": "8px 16px",
                        "borderRadius": "6px",
                        "fontSize": "0.8rem",
                        "cursor": "pointer",
                        "marginRight": "8px"
                    }),
                    html.Button([
                        html.I(className="fas fa-trash", style={"marginRight": "8px"}),
                        "Remove File"
                    ], id="remove-large-file-btn", style={
                        "background": "#f44336",
                        "color": "#fff",
                        "border": "none",
                        "padding": "10px 20px",
                        "borderRadius": "6px",
                        "fontSize": "0.9rem",
                        "cursor": "pointer"
                    })
                ], style={"textAlign": "center", "marginTop": "16px"})
            ]),
            
            # Processing status for both upload methods
            html.Div(id="processing-status", style={"marginTop": "16px", "textAlign": "center"}),
            # Progress Bar
            html.Div([
                html.Div([
                    html.Div(id='unified-upload-progress-inner', style={
                        "width": "0%",
                        "height": "100%",
                        "background": "linear-gradient(90deg, #00bcd4, #00acc1)",
                        "borderRadius": "8px",
                        "transition": "width 0.3s ease"
                    }),
                    html.Span("0%", id="unified-upload-progress-text", style={
                        "position": "absolute",
                        "left": "50%",
                        "top": "50%",
                        "transform": "translate(-50%, -50%)",
                        "color": "#fff",
                        "fontWeight": "600",
                        "fontSize": "0.9rem"
                    })
                ], id='unified-upload-progress-bar', style={
                    "position": "relative",
                    "display": "none",
                    "width": "100%",
                    "height": "24px",
                    "background": "#333",
                    "borderRadius": "8px",
                    "overflow": "hidden",
                    "marginTop": "16px"
                }),
                html.Div(id='unified-upload-output', style={"marginTop": "16px"})
            ]),
            # Process File Button
            html.Div([
                html.Button([
                    html.I(className="fas fa-play", style={"marginRight": "8px"}),
                    "Process File"
                ], id="unified-process-file-btn", className="btn-success", style={
                    "width": "100%",
                    "maxWidth": "300px"
                })
            ], style={"textAlign": "center", "marginBottom": "16px"}),
            # Status Messages
            html.Div(id="unified-upload-status", style={"marginBottom": "8px", "textAlign": "center"}),
            html.Div(id="unified-processing-status", style={"marginTop": "8px", "textAlign": "center"})
        ], id="unified-upload-container", style={"marginBottom": "32px"}),
        
    # Processing Options Section
    html.Div([
        html.H3([
            html.I(className="fas fa-cogs", style={"marginRight": "12px", "color": "#00bcd4"}),
            "Processing Options"
        ], style={
            "color": "#fff", 
            "fontSize": "1.5rem", 
            "marginBottom": "32px",
            "textAlign": "center"
        }),
        
        # Note about processing options
        html.Div([
            html.I(className="fas fa-lightbulb", style={"color": "#ffc107", "marginRight": "8px"}),
            html.Span("These options are for custom processing. Use 'Process File' buttons above for automatic processing.", 
                     style={"color": "#ffc107", "fontWeight": "500"})
        ], style={
            "background": "rgba(255,193,7,0.1)",
            "border": "1px solid rgba(255,193,7,0.3)",
            "borderRadius": "8px",
            "padding": "12px 16px",
            "marginBottom": "24px",
            "textAlign": "center"
        }),
        
        # Two-column layout for options
        html.Div([
            # Left column - Interpolation
            html.Div([
                html.H4([
                    html.I(className="fas fa-expand-arrows-alt", style={"marginRight": "8px", "color": "#00bcd4"}),
                    "Interpolation"
                ], style={"color": "#fff", "marginBottom": "16px", "textAlign": "center"}),
                
                html.Div([
                    dcc.Checklist(
                        id='interpolate-checkbox',
                        options=[{'label': 'Enable Interpolation', 'value': 'interpolate'}],
                        value=[],
                        style={"marginBottom": "16px"}
                    ),
                    html.Div([
                        html.Label("Number of Slices:", style={
                            "color": "#b0bec5", 
                            "display": "block", 
                            "marginBottom": "8px",
                            "fontWeight": "500"
                        }),
                        dcc.Input(
                            id='slices-input',
                            type='number',
                            min=1,
                            step=1,
                            value=1,
                            style={
                                "width": "100%",
                                "padding": "12px",
                                "borderRadius": "8px",
                                "border": "1px solid #444",
                                "background": "#2d2d2d",
                                "color": "#fff",
                                "fontSize": "1rem"
                            }
                        )
                    ], id='interpolate-options')
                ], style={
                    "background": "rgba(0,188,212,0.1)",
                    "borderRadius": "12px",
                    "padding": "24px",
                    "border": "1px solid rgba(0,188,212,0.3)"
                })
            ], style={"flex": "1", "marginRight": "16px"}),
            
            # Right column - Imputation
            html.Div([
                html.H4([
                    html.I(className="fas fa-magic", style={"marginRight": "8px", "color": "#00bcd4"}),
                    "Imputation"
                ], style={"color": "#fff", "marginBottom": "16px", "textAlign": "center"}),
                
                html.Div([
                    dcc.Checklist(
                        id='impute-checkbox',
                        options=[{'label': 'Enable Imputation', 'value': 'impute'}],
                        value=[],
                        style={"marginBottom": "16px"}
                    ),
                    html.Div([
                        html.Label("Radius:", style={
                            "color": "#b0bec5", 
                            "display": "block", 
                            "marginBottom": "8px",
                            "fontWeight": "500"
                        }),
                        dcc.Input(
                            id='radius-input',
                            type='number',
                            min=1,
                            step=1,
                            value=1,
                            style={
                                "width": "100%",
                                "padding": "12px",
                                "borderRadius": "8px",
                                "border": "1px solid #444",
                                "background": "#2d2d2d",
                                "color": "#fff",
                                "fontSize": "1rem"
                            }
                        )
                    ], id='impute-options')
                ], style={
                    "background": "rgba(0,188,212,0.1)",
                    "borderRadius": "12px",
                    "padding": "24px",
                    "border": "1px solid rgba(0,188,212,0.3)"
                })
            ], style={"flex": "1", "marginLeft": "16px"})
        ], style={"display": "flex", "marginBottom": "48px"})
    ]),
    
    # Run Button Section
    html.Div([
        html.Button(
            [
                html.I(className="fas fa-play-circle", style={"marginRight": "12px", "fontSize": "1.2rem"}),
                "Run Processing Pipeline"
            ],
            id="run-button",
            style={
                "background": "linear-gradient(135deg, #00bcd4, #00acc1)",
                "color": "#fff",
                "border": "none",
                "padding": "16px 32px",
                "borderRadius": "12px",
                "fontSize": "1.2rem",
                "fontWeight": "600",
                "cursor": "pointer",
                "transition": "all 0.3s ease",
                "boxShadow": "0 4px 15px rgba(0,188,212,0.3)",
                "width": "100%",
                "maxWidth": "400px"
            }
        ),
        html.Div(
            "For standard file uploads, you can also click this button to process your file.",
            style={
                "color": "#b0bec5",
                "fontSize": "1rem",
                "marginTop": "12px",
                "textAlign": "center"
            }
        ),
        dcc.Loading(
            id="loading-form",
            type="circle",
            color="#00bcd4",
            children=[
                html.Div(id="loading-output", style={"marginTop": "16px", "textAlign": "center"})
            ]
        )
    ], style={"textAlign": "center"})
    
], style={"maxWidth": "1200px", "margin": "0 auto"})