from dash import dcc, html, Input, Output, State, callback_context
import dash
from flask import session
import logging
import pandas as pd
import io
import zipfile
from datetime import datetime

logger = logging.getLogger("metavision")


def register_export_callback(app, cache):
    @app.callback(
        Output("tissues-selected-count", "children"),
        Input("tissues-checklist", "value"),
        Input("select-all-tissues", "value"),
    )
    def update_tissues_count(selected_tissues, select_all_check):
        if select_all_check != []:
            tissues_list = cache.get(f"{session['session_id']}:tissue_ids")
            return str(len(tissues_list))
        logger.debug(f"Selected tissues: {selected_tissues}")
        count = len(selected_tissues)
        logger.debug(f"Number of selected tissues: {count}")
        return str(count)

    @app.callback(
        Output("molecules-selected-count", "children"),
        Input("molecule-checklist", "value"),
        Input("select-all-molecules", "value"),
    )
    def update_molecules_count(selected_molecules, select_all_check):
        if select_all_check != []:
            molecules_list = cache.get(f"{session['session_id']}:molecules_list")
            return str(len(molecules_list))
        logger.debug(f"Selected molecules: {selected_molecules}")
        count = len(selected_molecules)
        logger.debug(f"Number of selected molecules: {count}")
        return str(count)

    @app.callback(
        Output("molecule-checklist", "value"),
        Input("select-all-molecules", "value"),
        State("molecule-checklist", "options"),
        prevent_initial_call=True
    )
    def handle_select_all_molecules(select_all, options):
        if "select-all" in select_all:
            return [option["value"] for option in options]
        return []

    @app.callback(
        Output("tissues-checklist", "value"),
        Input("select-all-tissues", "value"),
        State("tissues-checklist", "options"),
        prevent_initial_call=True
    )
    def handle_select_all_tissues(select_all, options):
        if "select-all" in select_all:
            return [option["value"] for option in options]
        return []

    @app.callback(
        Output("output-directory", "value"),
        Input("browse-button", "n_clicks"),
        prevent_initial_call=True
    )
    def handle_browse_click(n_clicks):
        if n_clicks:
            # For now, set a default directory since we can't use file dialogs in Dash
            return "./exports"
        return dash.no_update

    @app.callback(
        Output("export-status", "children"),
        Input("export-button", "n_clicks"),
        [
            State("select-all-molecules", "value"),
            State("molecule-checklist", "value"),
            State("normalization-select", "value"),
            State("export-format", "value"),
            State("output-directory", "value")
        ],
        prevent_initial_call=True
    )
    def update_export_status(n_clicks, select_all_check, checked_molecules, normalization, export_format, output_dir):
        if not n_clicks:
            return ""
        
        try:
            logger.info("Exporting file...")

            molecules_list = cache.get(f"{session['session_id']}:molecules_list")
            if not molecules_list:
                return html.Div([
                    html.I(className="fas fa-exclamation-circle", style={"color": "#f44336", "fontSize": "1.5rem", "marginRight": "10px"}),
                    html.Span("No molecules list found in cache.", style={"color": "#f44336", "fontSize": "16px"})
                ], style={"display": "flex", "alignItems": "center", "padding": "15px", "backgroundColor": "rgba(244,67,54,0.1)", "borderRadius": "8px", "border": "1px solid #f44336"})
                
            if select_all_check == [] and checked_molecules == []:
                return html.Div([
                    html.I(className="fas fa-exclamation-circle", style={"color": "#f44336", "fontSize": "1.5rem", "marginRight": "10px"}),
                    html.Span("No molecules selected for export.", style={"color": "#f44336", "fontSize": "16px"})
                ], style={"display": "flex", "alignItems": "center", "padding": "15px", "backgroundColor": "rgba(244,67,54,0.1)", "borderRadius": "8px", "border": "1px solid #f44336"})

            selected_molecules = []
            if "select-all" in select_all_check:
                selected_molecules = molecules_list
            else:
                selected_molecules = checked_molecules

            selected_tissues = cache.get(f"{session['session_id']}:tissue_ids")
            if selected_tissues is None:
                return html.Div([
                    html.I(className="fas fa-exclamation-circle", style={"color": "#f44336", "fontSize": "1.5rem", "marginRight": "10px"}),
                    html.Span("No tissues selected for export.", style={"color": "#f44336", "fontSize": "16px"})
                ], style={"display": "flex", "alignItems": "center", "padding": "15px", "backgroundColor": "rgba(244,67,54,0.1)", "borderRadius": "8px", "border": "1px solid #f44336"})

            logger.info(f"Selected molecules: {selected_molecules}")
            logger.info(f"Selected tissues: {selected_tissues}")

            if normalization == "none":
                df = cache.get(f"{session['session_id']}:df")
            else:
                df = cache.get(f"{session['session_id']}:norm_df")
                
            if df is None:
                return html.Div([
                    html.I(className="fas fa-exclamation-circle", style={"color": "#f44336", "fontSize": "1.5rem", "marginRight": "10px"}),
                    html.Span("No data found in cache.", style={"color": "#f44336", "fontSize": "16px"})
                ], style={"display": "flex", "alignItems": "center", "padding": "15px", "backgroundColor": "rgba(244,67,54,0.1)", "borderRadius": "8px", "border": "1px solid #f44336"})
                
            filtered_df = df[df['tissue_id'].isin(selected_tissues)]
            
            if filtered_df.empty:
                return html.Div([
                    html.I(className="fas fa-exclamation-circle", style={"color": "#f44336", "fontSize": "1.5rem", "marginRight": "10px"}),
                    html.Span("No data found for selected tissues.", style={"color": "#f44336", "fontSize": "16px"})
                ], style={"display": "flex", "alignItems": "center", "padding": "15px", "backgroundColor": "rgba(244,67,54,0.1)", "borderRadius": "8px", "border": "1px solid #f44336"})
            
            # Return success message
            return html.Div([
                html.I(className="fas fa-check-circle", style={"color": "#4CAF50", "fontSize": "1.5rem", "marginRight": "10px"}),
                html.Span("Export completed! Download will start automatically.", style={"color": "#4CAF50", "fontSize": "16px"})
            ], style={"display": "flex", "alignItems": "center", "padding": "15px", "backgroundColor": "rgba(76,175,80,0.1)", "borderRadius": "8px", "border": "1px solid #4CAF50"})
                
        except Exception as e:
            logger.error(f"Export error: {str(e)}")
            return html.Div([
                html.I(className="fas fa-exclamation-circle", style={"color": "#f44336", "fontSize": "1.5rem", "marginRight": "10px"}),
                html.Span(f"Export error: {str(e)}", style={"color": "#f44336", "fontSize": "16px"})
            ], style={"display": "flex", "alignItems": "center", "padding": "15px", "backgroundColor": "rgba(244,67,54,0.1)", "borderRadius": "8px", "border": "1px solid #f44336"})

    @app.callback(
        Output("export-download-data", "data"),
        Input("export-button", "n_clicks"),
        [
            State("select-all-molecules", "value"),
            State("molecule-checklist", "value"),
            State("normalization-select", "value"),
            State("export-format", "value")
        ],
        prevent_initial_call=True
    )
    def export_file(n_clicks, select_all_check, checked_molecules, normalization, export_format):
        if not n_clicks:
            return dash.no_update
        try:
            logger.info("Exporting file...")

            molecules_list = cache.get(f"{session['session_id']}:molecules_list")
            if not molecules_list:
                logger.error("No molecules list found in cache.")
                return dcc.send_bytes(b'', "error.txt")
            if select_all_check == [] and checked_molecules == []:
                logger.error("No molecules selected for export.")
                return dcc.send_bytes(b'', "error.txt")

            selected_molecules = []
            if "select-all" in select_all_check:
                selected_molecules = molecules_list
            else:
                selected_molecules = checked_molecules

            selected_tissues = cache.get(f"{session['session_id']}:tissue_ids")
            if selected_tissues is None:
                logger.error("No tissues selected for export.")
                return dcc.send_bytes(b'', "error.txt")

            logger.info(f"Selected molecules: {selected_molecules}")
            logger.info(f"Selected tissues: {selected_tissues}")

            if normalization == "none":
                df = cache.get(f"{session['session_id']}:df")
            else:
                df = cache.get(f"{session['session_id']}:norm_df")

            if df is None:
                logger.error("No data found in cache.")
                return dcc.send_bytes(b'', "error.txt")

            filtered_df = df[df['tissue_id'].isin(selected_tissues)]

            if filtered_df.empty:
                logger.error("No data found for selected tissues.")
                return dcc.send_bytes(b'', "error.txt")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if export_format == "multiple":
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, 'w') as zip_file:
                    for tissue in selected_tissues:
                        tissue_df = filtered_df[filtered_df['tissue_id'] == tissue]
                        if not tissue_df.empty:
                            export_df = tissue_df[['tissue_id'] + selected_molecules]
                            csv_buffer = io.StringIO()
                            export_df.to_csv(csv_buffer, index=False)
                            zip_file.writestr(f"tissue_{tissue}.csv", csv_buffer.getvalue())
                buffer.seek(0)
                return dcc.send_bytes(buffer.getvalue(), f"tissues_export_{timestamp}.zip")
            else:
                export_df = filtered_df[['tissue_id'] + selected_molecules]
                buffer = io.StringIO()
                export_df.to_csv(buffer, index=False)
                buffer.seek(0)
                return dcc.send_bytes(buffer.getvalue().encode('utf-8'), f"export_{timestamp}.csv")
        except Exception as e:
            logger.error(f"Export error: {str(e)}")
            return dcc.send_bytes(b'', "error.txt")

    @app.callback(
        Output("export-button", "children"),
        Output("export-reset-interval", "disabled"),
        Output("export-reset-interval", "n_intervals"),
        Input("export-button", "n_clicks"),
        Input("export-reset-interval", "n_intervals"),
        prevent_initial_call=True
    )
    def update_export_button(n_clicks, n_intervals):
        ctx = callback_context
        if not ctx.triggered:
            return [
                html.I(className="fas fa-download", style={"marginRight": "14px", "fontSize": "1.5rem"}),
                "EXPORT DATA"
            ], True, 0

        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if triggered_id == "export-button":
            # Show exporting, enable interval
            return [
                html.I(className="fas fa-spinner fa-spin", style={"marginRight": "14px", "fontSize": "1.5rem"}),
                "EXPORTING..."
            ], False, 0

        elif triggered_id == "export-reset-interval":
            # Reset to default and disable interval
            return [
                html.I(className="fas fa-download", style={"marginRight": "14px", "fontSize": "1.5rem"}),
                "EXPORT DATA"
            ], True, 0

        return [
            html.I(className="fas fa-download", style={"marginRight": "14px", "fontSize": "1.5rem"}),
            "EXPORT DATA"
        ], True, 0
                
                
