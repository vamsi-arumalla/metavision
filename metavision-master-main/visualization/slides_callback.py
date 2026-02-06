import matplotlib
matplotlib.use('Agg')  # Must be set before importing pyplot

from dash import Output, Input, State, callback, dcc, html, ALL, MATCH, ctx, no_update
import dash
from flask import session
import logging

logger = logging.getLogger("metavision")

def register_slides_callback(app, cache):
    """
    Simplified slides callback - removed molecule refresh to prevent unwanted slide generation
    """
    # No callbacks needed - slides will only update when user explicitly changes colormap or view
    pass