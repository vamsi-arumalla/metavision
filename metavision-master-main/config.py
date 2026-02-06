import logging
import os
from datetime import datetime
import sys

# Global variable to track if logger is already configured
_logger_configured = False

def setup_logger(log_level=logging.INFO, log_dir="logs", name="metavision", use_file=True):
    """
    Configure and set up the logger for the Metavision project.
    
    Args:
        log_level (int): Logging level (default: logging.INFO)
        log_dir (str): Directory to save log files (default: "logs")
        name (str): Logger name (default: "metavision")
        use_file (bool): Whether to log to a file (default: True)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    global _logger_configured
    
    # Create a logger
    logger = logging.getLogger(name)
    
    # Only configure the logger once to avoid duplicate handlers
    if not _logger_configured:
        logger.setLevel(log_level)
        
        # Clear existing handlers to avoid duplicates
        if logger.handlers:
            logger.handlers.clear()
        
        # Create console handler (outputs to stdout)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        
        # Create formatter and add it to the console handler
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        
        # Add console handler to logger
        logger.addHandler(console_handler)
        
        # Add file handler if requested
        if use_file:
            # Create logs directory if it doesn't exist
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            # Create file handler with timestamp in the filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_handler = logging.FileHandler(
                os.path.join(log_dir, f"{name}_{timestamp}.log")
            )
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            
            # Add file handler to logger
            logger.addHandler(file_handler)
        
        # Set flag to indicate logger is configured
        _logger_configured = True
    
    return logger