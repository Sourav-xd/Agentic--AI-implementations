#hr_chatbot/src/utils.py
import os
import json
import logging
from typing import Dict, Any, Union

def setup_logger(log_path_str: str, logger_name_str: str = "app") -> logging.Logger:
    """
    Configures a logger to write to a file and console.

    Args:
        log_path_str (str): Absolute path to the log directory.
        logger_name_str (str): Name of the logger instance.

    Returns:
        logging.Logger: Configured logger instance.
    """
    os.makedirs(log_path_str, exist_ok=True)
    log_file_path_str = os.path.join(log_path_str, "system.log")

    logger_ins = logging.getLogger(logger_name_str)
    logger_ins.setLevel(logging.INFO)

    # File Handler
    file_handler_ins = logging.FileHandler(log_file_path_str)
    file_formatter_ins = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler_ins.setFormatter(file_formatter_ins)

    # Stream Handler (Console)
    stream_handler_ins = logging.StreamHandler()
    stream_handler_ins.setFormatter(file_formatter_ins)

    if not logger_ins.handlers:
        logger_ins.addHandler(file_handler_ins)
        logger_ins.addHandler(stream_handler_ins)

    return logger_ins

def load_config(config_rel_path_str: str) -> Dict[str, Any]:
    """
    Loads JSON config and generates absolute paths dynamically.
    
    Args:
        config_rel_path_str (str): Relative path to config file from project root.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - status (str): 'success' or 'error'
            - data (dict): Config data with added 'processed_paths' key.
            - message (str): Status message.
    """
    try:
        # Determine Project Root (Assuming this script is run from root or src)
        current_dir_str = os.getcwd()
        config_abs_path_str = os.path.join(current_dir_str, config_rel_path_str)

        if not os.path.exists(config_abs_path_str):
            return {"status": "error", "message": f"Config not found at: {config_abs_path_str}", "data": {}}

        with open(config_abs_path_str, 'r') as f:
            config_dct = json.load(f)

        # Process Paths: Create Absolute versions
        processed_paths_dct = {}
        if "paths" in config_dct:
            for key_str, rel_path_str in config_dct["paths"].items():
                abs_path_str = os.path.join(current_dir_str, rel_path_str)
                processed_paths_dct[f"{key_str}Abs"] = abs_path_str
                processed_paths_dct[f"{key_str}Rel"] = rel_path_str
                
                # Ensure directories exist
                os.makedirs(abs_path_str, exist_ok=True)

        config_dct["processed_paths"] = processed_paths_dct
        
        return {"status": "success", "data": config_dct, "message": "Config loaded successfully"}

    except Exception as e:
        return {"status": "error", "message": str(e), "data": {}}