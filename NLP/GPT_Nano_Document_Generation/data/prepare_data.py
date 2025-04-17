import os
import logging
logger = logging.getLogger(name=__name__)

def read_input(path: str = "data/shakespeare.txt"):
    # Change relative path to absolute path
    abs_path = os.path.abspath(path) 
    logger.info(msg=f"Reading file from path: {abs_path}")
    try:
        with open(abs_path, mode="r", encoding="utf-8") as f:
            chars = f.read()
            return chars
    except FileNotFoundError:
        logger.error("File not found in the path: %s", abs_path)
    except Exception as e:
        logger.error("An error occurred while loading the input file: %s", e, exc_info=True)
