# logger.py
import logging
import os
from datetime import datetime

LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Create a formatter that includes the timestamp
formatter = logging.Formatter("[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s ", datefmt="%m/%d/%Y %H:%M:%S")

# Create a file handler with the specified log file
file_handler = logging.FileHandler(LOG_FILE_PATH)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Add the file handler to the root logger
logging.root.addHandler(file_handler)

# Create a console handler to log to the terminal
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Add the console handler to the root logger
logging.root.addHandler(console_handler)
