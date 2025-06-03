import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
'''# Set up log filename with timestamp


# Create logs directory path
logs_dir = os.path.join(os.getcwd(), "logs",LOG_FILE)
os.makedirs(logs_dir, exist_ok=True)

# Full path to the log file
logs_file_path = os.path.join(logs_dir, LOG_FILE)
# Configure logging'''

logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)  
logs_file_path = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=logs_file_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


if __name__ == "__main__":
    logging.info("Logger initialized successfully.")