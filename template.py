import os
import logging
from pathlib import Path

def setup_logger():
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')
    return logging.getLogger(__name__)

list_of_files = [
    "src/__init__.py",
    "src/helper.py",
    "src/RAG_Chatbot.py",
    ".env",
    "setup.py",
    "research/trials.ipynb",
    "app.py",
    "requirements.txt",
    "config/config.yaml",
    "Data",
    "Data/docs",
    "Data/temp",
    "Data/vector_stores"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir = filepath.parent

    if not filedir.exists():
        filedir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Creating directory: {filedir}")

    if not filepath.exists() or os.path.getsize(filepath) == 0:
        with open(filepath, 'w') as f:
            logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filepath.name} already exists")

