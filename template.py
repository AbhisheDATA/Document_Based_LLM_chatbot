import os
from pathlib import Path
import logging

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
    "store_index.py",
    "requirements.txt",
    "config/config.yaml",
    "Data",
    "Data/docs",
    "Docs/temp",
    "Docs/vector_stores"

]


for filepath in list_of_files:
   filepath = Path(filepath)
   filedir, filename = os.path.split(filepath)

   if filedir !="":
      os.makedirs(filedir, exist_ok=True)
      logging.info(f"Creating directory; {filedir} for the file {filename}")

   if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
      with open(filepath, 'w') as f:
         pass
         logging.info(f"Creating empty file: {filepath}")

   else:
      logging.info(f"{filename} is already created")
      
      
    