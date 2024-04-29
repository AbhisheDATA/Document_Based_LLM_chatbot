from pathlib import Path
from src.helper import read_yaml

CONFIG_FILE_PATH = Path("config/config.yaml")
config=read_yaml(CONFIG_FILE_PATH)
path=config.TEMP_DIR
def langchain_document_loader(TMP_DIR):
    """
    Load files from TMP_DIR (temporary directory) as documents. Files can be in txt, pdf, CSV or docx format.
    """

    documents = []

    pdf_loader = DirectoryLoader(
        TMP_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True
    )
    documents.extend(pdf_loader.load())
    return documents

documents = langchain_document_loader(path)
print(type(documents))
print(f"\nNumber of documents: {len(documents)}")