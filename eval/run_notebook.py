import nbformat
from nbclient import NotebookClient

nb = nbformat.read("eval/evaluation.ipynb", as_version=4)
NotebookClient(nb, timeout=600, kernel_name="python3").execute()
nbformat.write(nb, "eval/evaluation.ipynb")