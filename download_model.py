from huggingface_hub import snapshot_download
from transformers.utils import move_cache
from os import makedirs, environ

def download_model(repository: str, local_path: str):
  makedirs(local_path, exist_ok=True)
  snapshot_download(
    repository,
    local_dir=local_path,
    ignore_patterns="*.pt"
  )
  move_cache()

if __name__ == "__main__":
  model_repository = environ.get("DOWNLOAD_MODEL_REPOSITORY")
  model_local_path = environ.get("DOWNLOAD_MODEL_LOCAL_PATH")
  download_model(model_repository, model_local_path)
