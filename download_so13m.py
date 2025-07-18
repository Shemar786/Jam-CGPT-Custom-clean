from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="apcl/jam_so",
    repo_type="model",
    local_dir="pretrained_so13m"
)
