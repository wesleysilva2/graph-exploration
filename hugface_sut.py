from huggingface_hub import hf_hub_download

# Baixa o arquivo 'graph_gtfs.gpickle' do dataset 'suntdataset/sunt'
hf_hub_download(
    repo_id="suntdataset/sunt",
    repo_type="dataset",
    filename="graph_designer/graph_gtfs.gpickle",
    local_dir="./sunt",
    local_dir_use_symlinks=False
)
