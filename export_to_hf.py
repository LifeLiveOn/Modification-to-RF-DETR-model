from huggingface_hub import HfApi, HfFolder

api = HfApi()
repo_id = "tnkchaseme/rfdetr-roof-assessment"

# api.upload_file(
#     path_or_fileobj="exported_models/inference_model.onnx",
#     path_in_repo="inference_model.onnx",
#     repo_id=repo_id,
#     repo_type="model",
# )

api.upload_file(
    path_or_fileobj="merged_annotations/output/checkpoint.pth",
    path_in_repo="checkpoint.pth",
    repo_id=repo_id,
    repo_type="model",
)

api.upload_file(
    path_or_fileobj="merged_annotations/output/checkpoint_best_ema.pth",
    path_in_repo="checkpoint_best_ema.pth",
    repo_id=repo_id,
    repo_type="model",
)
