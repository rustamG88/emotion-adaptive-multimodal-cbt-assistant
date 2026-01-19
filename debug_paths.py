from pathlib import Path
from config.config import PROJECT_ROOT, load_config, _load_yaml_config

print(f"PROJECT_ROOT: {PROJECT_ROOT}")
cfg = load_config()
cfg_dict = _load_yaml_config(PROJECT_ROOT / "config" / "config.yaml")

text_model_rel_path = cfg_dict["iemocap_fusion"]["text_model_path"]
print(f"Rel path from config: {text_model_rel_path}")

abs_path = PROJECT_ROOT / text_model_rel_path
print(f"Absolute path: {abs_path}")
print(f"Exists? {abs_path.exists()}")

if abs_path.exists():
    import torch
    from src.models.text_distilbert import build_text_model
    print("Attempting to load model...")
    try:
        model = build_text_model(num_classes=9, device=torch.device("cpu"))
        model.load_state_dict(torch.load(abs_path, map_location="cpu"))
        print("Success!")
    except Exception as e:
        print(f"Error loading: {e}")
