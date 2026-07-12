import importlib
import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*")

import packaging
import packaging.version
import pkg_resources

if not hasattr(pkg_resources, "packaging"):
    pkg_resources.packaging = packaging


def load(name, device="cpu"):
    try:
        clip = importlib.import_module("clip")
        return clip.load(name, device=device)
    except Exception as clip_error:
        try:
            import open_clip
        except Exception as open_clip_error:
            raise RuntimeError(
                "Could not load CLIP. Install openai-clip or open-clip-torch."
            ) from open_clip_error

        model_name = "ViT-B-32-quickgelu" if name == "ViT-B/32" else name.replace("/", "-")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained="openai",
            device=device,
        )
        model.eval()
        print(f"Using open_clip fallback for {name} ({clip_error})")
        return model, preprocess
