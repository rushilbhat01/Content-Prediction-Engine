import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def select_torch_device(torch_module, prefer="cuda"):
    prefer = (prefer or "auto").lower()
    if prefer == "cpu":
        return "cpu"

    if prefer in ("auto", "cuda"):
        if torch_module.cuda.is_available():
            name = torch_module.cuda.get_device_name(0)
            print(f"Device: cuda ({name})")
            return "cuda"
        if prefer == "cuda":
            raise RuntimeError(
                "CUDA was requested, but PyTorch cannot see a CUDA device. "
                "Install a CUDA-enabled torch build and check your NVIDIA driver."
            )

    if prefer in ("auto", "mps") and hasattr(torch_module.backends, "mps"):
        if torch_module.backends.mps.is_available():
            print("Device: mps")
            return "mps"

    print("Device: cpu")
    return "cpu"


def move_model_to_device(model, device):
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    return model
