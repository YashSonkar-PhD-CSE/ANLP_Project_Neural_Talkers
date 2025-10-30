import torch

# Path to your original checkpoint
old_ckpt_path = "./checkpoints/autoencoder_epoch10.pt"
# Path to save the new checkpoint
new_ckpt_path = "./checkpoints/autoencoder_epoch10_512.pt"

state_dict = torch.load(old_ckpt_path, map_location="cpu")
new_state_dict = {}

for k, v in state_dict.items():
    # Check for positional encoding tensors of shape [5000, 64]
    if isinstance(v, torch.Tensor) and v.shape == (5000, 64):
        print(f"Slicing {k}: {v.shape} -> {v[:512].shape}")
        new_state_dict[k] = v[:512]
    else:
        new_state_dict[k] = v

torch.save(new_state_dict, new_ckpt_path)
print(f"Saved new checkpoint to {new_ckpt_path}")