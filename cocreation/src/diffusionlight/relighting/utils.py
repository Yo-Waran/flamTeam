import argparse
import os
from pathlib import Path
from PIL import Image
import hashlib

import numpy as np
import torch

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}

def save_result(
    image, image_path,
    mask=None, mask_path=None,
    normal=None, normal_path=None,
):
    assert isinstance(image, Image.Image)
    os.makedirs(Path(image_path).parent, exist_ok=True)
    image.save(image_path)

    if (mask is not None) and (mask_path is not None):
        assert isinstance(mask, Image.Image)
        os.makedirs(Path(mask_path).parent, exist_ok=True)
        mask.save(mask_path)

    if (normal is not None) and (normal_path is not None):
        assert isinstance(normal, Image.Image)
        os.makedirs(Path(normal_path).parent, exist_ok=True)
        normal.save(normal_path)
        
def name2hash(name: str):
    """
    @see https://stackoverflow.com/questions/16008670/how-to-hash-a-string-into-8-digits
    """
    hash_number = int(hashlib.sha1(name.encode("utf-8")).hexdigest(), 16) % (10 ** 8)
    return hash_number

# Helper functions for spherical and Cartesian transformations
def create_envmap_grid(size: int) -> torch.Tensor:
    """Create the grid of the environment map in spherical coordinates."""
    theta = torch.linspace(0, np.pi * 2, size * 2)
    phi = torch.linspace(0, np.pi, size)
    theta, phi = torch.meshgrid(theta, phi, indexing='xy')
    theta_phi = torch.cat([theta[..., None], phi[..., None]], dim=-1).numpy()
    return theta_phi

def get_cartesian_from_spherical(theta: np.ndarray, phi: np.ndarray, r=1.0) -> np.ndarray:
    """Convert spherical to Cartesian coordinates."""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)

def get_normal_vector(incoming_vector: np.ndarray, reflect_vector: np.ndarray) -> np.ndarray:
    """Calculate normal vector based on incoming and reflect vectors."""
    N = (incoming_vector + reflect_vector) / np.linalg.norm(incoming_vector + reflect_vector, axis=-1, keepdims=True)
    return N
