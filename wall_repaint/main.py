# run this if there is no model available: gdown --id 1seW8qSnaBOQ4_S9bQ4ThVOdeJGYJ-f74 -O src/PIH/pretrained/ckpt_g39.pth
import sys
import os
# Add PIH path to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src/PIH")))

# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)

# Simulate command-line arguments for inference.py
sys.argv = [
    "wall_repaint/src/PIH/inference.py",
    "--bg", "/workspace/flamTeam/wall_repaint/assets/Harm01_BG.jpeg",
    "--fg", "/workspace/flamTeam/wall_repaint/assets/Harm01_FG.png",
    "--checkpoints", "/workspace/flamTeam/wall_repaint/src/PIH/pretrained/ckpt_g39.pth",
    "--gpu",
]

# Import and run
from src.PIH.inference import Evaluater

evaluater = Evaluater()
evaluater.evaluate()