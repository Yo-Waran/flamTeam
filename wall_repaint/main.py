import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src/PIH")))

# Simulate command-line arguments for inference.py
sys.argv = [
    "wall_repaint/src/PIH/inference.py",  # the script name (ignored)
    "--bg", "/workspace/flamTeam/wall_repaint/assets/Harm01_BG.jpeg",
    "--fg", "/workspace/flamTeam/wall_repaint/assets/Harm01_FG.png",
    "--checkpoints", "/workspace/flamTeam/wall_repaint/src/PIH/pretrained/ckpt_g39.pth",
    "--gpu",  # optional, include if needed
]

# Now import Evaluater class AFTER setting sys.argv
from src.PIH.inference import Evaluater

# Run it as usual
evaluater = Evaluater()
evaluater.evaluate()