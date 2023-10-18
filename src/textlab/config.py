import os
from pathlib import Path

# SET PATHS
filepath = Path(__file__)


class Config:
    GLOBALSEED = 42
    PROJECTPATH = filepath.parents[2]
    PROJECTNAME = PROJECTPATH.name.lower()
    LOGSPATH = os.path.join(PROJECTPATH, "logs")
    TORCHPROFILERPATH = os.path.join(LOGSPATH, "torch_profiler")
    SIMPLEPROFILERPATH = os.path.join(LOGSPATH, "simple_profiler")
    CHKPTSPATH = os.path.join(PROJECTPATH, "models", "trials")
    MODELPATH = os.path.join(PROJECTPATH, "models", "production", "model.onnx")
    DATAPATH = os.path.join(PROJECTPATH, "data", "cache")
    PREDSPATH = os.path.join(PROJECTPATH, "data", "predictions", "predictions.pt")
    SPLITSPATH = os.path.join(PROJECTPATH, "data", "training_split")
    WANDBPATH = os.path.join(PROJECTPATH, "logs", "wandb")
    CSVLOGGERPATH = os.path.join(PROJECTPATH, "logs", "csv")
    OPTUNAPATH = os.path.join(PROJECTPATH, "logs", "optuna")
