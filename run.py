import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from hydra.core.hydra_config import HydraConfig

from src.utils.env import PROJECT_ROOT
def run(cfg):
    pass

@hydra.main(config_path=str(PROJECT_ROOT / "configs"), config_name="default", version_base=None)
def main(cfg):
    print(cfg)
    run(cfg)


if __name__ == "__main__":
    main()
