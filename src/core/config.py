from omegaconf import OmegaConf
from pathlib import Path
from src.utils.logger import logger

def load_config(config_name: str = "rpi5_16gb.yaml"):
    """
    Loads the system configuration from the config directory.
    Defaults to the Raspberry Pi 5 16GB profile.
    """
    config_path = Path(f"config/{config_name}")
    
    if not config_path.exists():
        logger.error("Configuration file not found", config_path=str(config_path))
        raise FileNotFoundError(f"Config file {config_path} not found.")
        
    logger.info("Loading configuration", config_file=config_name)
    config = OmegaConf.load(config_path)
    return config
