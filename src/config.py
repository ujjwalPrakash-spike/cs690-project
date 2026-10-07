import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Automatically load .env if present
load_dotenv()

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

def load_config(config_path: str = None) -> dict:
    target = Path(config_path) if config_path else CONFIG_PATH
    if not target.exists():
        raise FileNotFoundError(f"Configuration file not found: {target}")
    with open(target, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def get_gemini_api_key() -> str:
    # 1. Environment variable
    key = os.environ.get("GEMINI_API_KEY")
    if key and key.strip():
        return key.strip()

    # 2. Local geminikey.txt fallback if present
    gemini_file = Path(__file__).resolve().parent.parent / "geminikey.txt"
    if gemini_file.exists():
        with open(gemini_file, "r") as f:
            content = f.read().strip()
            if content:
                return content

    raise ValueError(
        "GEMINI_API_KEY environment variable is not set and no geminikey.txt found. "
        "Please export GEMINI_API_KEY='...' or create a .env file."
    )

def get_gemini_model(default_model: str = "gemini-3.5-flash-lite") -> str:
    return os.environ.get("GEMINI_MODEL", default_model).strip()
