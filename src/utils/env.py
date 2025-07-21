import os
from pathlib import Path
from typing import Optional
import dotenv

def get_env(env_name: str, default: Optional[str] = None) -> str:
    if env_name not in os.environ:
        if default is None:
            raise KeyError(
                f"{env_name} not defined and no default value is present!")
        return default

    env_value: str = os.environ[env_name]
    if not env_value:
        if default is None:
            raise ValueError(
                f"{env_name} has yet to be configured and no default value is present!"
            )
        return default

    return env_value

def load_envs(env_file: Optional[str] = None) -> None:
    dotenv.load_dotenv(dotenv_path=env_file, override=True)

load_envs()

PROJECT_ROOT: Path = Path(get_env("PROJECT_ROOT"))
assert PROJECT_ROOT.exists(), "You must configure the PROJECT_ROOT environment variable in a .env file!"

os.chdir(PROJECT_ROOT)