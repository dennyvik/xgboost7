import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.train_pipeline import run_pipeline
from src.utils.config import load_config


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    config = load_config(config_path)
    run_pipeline(config, config_path=config_path)


if __name__ == "__main__":
    main()