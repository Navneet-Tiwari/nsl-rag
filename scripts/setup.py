"""
setup.py
--------
Project scaffold generator for NSL-RAG.
Reads structure.yaml and creates all directories and files.
Idempotent — safe to run multiple times without overwriting existing files.

Usage:
    python scripts/setup.py
"""

import logging
import yaml
from pathlib import Path


# ── Logger Setup ─────────────────────────────────────────────────────────────


def get_logger() -> logging.Logger:
    """
    Configure and return a script-level logger.
    Uses a named logger — does not affect root logger.
    """
    logger = logging.getLogger("nsl_rag.scaffold")

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


log = get_logger()


# ── Config ───────────────────────────────────────────────────────────────────


def load_config(config_path: Path) -> dict:
    """
    Load and return structure configuration from yaml.
    Raises FileNotFoundError if structure.yaml does not exist.
    """
    if not config_path.exists():
        log.error("structure.yaml not found at: %s", config_path)
        raise FileNotFoundError(f"structure.yaml not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    log.info(
        "Loaded config: %s v%s", config["project"]["name"], config["project"]["version"]
    )
    return config


# ── Directory Creation ────────────────────────────────────────────────────────


def create_directories(directories: list, base_path: Path) -> tuple[int, int]:
    """
    Create all directories defined in structure.yaml.
    Returns (created_count, existed_count).
    """
    created = 0
    existed = 0

    for entry in directories:
        dir_path = base_path / entry["path"]
        try:
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                log.info("  [CREATED] dir  → %s", entry["path"])
                created += 1
            else:
                log.info("  [EXISTS]  dir  → %s", entry["path"])
                existed += 1
        except OSError as e:
            log.error("  [FAILED]  dir  → %s | reason: %s", entry["path"], e)

    return created, existed


# ── File Creation ─────────────────────────────────────────────────────────────


def create_files(files: list, base_path: Path) -> tuple[int, int]:
    """
    Create all files defined in structure.yaml.
    Never overwrites existing files.
    Returns (created_count, existed_count).
    """
    created = 0
    existed = 0

    for entry in files:
        file_path = base_path / entry["path"]

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)

            if file_path.exists():
                log.info("  [EXISTS]  file → %s", entry["path"])
                existed += 1
                continue

            description = entry.get("description", "").strip()
            content = build_file_content(entry["path"], description)
            file_path.write_text(content, encoding="utf-8")
            log.info("  [CREATED] file → %s", entry["path"])
            created += 1

        except OSError as e:
            log.error("  [FAILED]  file → %s | reason: %s", entry["path"], e)

    return created, existed


# ── File Content Builders ─────────────────────────────────────────────────────


def build_file_content(file_path: str, description: str) -> str:
    """
    Build starter content for a new file based on its type.
    Python files get a docstring header.
    YAML, TOML, Markdown files get a comment header.
    Special files get their own builder.
    """
    name = Path(file_path).name
    ext = Path(file_path).suffix

    if name == ".gitignore":
        return build_gitignore_content()
    elif name == ".env.example":
        return build_env_example_content()
    elif ext == ".py":
        return build_python_content(name, description)
    elif ext in (".yaml", ".yml"):
        return build_yaml_content(name, description)
    elif ext == ".toml":
        return build_toml_content(name, description)
    elif ext == ".md":
        return build_markdown_content(name, description)
    else:
        return ""


def build_python_content(name: str, description: str) -> str:
    """Generate a Python file with a clean docstring header."""
    if not description:
        return '"""TODO: Add module description."""\n'
    return f'"""\n' f"{name}\n" f'{"-" * len(name)}\n' f"{description}\n" f'"""\n'


def build_yaml_content(name: str, description: str) -> str:
    """Generate a YAML file with a comment header."""
    if not description:
        return f"# {name}\n"
    return f"# {name}\n# {description}\n"


def build_toml_content(name: str, description: str) -> str:
    """Generate a TOML file with a comment header."""
    if not description:
        return f"# {name}\n"
    return f"# {name}\n# {description}\n"


def build_markdown_content(name: str, description: str) -> str:
    """Generate a Markdown file with a title and description."""
    title = name.replace(".md", "").replace("_", " ").title()
    if not description:
        return f"# {title}\n"
    return f"# {title}\n\n{description}\n"


def build_env_example_content() -> str:
    """Generate the .env.example template."""
    return (
        "# .env.example\n"
        "# Copy this file to .env and fill in your values.\n"
        "# NEVER commit .env to git.\n"
        "\n"
        "# Gemini API\n"
        "GEMINI_API_KEY=your_gemini_api_key_here\n"
        "\n"
        "# Environment\n"
        "NSL_RAG_ENV=development\n"
        "\n"
        "# Logging\n"
        "LOG_LEVEL=DEBUG\n"
    )


def build_gitignore_content() -> str:
    """Generate a comprehensive .gitignore for Python projects."""
    return (
        "# Environment\n"
        ".env\n"
        "*.env\n"
        "\n"
        "# Python\n"
        "__pycache__/\n"
        "*.pyc\n"
        "*.pyo\n"
        "*.pyd\n"
        ".Python\n"
        "\n"
        "# Virtual environments\n"
        "venv/\n"
        "venv_3.11/\n"
        ".venv/\n"
        "env/\n"
        "\n"
        "# Testing\n"
        ".pytest_cache/\n"
        "htmlcov/\n"
        ".coverage\n"
        "coverage.xml\n"
        "\n"
        "# Build\n"
        "dist/\n"
        "build/\n"
        "*.egg-info/\n"
        "\n"
        "# IDE\n"
        ".vscode/\n"
        ".idea/\n"
        "\n"
        "# OS\n"
        ".DS_Store\n"
        "Thumbs.db\n"
        "\n"
        "# Notebooks\n"
        ".ipynb_checkpoints/\n"
        "\n"
        "# Logs\n"
        "*.log\n"
        "logs/\n"
    )


# ── Summary ───────────────────────────────────────────────────────────────────


def print_summary(
    config: dict,
    dirs_created: int,
    dirs_existed: int,
    files_created: int,
    files_existed: int,
) -> None:
    """Print a clean summary of scaffold results."""
    log.info("─" * 50)
    log.info(
        "Project  : %s v%s", config["project"]["name"], config["project"]["version"]
    )
    log.info("Dirs     : %d created, %d already existed", dirs_created, dirs_existed)
    log.info("Files    : %d created, %d already existed", files_created, files_existed)
    log.info("─" * 50)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    base_path = Path(__file__).parent.parent
    config_path = base_path / "structure.yaml"

    log.info("=" * 50)
    log.info("NSL-RAG Project Scaffold Generator")
    log.info("=" * 50)

    config = load_config(config_path)

    log.info("Creating directories...")
    dirs_created, dirs_existed = create_directories(
        config["structure"]["directories"], base_path
    )

    log.info("Creating files...")
    files_created, files_existed = create_files(config["structure"]["files"], base_path)

    print_summary(config, dirs_created, dirs_existed, files_created, files_existed)

    log.info("Scaffold complete.")
    log.info("Next: fill in pyproject.toml and config files.")


if __name__ == "__main__":
    main()
