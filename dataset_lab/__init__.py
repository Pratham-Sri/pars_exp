from flask import Flask
from pathlib import Path


def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY="development-key-change-before-deployment",
        DATA_ROOT=Path(app.root_path).parent / "data",
        ARCHIVE_ROOT=Path(app.root_path).parent / "datasets",
        MAX_CONTENT_LENGTH=128 * 1024 * 1024,
    )
    if test_config:
        app.config.update(test_config)
    Path(app.config["DATA_ROOT"]).mkdir(parents=True, exist_ok=True)
    Path(app.config["ARCHIVE_ROOT"]).mkdir(parents=True, exist_ok=True)

    from .routes import bp
    app.register_blueprint(bp)
    return app
