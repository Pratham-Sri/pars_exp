"""Development entry point for the multi-experiment dataset lab."""

from dataset_lab import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
