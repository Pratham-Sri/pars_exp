# Flask Experiment Lab

A reusable Flask workspace for adding experiments, inspecting local datasets, and recording model evaluation results. The first experiment explores IDRiD retinal images and the PHM 2010 CNC milling benchmark. When dataset files are absent, the UI uses clearly marked synthetic example rows and example chart values so the full interface can be explored without downloading large datasets.

## Run

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Open `http://127.0.0.1:5000`. The app creates its `datasets/` and `data/` directories when it starts. Put ZIP archives in `datasets/IDRiD/` and `datasets/PHM2010/`, then select **Extract local ZIPs** on the experiment page. Extracted files are stored under matching folders in `data/`.

### Getting the source data

- IDRiD: [official dataset page](https://idrid.grand-challenge.org/Data/). The official page points to IEEE DataPort; follow its access, account, and data-use terms, then place the archive in `datasets/IDRiD/`.
- PHM 2010: [PHM Society challenge page](https://phmsociety.org/phm_competition/2010-phm-society-conference-data-challenge/). The official page describes six cutter archives of roughly 800 MB each. Save the original archives in `datasets/PHM2010/`. The page notes that c3 wear labels found in some copies are incorrect; only training cutters c1, c4, and c6 have wear labels.

Dataset archives are not bundled or fetched automatically: IDRiD uses an external access-controlled repository and PHM files are very large. The app performs safe local ZIP extraction. Source pages are provided directly in the UI.

## Model results

Append measured evaluation results to `results/model_metrics.csv` with the columns `dataset,model,metric,value`. Optional columns such as `split` and `notes` are supported. Example:

```csv
dataset,model,metric,value,split,notes
IDRiD,ResNet50,macro_f1,0.81,test,patient-level split
PHM 2010,RandomForest,RMSE,7.4,cutter-held-out,c1/c4/c6 training protocol
```

The app plots the recorded rows and displays their values. If the results file has only its header or no measured results, the UI shows clearly marked synthetic values for layout demonstration; they are not benchmark results and must not be cited. Once recorded results exist, they replace the examples. Classification, segmentation, and regression results should be interpreted separately.

The no-download dataset previews use invented `DEMO-*` rows and are labeled as synthetic. Descriptive statistics and findings shown for those rows are computed from the five constructed examples only; they do not represent dataset-wide analysis. Metadata facts such as IDRiD image dimensions and total published image/annotation counts are kept separate from the local file inventory. The UI marks each dataset as awaiting source data until actual files are present.

## Experiment 08: Overfitting and L2 regularization

The Palmer Penguins regularization experiment uses the local CSV and a deterministic, stratified 30/70 train/test split. A high-capacity RBF softmax classifier with one center per training observation is compared with the same model using an L2 coefficient penalty. The page generates separate decision-boundary and train/test performance plots on each visit, and reports accuracy, macro-F1, log loss, and the generalization gap. Scaling and RBF centers are derived from training data only.

## Add another experiment

1. Add experiment metadata to `EXPERIMENTS` in `dataset_lab/registry.py`.
2. Add each dataset's description, source, feature guide, and analysis notes to `DATASETS` in that same registry.
3. The existing experiment page automatically renders the registered dataset inventory. For a different workflow, add a route/template under `dataset_lab/` and register its blueprint in `create_app`.

## Analysis included

- Supported file inventory, extension counts, and indexed size.
- CSV/Excel schema and first-five-row preview.
- Image dimensions and color mode when Pillow can read the image.
- A sampled seven-channel PHM signal summary (mean, standard deviation, RMS, minimum, maximum) for recognized numeric CSVs.
- Per-dataset interpretation notes covering annotation/task structure, leakage risks, split strategy, class imbalance, and appropriate metrics.

## Experiment 09: Principal Component Analysis

Four-feature PCA on Palmer Penguins, fitted using the training partition. The page displays PC1–PC4 loadings, eigenvalues, individual and cumulative variance, held-out component scores, the PC1 projection line, the PC1–PC2 score plane, and the component dot-product matrix.

## Experiment 10: Linear Discriminant Analysis

Four-feature LDA on Palmer Penguins with a stratified 80/20 train/test split. The page displays LD1/LD2 projections, the reduced LD1 score line, held-out species predictions, confusion matrix, and test accuracy.

## Music recommender systems

The home page includes one recommender hub with subpages for content-based, collaborative, cold-start, user-neighbor, item-neighbor, adjusted-cosine item-neighbor, item k-NN, hybrid, and weighted knowledge-based recommenders. The Jamendo metadata TSVs live in `datasets/`; a 180-track catalog slice is shown with links to the Jamendo track pages. Use the Song Library to open a track, return after listening, and save a listen event with an optional 1–5 rating. Both are stored locally in `data/recommender/interactions.sqlite3`; ratings feed collaborative and neighborhood methods, while listening history informs the hybrid recommender. This records activity entered in this app, not ratings from the Jamendo metadata.

The root `6090214.json` file is Zenodo record metadata for the Last.fm dataset, not its user listening rows. To add its interaction history, provide the extracted Last.fm 1K file `userid-timestamp-artid-artname-traid-traname.tsv` (or a smaller subset). Those tracks have no playable audio in this dataset; they should remain a separate interaction source unless they can be mapped to Jamendo. Last.fm data is provided for non-commercial use; see the [Zenodo record](https://zenodo.org/records/6090214).
