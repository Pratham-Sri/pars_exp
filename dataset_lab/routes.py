from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Blueprint, abort, current_app, flash, jsonify, redirect, render_template, request, url_for

from .analysis import extract_archives, inspect_dataset, load_metrics
from .registry import DATASETS, EXPERIMENTS

bp = Blueprint("main", __name__)


def generate_benchmark_plot(metrics, output_path: Path):
    """Generate high-resolution, publication-quality benchmark comparison charts."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Filter and group metrics by dataset
    df = pd.DataFrame(metrics)
    if df.empty:
        return
    
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=180)
    
    # Palette definition
    colors_idrid = ["#164e3d", "#2d7a5e", "#4ea583", "#76c8a7"]
    colors_phm = ["#1e3a8a", "#2563eb", "#3b82f6", "#60a5fa"]
    
    # 1. IDRiD Plot (Classification Metrics: QWK, Macro-F1, ROC-AUC)
    df_idrid = df[df["dataset"] == "IDRiD"]
    if not df_idrid.empty:
        idrid_metrics = ["QWK", "Macro-F1", "ROC-AUC"]
        models_idrid = ["EfficientNet-B4", "Swin-Transformer", "DenseNet-121", "ResNet-50"]
        x = np.arange(len(idrid_metrics))
        width = 0.18
        
        for idx, model in enumerate(models_idrid):
            model_df = df_idrid[df_idrid["model"] == model]
            vals = []
            for m in idrid_metrics:
                row = model_df[model_df["metric"] == m]
                vals.append(row["value"].values[0] if not row.empty else 0)
            rects = ax1.bar(x + idx * width - (width * 1.5), vals, width, label=model, color=colors_idrid[idx], edgecolor="#fff", linewidth=0.8)
            for rect in rects:
                height = rect.get_height()
                if height > 0:
                    ax1.annotate(f"{height:.3f}",
                                 xy=(rect.get_x() + rect.get_width() / 2, height),
                                 xytext=(0, 3), textcoords="offset points",
                                 ha="center", va="bottom", fontsize=7.5, fontweight="600", color="#1c2b24")
        
        ax1.set_title("IDRiD Retinal Ordinal Grading & Classification", fontsize=12, fontweight="bold", pad=12, color="#164e3d")
        ax1.set_xticks(x)
        ax1.set_xticklabels(idrid_metrics, fontsize=10, fontweight="600")
        ax1.set_ylabel("Score (0.00 – 1.00)", fontsize=10, fontweight="500")
        ax1.set_ylim(0.65, 1.02)
        ax1.legend(frameon=True, facecolor="#f8faf7", edgecolor="#d5e0d8", fontsize=8.5, loc="lower right")
        ax1.grid(axis="y", linestyle="--", alpha=0.5)
    
    # 2. PHM 2010 Plot (Tool Wear Regression RMSE and MAE in μm)
    df_phm = df[df["dataset"] == "PHM 2010"]
    if not df_phm.empty:
        phm_metrics = ["RMSE", "MAE"]
        models_phm = ["TCN", "1D-CNN + BiLSTM", "XGBoost", "Random Forest"]
        x = np.arange(len(phm_metrics))
        width = 0.18
        
        for idx, model in enumerate(models_phm):
            model_df = df_phm[df_phm["model"] == model]
            vals = []
            for m in phm_metrics:
                row = model_df[model_df["metric"] == m]
                vals.append(row["value"].values[0] if not row.empty else 0)
            rects = ax2.bar(x + idx * width - (width * 1.5), vals, width, label=model, color=colors_phm[idx], edgecolor="#fff", linewidth=0.8)
            for rect in rects:
                height = rect.get_height()
                if height > 0:
                    ax2.annotate(f"{height:.2f} μm",
                                 xy=(rect.get_x() + rect.get_width() / 2, height),
                                 xytext=(0, 3), textcoords="offset points",
                                 ha="center", va="bottom", fontsize=7.5, fontweight="600", color="#0f172a")
        
        ax2.set_title("PHM 2010 Flank Wear VB Prediction Error (Held-Out c4)", fontsize=12, fontweight="bold", pad=12, color="#1e3a8a")
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"{m} (Lower is better)" for m in phm_metrics], fontsize=10, fontweight="600")
        ax2.set_ylabel("Error in Flank Wear (μm)", fontsize=10, fontweight="500")
        ax2.set_ylim(0, 11.5)
        ax2.legend(frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=8.5, loc="upper left")
        ax2.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout(pad=2.5)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


@bp.get("/")
def home():
    return render_template("home.html", experiments=EXPERIMENTS)


@bp.route("/experiments/<slug>", methods=["GET", "POST"])
def experiment(slug):
    if slug not in EXPERIMENTS:
        abort(404)
    exp_data = EXPERIMENTS[slug]
    if slug == "recommender-systems":
        from .recommenders import METHODS, load_catalog
        project_root = Path(current_app.root_path).parent
        catalog = load_catalog(project_root)
        return render_template(
            "recommender_hub.html",
            experiment=exp_data,
            methods=METHODS,
            catalog_count=len(catalog),
            source_url="https://github.com/MTG/mtg-jamendo-dataset",
            interaction_metadata_present=(project_root / "6090214.json").exists(),
            interaction_archive_present=any((project_root / "datasets" / name).exists() for name in ("lastfm-dataset-1K.tar.gz", "lastfm-dataset-360K.tar.gz", "usersha1-artmbid-artname-plays.tsv", "userid-timestamp-artid-artname-traid-traname.tsv")),
            interaction_source_url="https://zenodo.org/records/6090214",
        )
    if slug == "multimodal-feature-extraction":
        from .multimodal import generate_segment_max_feature_charts
        static_dir = Path(current_app.root_path) / "static"
        project_root = Path(current_app.root_path).parent
        corpus_dir = project_root / "data" / "frank_ocean_corpus"
        source_audio = corpus_dir / "01_Nikes_source.mp3"
        chart_names = [
            "multimodal_sample_max_centroid.png",
            "multimodal_sample_max_rms.png",
            "multimodal_sample_max_zcr.png",
        ]
        timeline, samples = [], []
        if source_audio.exists():
            result = generate_segment_max_feature_charts(source_audio, corpus_dir / "samples", static_dir)
            chart_names = result["charts"]
            timeline = [
                {
                    **entry,
                    "start": f"{int(entry['start_sec'] // 60):02d}:{int(entry['start_sec'] % 60):02d}",
                    "end": f"{int(entry['end_sec'] // 60):02d}:{int(entry['end_sec'] % 60):02d}",
                    "duration": f"{entry['duration_sec']:.2f} s",
                }
                for entry in result["timeline"]
            ]
            samples = [
                {
                    "name": f"Sample {index + 1}",
                    "spectral_centroid_hz": result["features"]["spectral_centroid_hz"][index],
                    "rms_dbfs": result["features"]["rms_dbfs"][index],
                    "zero_crossing_rate": result["features"]["zero_crossing_rate"][index],
                }
                for index in range(len(result["features"]["spectral_centroid_hz"]))
            ]
        else:
            chart_names = [name for name in chart_names if (static_dir / name).exists()]
        return render_template(
            "multimodal.html",
            experiment=exp_data,
            charts=[url_for("static", filename=name) for name in chart_names],
            timeline=timeline,
            samples=samples,
        )

    if slug == "bayesian-classification":
        from .bayesian import (
            DATASET_CONFIGS,
            fit_and_evaluate_dataset,
            generate_dataset_charts,
            DATASET_CODE_SNIPPETS
        )
        static_dir = Path(current_app.root_path) / "static"
        
        # Ensure publication charts exist for both real datasets
        p_chart = static_dir / "bayesian_penguins.png"
        if not p_chart.exists():
            generate_dataset_charts("penguins", p_chart)
            
        s_chart = static_dir / "bayesian_seeds.png"
        if not s_chart.exists():
            generate_dataset_charts("wheat_seeds", s_chart)
            
        active_ds = request.args.get("dataset", "penguins")
        if active_ds not in DATASET_CONFIGS:
            active_ds = "penguins"
            
        eval_penguins = fit_and_evaluate_dataset("penguins")
        eval_seeds = fit_and_evaluate_dataset("wheat_seeds")
        
        active_eval = eval_penguins if active_ds == "penguins" else eval_seeds
        active_chart = url_for("static", filename="bayesian_penguins.png" if active_ds == "penguins" else "bayesian_seeds.png")

        return render_template(
            "bayesian.html",
            experiment=exp_data,
            active_ds=active_ds,
            configs=DATASET_CONFIGS,
            eval_data=active_eval,
            eval_penguins=eval_penguins,
            eval_seeds=eval_seeds,
            chart_url=active_chart,
            code_snippets=DATASET_CODE_SNIPPETS,
        )

    if slug == "text-nlp-bayesian":
        from .nlp_bayesian import (
            full_nlp_pipeline,
            classify_text_bayes,
            evaluate_test_corpus,
            PYTHON_NLP_BAYES_CODE,
            CLASSES,
            CLASS_COLORS
        )
        sample_text = request.form.get("text") if request.method == "POST" else (
            "Artificial intelligence models optimize deep neural architectures on distributed cloud servers. "
            "Physicians in clinical research hospitals evaluate diabetic retinopathy lesions and prescribe targeted therapies. "
            "Simultaneously, atmospheric greenhouse emissions accelerate Arctic glacier melting across oceanic ecosystems."
        )
        nlp_result = full_nlp_pipeline(sample_text)
        bayes_result = classify_text_bayes(sample_text)
        test_samples, test_acc = evaluate_test_corpus()

        return render_template(
            "nlp_bayesian.html",
            experiment=exp_data,
            sample_text=sample_text,
            nlp_result=nlp_result,
            bayes_result=bayes_result,
            test_samples=test_samples,
            test_acc=test_acc,
            python_code=PYTHON_NLP_BAYES_CODE,
            classes=CLASSES,
            class_colors=CLASS_COLORS,
        )

    if slug == "bayesian-risk-minimization":
        from .risk import (
            CLASSES,
            ACTIONS,
            DEFAULT_LOSS_MATRIX,
            LOSS_PRESETS,
            compute_conditional_risks,
            evaluate_test_scenarios,
            PYTHON_RISK_MINIMIZATION_CODE,
        )
        init_posteriors = [0.65, 0.18, 0.17]
        init_calc = compute_conditional_risks(init_posteriors, DEFAULT_LOSS_MATRIX)
        test_scenarios = evaluate_test_scenarios(DEFAULT_LOSS_MATRIX)

        return render_template(
            "risk_minimization.html",
            experiment=exp_data,
            classes=CLASSES,
            actions=ACTIONS,
            loss_matrix=DEFAULT_LOSS_MATRIX,
            loss_presets=LOSS_PRESETS,
            init_posteriors=init_posteriors,
            calc=init_calc,
            test_scenarios=test_scenarios,
            python_code=PYTHON_RISK_MINIMIZATION_CODE,
        )

    if slug == "linear-regression-gradient-descent":
        from .linear_regression import (
            load_penguins_regression_data,
            gradient_descent_linear_regression,
            generate_gradient_descent_charts,
            generate_sample_predictions,
            PYTHON_GRADIENT_DESCENT_CODE
        )
        static_dir = Path(current_app.root_path) / "static"
        chart_path = static_dir / "linear_regression_bd.png"
        if not chart_path.exists():
            generate_gradient_descent_charts(chart_path)

        opt_res, sample_predictions = generate_sample_predictions()

        return render_template(
            "linear_regression.html",
            experiment=exp_data,
            res=opt_res,
            sample_predictions=sample_predictions,
            chart_url=url_for("static", filename="linear_regression_bd.png"),
            python_code=PYTHON_GRADIENT_DESCENT_CODE,
        )

    if slug == "logistic-regression-gradient-descent":
        from .logistic_regression import (
            load_penguins_classification_data,
            generate_logistic_regression_charts,
            generate_classification_test_ledger,
            predict_sample,
            PYTHON_LOGISTIC_REGRESSION_CODE
        )
        static_dir = Path(current_app.root_path) / "static"
        chart_path = static_dir / "logistic_gradient_descent_curve.png"
        if not chart_path.exists():
            generate_logistic_regression_charts(chart_path)

        bin_res, multi_res, ledger = generate_classification_test_ledger()
        init_prediction = predict_sample(47.5, 215.0, bin_res, multi_res)

        return render_template(
            "logistic_regression.html",
            experiment=exp_data,
            bin_res=bin_res,
            multi_res=multi_res,
            test_ledger=ledger,
            init_prediction=init_prediction,
            chart_url=url_for("static", filename="logistic_gradient_descent_curve.png"),
            python_code=PYTHON_LOGISTIC_REGRESSION_CODE,
        )

    if slug == "logistic-regularization":
        from .regularization import run_regularization_experiment
        analysis = run_regularization_experiment(Path(current_app.root_path) / "static")
        return render_template(
            "regularization.html",
            experiment=exp_data,
            analysis=analysis,
        )

    if slug == "principal-component-analysis":
        from .pca import run_pca_experiment
        analysis = run_pca_experiment(Path(current_app.root_path) / "static")
        return render_template("pca.html", experiment=exp_data, analysis=analysis)

    if slug == "linear-discriminant-analysis":
        from .lda import run_lda_experiment
        analysis = run_lda_experiment(Path(current_app.root_path) / "static")
        return render_template("lda.html", experiment=exp_data, analysis=analysis)

    if slug == "dataset-explorer":
        feature_datasets = {key: DATASETS[key] for key in exp_data["datasets"]}
        metrics_path = Path(current_app.root_path).parent / "results" / "model_metrics.csv"
        metrics, _ = load_metrics(metrics_path)
        chart_url = None
        if metrics:
            chart_path = Path(current_app.root_path) / "static" / "metrics.png"
            generate_benchmark_plot(metrics, chart_path)
            chart_url = url_for("static", filename="metrics.png")
        return render_template("experiment.html", experiment=exp_data, datasets=feature_datasets, chart_url=chart_url)

    if request.method == "POST":
        key = request.form.get("dataset")
        if key not in exp_data["datasets"]:
            abort(400)
        details = DATASETS[key]
        root = Path(current_app.config["ARCHIVE_ROOT"]) / details["folder"]
        target = Path(current_app.config["DATA_ROOT"]) / details["folder"]
        count, logs = extract_archives(root, target)
        flash(f"Extraction pipeline complete: {count} archives processed. " + ("; ".join(logs[:5]) if logs else "Local directory synchronized."), "success" if count else "info")
        return redirect(url_for("main.experiment", slug=slug))

    datasets, data_root = {}, Path(current_app.config["DATA_ROOT"])
    for key in exp_data["datasets"]:
        details = DATASETS[key]
        source_dir = Path(current_app.config["ARCHIVE_ROOT"]) / details["folder"]
        info = inspect_dataset(source_dir, data_root / details["folder"], dataset_key=key)
        
        # Format ground-truth preview table
        preview_df = pd.DataFrame(details["preview_rows"])
        info["preview"] = preview_df.to_html(index=False, classes="data-table", escape=True, border=0)
        info["preview_name"] = details["preview_name"]
        
        info["archive_names"] = [p.name for p in source_dir.glob("*.zip")] if source_dir.exists() else []
        datasets[key] = {**details, **info}

    metrics_path = Path(current_app.root_path).parent / "results" / "model_metrics.csv"
    metrics, metric_error = load_metrics(metrics_path)
    metrics_mode = "VERIFIED BENCHMARK RESULTS · 5-FOLD STRATIFIED / CUTTER-HELD-OUT CV"
    
    plot_url = None
    if metrics:
        output = Path(current_app.root_path) / "static" / "metrics.png"
        generate_benchmark_plot(metrics, output)
        plot_url = url_for("static", filename="metrics.png")

    return render_template(
        "experiment.html",
        experiment=exp_data,
        datasets=datasets,
        metrics=metrics,
        metrics_mode=metrics_mode,
        metric_error=metric_error,
        plot_url=plot_url,
        metrics_path="results/model_metrics.csv"
    )


@bp.post("/api/bayesian/classify")
def bayesian_classify_api():
    """Live interactive API for Bayesian Single & Multi-Feature Classification on Real Datasets."""
    from .bayesian import classify_custom_point
    data = request.get_json(silent=True) or request.form
    try:
        ds_id = data.get("dataset", "penguins")
        x1 = float(data.get("x1", 45.0))
        x2 = float(data.get("x2", 200.0))
        mode = data.get("mode", "multi")
        prior_mode = data.get("prior_mode", "empirical")
        
        result = classify_custom_point(ds_id, x1, x2, mode=mode, prior_mode=prior_mode)
        return jsonify({"status": "success", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@bp.post("/api/nlp/process")
def nlp_process_api():
    """Live interactive API for NLP paragraph extraction & Bayesian text classification."""
    from .nlp_bayesian import full_nlp_pipeline, classify_text_bayes
    data = request.get_json(silent=True) or request.form
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"status": "error", "message": "Text paragraph is empty"}), 400
    try:
        nlp_res = full_nlp_pipeline(text)
        clf_res = classify_text_bayes(text)
        return jsonify({
            "status": "success",
            "nlp": nlp_res,
            "classification": clf_res
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@bp.post("/api/risk/calculate")
def risk_calculate_api():
    """Live interactive API for Bayesian Risk Minimization & Action Triggering."""
    from .risk import compute_conditional_risks, evaluate_test_scenarios, DEFAULT_LOSS_MATRIX
    data = request.get_json(silent=True) or request.form
    try:
        posteriors = [
            float(data.get("p1", 0.65)),
            float(data.get("p2", 0.18)),
            float(data.get("p3", 0.17))
        ]
        loss_matrix = data.get("loss_matrix", DEFAULT_LOSS_MATRIX)
        calc = compute_conditional_risks(posteriors, loss_matrix)
        scenarios = evaluate_test_scenarios(loss_matrix)
        return jsonify({
            "status": "success",
            "calc": calc,
            "scenarios": scenarios
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@bp.post("/api/regression/optimize")
def regression_optimize_api():
    """Live interactive API for Gradient Descent Linear Regression on Palmer Penguins."""
    from .linear_regression import load_penguins_regression_data, gradient_descent_linear_regression
    data = request.get_json(silent=True) or request.form
    try:
        alpha = float(data.get("alpha", 0.05))
        iterations = int(data.get("iterations", 400))
        flipper_input = float(data.get("flipper_input", 200.0))

        df, x, y, species = load_penguins_regression_data()
        res = gradient_descent_linear_regression(x, y, alpha=alpha, iterations=iterations)

        predicted_mass = res["slope"] * flipper_input + res["intercept"]
        return jsonify({
            "status": "success",
            "slope": res["slope"],
            "intercept": res["intercept"],
            "r2_score": res["r2_score"],
            "rmse": res["rmse"],
            "initial_cost": res["initial_cost"],
            "final_cost": res["final_cost"],
            "flipper_input": flipper_input,
            "predicted_mass": predicted_mass,
            "cost_sample": res["cost_history"][::max(1, len(res["cost_history"]) // 25)]
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@bp.post("/api/logistic/optimize")
def logistic_optimize_api():
    """Live interactive API for Binary & Multiclass Logistic Regression on Palmer Penguins."""
    from .logistic_regression import (
        load_penguins_classification_data,
        fit_binary_logistic_regression,
        fit_multiclass_logistic_regression,
        predict_sample
    )
    data = request.get_json(silent=True) or request.form
    try:
        alpha = float(data.get("alpha", 0.1))
        iterations = int(data.get("iterations", 500))
        bill = float(data.get("bill_length", 47.5))
        flipper = float(data.get("flipper_length", 215.0))

        df, X, y_multi, species_list, X_bin, y_bin = load_penguins_classification_data()
        bin_res = fit_binary_logistic_regression(X_bin, y_bin, alpha=alpha, iterations=iterations)
        multi_res = fit_multiclass_logistic_regression(X, y_multi, K=3, alpha=alpha, iterations=iterations)

        pred = predict_sample(bill, flipper, bin_res, multi_res)

        step_bin = max(1, len(bin_res["cost_history"]) // 25)
        step_multi = max(1, len(multi_res["cost_history"]) // 25)

        return jsonify({
            "status": "success",
            "binary": {
                "weights_norm": bin_res["weights_norm"],
                "bias_norm": bin_res["bias_norm"],
                "weights_raw": bin_res["weights_raw"],
                "bias_raw": bin_res["bias_raw"],
                "initial_cost": bin_res["initial_cost"],
                "final_cost": bin_res["final_cost"],
                "accuracy": bin_res["accuracy"],
                "confusion_matrix": bin_res["confusion_matrix"],
                "p_gentoo": pred["binary"]["p_gentoo"],
                "p_adelie": pred["binary"]["p_adelie"],
                "prediction": pred["binary"]["prediction"],
                "confidence": pred["binary"]["confidence"],
                "cost_sample": bin_res["cost_history"][::step_bin]
            },
            "multiclass": {
                "initial_cost": multi_res["initial_cost"],
                "final_cost": multi_res["final_cost"],
                "accuracy": multi_res["accuracy"],
                "confusion_matrix": multi_res["confusion_matrix"],
                "probabilities": pred["multiclass"]["probabilities"],
                "winner": pred["multiclass"]["winner"],
                "confidence": pred["multiclass"]["confidence"],
                "cost_sample": multi_res["cost_history"][::step_multi]
            },
            "input": {
                "bill_length": bill,
                "flipper_length": flipper,
                "alpha": alpha,
                "iterations": iterations
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@bp.route("/experiments/recommender-systems/<method>", methods=["GET", "POST"])
def recommender_method(method):
    from .recommenders import METHODS, load_catalog, load_history, load_ratings, preference_tags, recommend, save_rating

    methods = {entry["slug"]: entry for entry in METHODS}
    if method not in methods:
        abort(404)
    project_root = Path(current_app.root_path).parent
    catalog = load_catalog(project_root)
    db_path = Path(current_app.config["DATA_ROOT"]) / "recommender" / "interactions.sqlite3"
    user_id = (request.values.get("user_id") or "listener_1").strip()[:48]
    if request.method == "POST" and request.form.get("action") == "rate":
        try:
            save_rating(db_path, user_id, request.form.get("track_id", ""), float(request.form.get("rating", "")))
            flash("Rating saved.", "success")
        except (TypeError, ValueError) as exc:
            flash(str(exc), "error")
        return redirect(url_for("main.recommender_method", method=method, user_id=user_id))

    options = {
        "seed_track": request.values.get("seed_track") or (catalog[0]["track_id"] if catalog else ""),
        "target_track": request.values.get("target_track") or (catalog[0]["track_id"] if catalog else ""),
        "epochs": min(100, max(3, request.values.get("epochs", default=10, type=int))),
        "seen_mode": request.values.get("seen_mode", "unseen"),
        "k": min(20, max(1, request.values.get("k", default=5, type=int))),
        "preferences": request.form.getlist("preferences") if request.method == "POST" else request.args.getlist("preferences"),
        "genre_weight": request.values.get("genre_weight", default="1"),
        "mood_weight": request.values.get("mood_weight", default="1"),
        "instrument_weight": request.values.get("instrument_weight", default="1"),
    }
    analysis = {"items": [], "metrics": [], "message": "", "neighbors": [], "predicted_rating": None, "epoch_losses": []}
    if request.method == "POST" and request.form.get("action") == "recommend":
        analysis = recommend(method, catalog, load_ratings(db_path), user_id, options, history=load_history(db_path, limit=10000))
    return render_template(
        "recommender_method.html",
        method=methods[method], methods=METHODS, catalog=catalog,
        tag_groups=preference_tags(catalog), user_id=user_id,
        options=options, analysis=analysis,
    )


@bp.route("/experiments/recommender-systems/library", methods=["GET", "POST"])
def recommender_library():
    from .recommenders import load_catalog, load_user_ratings, record_listening, save_rating

    project_root = Path(current_app.root_path).parent
    catalog = load_catalog(project_root)
    db_path = Path(current_app.config["DATA_ROOT"]) / "recommender" / "interactions.sqlite3"
    user_id = (request.values.get("user_id") or "listener_1").strip()[:48]
    if request.method == "POST":
        track_id = request.form.get("track_id", "")
        if track_id not in {track["track_id"] for track in catalog}:
            abort(404)
        try:
            record_listening(db_path, user_id, track_id)
            rating = request.form.get("rating", "").strip()
            if rating:
                save_rating(db_path, user_id, track_id, float(rating))
            flash("Listening history saved" + (" with rating." if rating else "."), "success")
        except (TypeError, ValueError) as exc:
            flash(str(exc), "error")
        return redirect(url_for("main.recommender_library", user_id=user_id, q=request.form.get("q", "")))
    query = request.args.get("q", "").strip().lower()
    visible_tracks = [track for track in catalog if not query or query in track["title"].lower() or query in track["artist"].lower() or any(query in tag.lower() for tag in track["tags"])]
    ratings = load_user_ratings(db_path, user_id)
    return render_template(
        "recommender_library.html", catalog=visible_tracks, user_id=user_id,
        query=request.args.get("q", ""), ratings=ratings,
    )


@bp.get("/experiments/recommender-systems/history")
def recommender_history():
    from .recommenders import load_catalog, load_history, load_interaction_counts, load_user_ratings

    project_root = Path(current_app.root_path).parent
    catalog = load_catalog(project_root)
    by_id = {track["track_id"]: track for track in catalog}
    db_path = Path(current_app.config["DATA_ROOT"]) / "recommender" / "interactions.sqlite3"
    user_id = (request.args.get("user_id") or "listener_1").strip()[:48]
    history = load_history(db_path, user_id=user_id, limit=250)
    history = [{**entry, "track": by_id.get(entry["track_id"])} for entry in history]
    ratings = [{"track_id": track_id, "rating": rating, "track": by_id.get(track_id)} for track_id, rating in load_user_ratings(db_path, user_id).items()]
    return render_template(
        "recommender_history.html", user_id=user_id, history=history,
        ratings=ratings, counts=load_interaction_counts(db_path, user_id),
    )


@bp.get("/recommender/listen/<track_id>")
def recommender_listen(track_id):
    from .recommenders import load_catalog, save_track_open

    catalog = load_catalog(Path(current_app.root_path).parent)
    track = next((item for item in catalog if item["track_id"] == track_id), None)
    if not track or not track.get("url", "").startswith("https://"):
        abort(404)
    save_track_open(Path(current_app.config["DATA_ROOT"]) / "recommender" / "interactions.sqlite3", request.args.get("user_id", "listener"), track_id)
    return redirect(track["url"])



