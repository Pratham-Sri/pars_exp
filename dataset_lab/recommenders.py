"""Local music catalog and recommender algorithms for the Flask lab."""

from __future__ import annotations

import csv
import math
import sqlite3
from functools import lru_cache
from pathlib import Path

import numpy as np

CATALOG_LIMIT = 180
METHODS = [
    {"slug": "content-based", "title": "Content Based", "detail": "Similar tracks from Jamendo genre, mood, and instrument tags."},
    {"slug": "collaborative", "title": "Collaborative Filtering", "detail": "Latent-factor predictions learned from listener ratings."},
    {"slug": "cold-start", "title": "Cold Start", "detail": "Recommendations from a new listener’s selected preferences."},
    {"slug": "user-neighbor", "title": "User Based Nearest Neighbor", "detail": "Nearest listeners and weighted rating predictions."},
    {"slug": "item-neighbor", "title": "Item Based Nearest Neighbor", "detail": "Item cosine similarity and predicted ratings."},
    {"slug": "adjusted-cosine", "title": "Adjusted Cosine", "detail": "Item similarity after removing each listener’s rating bias."},
    {"slug": "item-knn", "title": "Item Based k-Nearest Neighbor", "detail": "Top-k neighbors for seen or unseen tracks."},
    {"slug": "hybrid", "title": "Hybrid Recommender", "detail": "Blended tag similarity and rating-based predictions."},
    {"slug": "knowledge-based", "title": "Knowledge Based", "detail": "Preference scores with editable category weights."},
]


def _roots(project_root: Path):
    data_root = Path(project_root) / "datasets"
    return data_root / "mtg-jamendo-subset.tsv", data_root / "mtg-jamendo-track-metadata.tsv"


@lru_cache(maxsize=2)
def _load_catalog_cached(project_root_string: str):
    tag_file, metadata_file = _roots(Path(project_root_string))
    tags_by_id = {}
    with tag_file.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        next(reader, None)
        for row in reader:
            if len(row) < 6:
                continue
            track_id, artist_id, album_id, path, duration = row[:5]
            tags = [tag.strip() for tag in row[5:] if tag.strip()]
            try:
                duration_value = float(duration)
            except ValueError:
                duration_value = 0.0
            tags_by_id[track_id] = {
                "track_id": track_id,
                "artist_id": artist_id,
                "album_id": album_id,
                "audio_path": path,
                "duration": duration_value,
                "tags": tags,
            }

    metadata_by_id = {}
    with metadata_file.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        next(reader, None)
        for row in reader:
            if len(row) < 8:
                continue
            track_id = row[0]
            metadata_by_id[track_id] = {
                "title": row[3].strip() if len(row) > 3 else track_id,
                "artist": row[4].strip() if len(row) > 4 else "Unknown artist",
                "album": " ".join(value.strip() for value in row[5:-2] if value.strip()),
                "release_date": row[-2].strip(),
                "url": row[-1].strip().replace("http://", "https://"),
            }

    catalog = []
    for track_id, detail in tags_by_id.items():
        info = metadata_by_id.get(track_id)
        if not info:
            continue
        tags = detail["tags"]
        catalog.append({**detail, **info,
                        "genres": [tag.removeprefix("genre---") for tag in tags if tag.startswith("genre---")],
                        "moods": [tag.removeprefix("mood/theme---") for tag in tags if tag.startswith("mood/theme---")],
                        "instruments": [tag.removeprefix("instrument---") for tag in tags if tag.startswith("instrument---")],
                        "tag_set": set(tags)})
    catalog.sort(key=lambda track: int(track["track_id"].split("_")[-1]))
    if len(catalog) > CATALOG_LIMIT:
        indices = np.linspace(0, len(catalog) - 1, CATALOG_LIMIT, dtype=int)
        catalog = [catalog[index] for index in indices]
    return catalog


def load_catalog(project_root: Path):
    tag_file, metadata_file = _roots(project_root)
    if not tag_file.exists() or not metadata_file.exists():
        return []
    return _load_catalog_cached(str(Path(project_root).resolve()))


def _db(db_path: Path):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.execute("CREATE TABLE IF NOT EXISTS ratings (user_id TEXT NOT NULL, track_id TEXT NOT NULL, rating REAL NOT NULL CHECK(rating BETWEEN 1 AND 5), updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY(user_id, track_id))")
    connection.execute("CREATE TABLE IF NOT EXISTS track_opens (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, track_id TEXT NOT NULL, opened_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)")
    connection.execute("CREATE TABLE IF NOT EXISTS listening_history (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, track_id TEXT NOT NULL, listened_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)")
    return connection


def save_rating(db_path: Path, user_id: str, track_id: str, rating: float):
    user_id = (user_id or "").strip()[:48]
    rating = float(rating)
    if not user_id or not math.isfinite(rating) or not 1 <= rating <= 5:
        raise ValueError("Enter a profile name and a rating from 1 to 5.")
    connection = _db(db_path)
    try:
        connection.execute("INSERT INTO ratings(user_id, track_id, rating) VALUES (?, ?, ?) ON CONFLICT(user_id, track_id) DO UPDATE SET rating=excluded.rating, updated_at=CURRENT_TIMESTAMP", (user_id, track_id, rating))
        connection.commit()
    finally:
        connection.close()


def save_track_open(db_path: Path, user_id: str, track_id: str):
    connection = _db(db_path)
    try:
        connection.execute("INSERT INTO track_opens(user_id, track_id) VALUES (?, ?)", ((user_id or "listener").strip()[:48], track_id))
        connection.commit()
    finally:
        connection.close()


def record_listening(db_path: Path, user_id: str, track_id: str):
    user_id = (user_id or "").strip()[:48]
    if not user_id or not track_id:
        raise ValueError("Enter a listener profile before logging a listen.")
    connection = _db(db_path)
    try:
        connection.execute("INSERT INTO listening_history(user_id, track_id) VALUES (?, ?)", (user_id, track_id))
        connection.commit()
    finally:
        connection.close()


def load_ratings(db_path: Path):
    connection = _db(db_path)
    try:
        return [(str(user), str(track), float(rating)) for user, track, rating in connection.execute("SELECT user_id, track_id, rating FROM ratings")]
    finally:
        connection.close()


def load_history(db_path: Path, user_id: str | None = None, limit: int = 100):
    connection = _db(db_path)
    try:
        if user_id:
            rows = connection.execute("SELECT user_id, track_id, listened_at FROM listening_history WHERE user_id=? ORDER BY id DESC LIMIT ?", (user_id, limit))
        else:
            rows = connection.execute("SELECT user_id, track_id, listened_at FROM listening_history ORDER BY id DESC LIMIT ?", (limit,))
        return [{"user_id": user, "track_id": track, "listened_at": listened_at} for user, track, listened_at in rows]
    finally:
        connection.close()


def load_user_ratings(db_path: Path, user_id: str):
    connection = _db(db_path)
    try:
        return {track: float(rating) for track, rating in connection.execute("SELECT track_id, rating FROM ratings WHERE user_id=? ORDER BY updated_at DESC", (user_id,))}
    finally:
        connection.close()


def load_interaction_counts(db_path: Path, user_id: str):
    connection = _db(db_path)
    try:
        listens = connection.execute("SELECT COUNT(*) FROM listening_history WHERE user_id=?", (user_id,)).fetchone()[0]
        ratings = connection.execute("SELECT COUNT(*) FROM ratings WHERE user_id=?", (user_id,)).fetchone()[0]
        return {"listens": listens, "ratings": ratings}
    finally:
        connection.close()


def _tag_similarity(left, right):
    union = left["tag_set"] | right["tag_set"]
    return len(left["tag_set"] & right["tag_set"]) / len(union) if union else 0.0


def _index(catalog):
    return {track["track_id"]: track for track in catalog}


def _format_recommendations(scores, catalog, excluded=(), reasons=None, limit=8):
    by_id = _index(catalog)
    excluded = set(excluded)
    reasons = reasons or {}
    output = []
    for track_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        if track_id in excluded or track_id not in by_id:
            continue
        item = by_id[track_id]
        output.append({**item, "score": float(score), "reason": reasons.get(track_id, "Recommendation score")})
        if len(output) >= limit:
            break
    return output


def _matrix(ratings):
    users, items = sorted({row[0] for row in ratings}), sorted({row[1] for row in ratings})
    user_pos, item_pos = {key: i for i, key in enumerate(users)}, {key: i for i, key in enumerate(items)}
    matrix = np.full((len(users), len(items)), np.nan)
    for user, item, rating in ratings:
        matrix[user_pos[user], item_pos[item]] = rating
    return users, items, user_pos, item_pos, matrix


def _cosine(left, right):
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 1:
        return 0.0
    a, b = left[mask], right[mask]
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def _user_similarity_predictions(user_id, ratings):
    users, items, user_pos, item_pos, matrix = _matrix(ratings)
    if user_id not in user_pos or len(items) == 0:
        return {}, []
    target_index = user_pos[user_id]
    target = matrix[target_index]
    target_mean = float(np.nanmean(target))
    neighbors = []
    for other_index, other_user in enumerate(users):
        if other_index == target_index:
            continue
        common = np.isfinite(target) & np.isfinite(matrix[other_index])
        if common.sum() < 2:
            continue
        a, b = target[common], matrix[other_index][common]
        a, b = a - a.mean(), b - b.mean()
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        sim = float(np.dot(a, b) / denom) if denom else 0.0
        if sim > 0:
            neighbors.append((other_user, sim, other_index))
    neighbors.sort(key=lambda row: row[1], reverse=True)
    predictions = {}
    for item_index, item_id in enumerate(items):
        if np.isfinite(target[item_index]):
            continue
        numerator = denominator = 0.0
        for neighbor_id, sim, neighbor_index in neighbors[:20]:
            value = matrix[neighbor_index, item_index]
            if np.isfinite(value):
                neighbor_mean = float(np.nanmean(matrix[neighbor_index]))
                numerator += sim * (value - neighbor_mean)
                denominator += sim
        if denominator:
            predictions[item_id] = float(np.clip(target_mean + numerator / denominator, 1, 5))
    return predictions, [(user, sim) for user, sim, _ in neighbors[:5]]


def _item_similarities(ratings, adjusted=False):
    users, items, user_pos, item_pos, matrix = _matrix(ratings)
    centered = matrix.copy()
    if adjusted:
        for row in range(centered.shape[0]):
            observed = np.isfinite(centered[row])
            if observed.any():
                centered[row, observed] -= np.nanmean(centered[row])
    similarities = np.zeros((len(items), len(items)))
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            sim = _cosine(centered[:, i], centered[:, j])
            similarities[i, j] = similarities[j, i] = sim
    return users, items, user_pos, item_pos, matrix, similarities


def _item_prediction(user_id, target_id, ratings, adjusted=False, k=20):
    users, items, user_pos, item_pos, matrix, similarities = _item_similarities(ratings, adjusted)
    if user_id not in user_pos or target_id not in item_pos:
        return None
    u, i = user_pos[user_id], item_pos[target_id]
    rated = np.flatnonzero(np.isfinite(matrix[u]))
    rated = [j for j in rated if j != i]
    nearest = sorted(rated, key=lambda j: similarities[i, j], reverse=True)[:k]
    weighted = [(similarities[i, j], matrix[u, j]) for j in nearest if similarities[i, j] > 0]
    if not weighted:
        return None
    baseline = float(np.nanmean(matrix[u]))
    denom = sum(sim for sim, _ in weighted)
    score = baseline + sum(sim * (value - baseline) for sim, value in weighted) / denom if adjusted else sum(sim * value for sim, value in weighted) / denom
    return float(np.clip(score, 1, 5)), len(weighted), baseline


def _latent_factor_predictions(user_id, ratings):
    users, items, user_pos, item_pos, matrix = _matrix(ratings)
    if user_id not in user_pos or len(users) < 2 or len(items) < 2:
        return {}
    observed = np.isfinite(matrix)
    if observed.sum() < 4:
        return {}
    global_mean = float(np.nanmean(matrix))
    user_means = np.nanmean(matrix, axis=1)
    item_means = np.nanmean(matrix, axis=0)
    user_means = np.where(np.isfinite(user_means), user_means, global_mean)
    item_means = np.where(np.isfinite(item_means), item_means, global_mean)
    baseline = np.where(observed, matrix - (user_means[:, None] + item_means[None, :] - global_mean), 0)
    u, singular, vt = np.linalg.svd(baseline, full_matrices=False)
    rank = min(12, max(1, min(baseline.shape) - 1))
    reconstruction = u[:, :rank] @ np.diag(singular[:rank]) @ vt[:rank, :]
    prediction = user_means[:, None] + item_means[None, :] - global_mean + reconstruction
    row = user_pos[user_id]
    return {item: float(np.clip(prediction[row, index], 1, 5)) for index, item in enumerate(items) if not observed[row, index]}


def _matrix_factorization_sgd(ratings, user_id, target_id, epochs=10, factors=8, learning_rate=0.01, regularization=0.02):
    """Fit explicit-feedback user/item factors with SGD and estimate one unrated pair."""
    if len(ratings) < 4:
        return None, "Add at least four listener ratings before training matrix factorization."
    users, items, user_pos, item_pos, matrix = _matrix(ratings)
    if user_id not in user_pos:
        return None, "This profile has no ratings yet. Rate songs under this profile, then retry."
    if target_id not in item_pos:
        return None, "The selected song needs ratings from other profiles before its unknown rating can be estimated."
    u, i = user_pos[user_id], item_pos[target_id]
    if np.isfinite(matrix[u, i]):
        return None, "That song already has a rating for this profile. Select an unrated song."
    if len(users) < 2 or int(np.isfinite(matrix).sum()) < 4:
        return None, "Rate songs under at least two profiles; include the target song under another profile."

    mean = float(np.nanmean(matrix))
    rng = np.random.default_rng(42)
    user_factors = rng.normal(0, 0.08, size=(len(users), factors))
    item_factors = rng.normal(0, 0.08, size=(len(items), factors))
    user_bias = np.zeros(len(users), dtype=float)
    item_bias = np.zeros(len(items), dtype=float)
    observations = [(user_pos[listener], item_pos[track], float(value)) for listener, track, value in ratings]
    rng.shuffle(observations)
    loss_by_epoch = []
    epoch_count = max(3, min(100, int(epochs)))

    for _epoch in range(epoch_count):
        rng.shuffle(observations)
        for row, col, actual in observations:
            predicted = mean + user_bias[row] + item_bias[col] + float(np.dot(user_factors[row], item_factors[col]))
            error = actual - predicted
            old_user = user_factors[row].copy()
            old_item = item_factors[col].copy()
            user_bias[row] += learning_rate * (error - regularization * user_bias[row])
            item_bias[col] += learning_rate * (error - regularization * item_bias[col])
            user_factors[row] += learning_rate * (error * old_item - regularization * old_user)
            item_factors[col] += learning_rate * (error * old_user - regularization * old_item)

        squared_errors = []
        for row, col, actual in observations:
            predicted = mean + user_bias[row] + item_bias[col] + float(np.dot(user_factors[row], item_factors[col]))
            squared_errors.append((actual - predicted) ** 2)
        loss_by_epoch.append(float(np.sqrt(np.mean(squared_errors))))

    estimate = mean + user_bias[u] + item_bias[i] + float(np.dot(user_factors[u], item_factors[i]))
    return {
        "rating": float(np.clip(estimate, 1, 5)),
        "epochs": epoch_count,
        "factors": factors,
        "rating_count": len(observations),
        "profile_count": len(users),
        "epoch_rmse": loss_by_epoch,
    }, ""


def recommend(method, catalog, ratings, user_id="", options=None, history=None):
    """Return recommendation rows plus small method-specific metrics for a page."""
    options = options or {}
    history = history or []
    by_id = _index(catalog)
    user_id = (user_id or "").strip()
    listened_ids = {row["track_id"] for row in history if row["user_id"] == user_id}
    rated_ids = {track for listener, track, _rating in ratings if listener == user_id}
    seen_ids = listened_ids | rated_ids
    seed = options.get("seed_track") or (catalog[0]["track_id"] if catalog else "")
    result = {"items": [], "metrics": [], "message": "", "neighbors": [], "predicted_rating": None}
    if not catalog:
        result["message"] = "Jamendo catalog files are missing from /datasets."
        return result
    if method == "matrix-factorization-sgd":
        target = options.get("target_track") or seed
        fit, message = _matrix_factorization_sgd(
            ratings,
            user_id,
            target,
            epochs=options.get("epochs", 10),
            factors=options.get("factors", 8),
        )
        if fit:
            target_track = by_id.get(target)
            result["predicted_rating"] = fit["rating"]
            result["items"] = [{**target_track, "score": fit["rating"], "reason": "Predicted unknown rating · SGD matrix factorization"}]
            result["metrics"] = [
                f"Epochs: {fit['epochs']}",
                f"Latent factors: {fit['factors']}",
                f"Training ratings: {fit['rating_count']}",
                f"Profiles: {fit['profile_count']}",
            ]
            result["epoch_losses"] = fit["epoch_rmse"]
        else:
            result["message"] = message
    elif method == "content-based":
        if seed not in by_id:
            seed = catalog[0]["track_id"]
        source = by_id[seed]
        scores = {item["track_id"]: _tag_similarity(source, item) for item in catalog if item["track_id"] != seed}
        result["items"] = _format_recommendations(scores, catalog, {seed}, {k: "Shared Jamendo tags" for k in scores})
        result["metrics"] = [f"Seed: {source['title']}", f"Tags: {', '.join(source['tags'][:5]) or 'none'}"]
    elif method == "cold-start":
        selected = set(options.get("preferences", []))
        weights = {"genre": float(options.get("genre_weight", 1)), "mood/theme": float(options.get("mood_weight", 1)), "instrument": float(options.get("instrument_weight", 1))}
        scores, reasons = {}, {}
        for item in catalog:
            score = sum(weights[category] for tag in selected for category in (tag.split("---", 1)[0],) if tag in item["tag_set"])
            if score:
                scores[item["track_id"]] = score
                reasons[item["track_id"]] = "Matches selected preferences"
        result["items"] = _format_recommendations(scores, catalog, reasons=reasons)
        result["metrics"] = [f"Selected tags: {len(selected)}", "New listener profile"]
        if not selected:
            result["message"] = "Select one or more genre, mood, or instrument tags."
    elif method in {"collaborative", "user-neighbor", "hybrid"}:
        predictions, neighbors = _user_similarity_predictions(user_id, ratings)
        result["neighbors"] = [{"user_id": item, "similarity": sim} for item, sim in neighbors]
        if method == "collaborative":
            scores = _latent_factor_predictions(user_id, ratings)
            result["items"] = _format_recommendations(scores, catalog, seen_ids, reasons={k: "Latent-factor predicted rating" for k in scores})
            result["metrics"] = [f"Ratings: {len(ratings)}", f"Profiles: {len({r[0] for r in ratings})}"]
            if not scores:
                result["message"] = "Rate songs under at least two profile names to build the collaborative rating matrix."
        elif method == "user-neighbor":
            result["items"] = _format_recommendations(predictions, catalog, seen_ids, reasons={k: "Weighted nearest-listener prediction" for k in predictions})
            result["metrics"] = [f"Nearest listeners: {len(neighbors)}", f"Ratings: {len(ratings)}"]
            if not predictions:
                result["message"] = "Rate songs under multiple profile names; profiles need at least two overlapping song ratings."
        else:
            content_seeds = [by_id[track_id] for track_id in (list(listened_ids) + [seed]) if track_id in by_id]
            content_scores = {item["track_id"]: max((_tag_similarity(source, item) for source in content_seeds), default=0.0) * 5 for item in catalog}
            all_ids = set(content_scores) | set(predictions)
            blended = {item: 0.5 * content_scores.get(item, 0) + 0.5 * predictions.get(item, 0) for item in all_ids}
            result["items"] = _format_recommendations(blended, catalog, seen_ids | {seed}, reasons={k: "Equal blend of listening history, tags, and rating prediction" for k in blended}, limit=8)
            result["metrics"] = ["Content weight: 0.50", "Collaborative weight: 0.50"]
            if not result["items"]:
                result["message"] = "Choose a seed song for content recommendations or add ratings for collaborative results."
    elif method in {"item-neighbor", "adjusted-cosine", "item-knn"}:
        adjusted = method == "adjusted-cosine"
        _, item_ids, user_pos, item_pos, matrix, similarities = _item_similarities(ratings, adjusted)
        if method == "item-knn":
            seen_mode = options.get("seen_mode", "unseen")
            rated_ids = {item_ids[i] for i in range(len(item_ids)) if user_id in user_pos and np.isfinite(matrix[user_pos[user_id], i])}
            source_ids = ({seed} if seed in by_id else set()) | rated_ids | listened_ids
            if seen_mode == "seen":
                candidates = rated_ids | listened_ids
            else:
                candidates = set(by_id) - rated_ids - listened_ids - source_ids
            scores = {}
            for candidate in candidates:
                if candidate not in item_pos:
                    continue
                sims = [similarities[item_pos[candidate], item_pos[source]] for source in source_ids if source in item_pos]
                if sims:
                    scores[candidate] = max(sims)
            result["items"] = _format_recommendations(scores, catalog, limit=int(options.get("k", 5)), reasons={k: f"Top-{options.get('k', 5)} item-neighbor similarity" for k in scores})
            result["metrics"] = [f"Mode: {seen_mode}", f"k: {options.get('k', 5)}"]
            if not scores:
                result["message"] = "Item k-NN needs shared item ratings from multiple profiles to estimate neighbors."
        else:
            target = options.get("target_track") or seed
            prediction = _item_prediction(user_id, target, ratings, adjusted=adjusted)
            scores = {}
            for item in catalog:
                if item["track_id"] == target:
                    continue
                if user_id in user_pos and item["track_id"] in item_pos and np.isfinite(matrix[user_pos[user_id], item_pos[item["track_id"]]]):
                    continue
                estimate = _item_prediction(user_id, item["track_id"], ratings, adjusted=adjusted)
                if estimate:
                    scores[item["track_id"]] = estimate[0]
            result["metrics"] = [f"Target: {by_id.get(target, {}).get('title', target)}", f"Ratings: {len(ratings)}"]
            if prediction:
                value, neighbor_count, baseline = prediction
                result["predicted_rating"] = value
                result["items"] = _format_recommendations(scores, catalog, reasons={k: "Predicted rating from neighboring items" for k in scores})
                result["metrics"] += [f"Predicted target rating: {value:.2f} / 5", f"Target neighbors: {neighbor_count}", f"Listener mean: {baseline:.2f}"]
            else:
                result["items"] = _format_recommendations(scores, catalog, reasons={k: "Predicted rating from neighboring items" for k in scores})
                result["message"] = "Rate items under this profile and across other profiles to calculate predictions."
    elif method == "knowledge-based":
        selected = set(options.get("preferences", []))
        weights = {"genre": float(options.get("genre_weight", 1)), "mood/theme": float(options.get("mood_weight", 1)), "instrument": float(options.get("instrument_weight", 1))}
        scores = {}
        for item in catalog:
            score = sum(weights.get(tag.split("---", 1)[0], 0) for tag in selected if tag in item["tag_set"])
            if score:
                scores[item["track_id"]] = score
        result["items"] = _format_recommendations(scores, catalog, reasons={k: "Weighted preference score" for k in scores})
        result["metrics"] = [f"Genre weight: {weights['genre']:.2f}", f"Mood weight: {weights['mood/theme']:.2f}", f"Instrument weight: {weights['instrument']:.2f}"]
        if not selected:
            result["message"] = "Select preference tags and adjust the category weights."
    return result


def preference_tags(catalog):
    return {
        "genre": sorted({tag for item in catalog for tag in item["tags"] if tag.startswith("genre---")}),
        "mood/theme": sorted({tag for item in catalog for tag in item["tags"] if tag.startswith("mood/theme---")}),
        "instrument": sorted({tag for item in catalog for tag in item["tags"] if tag.startswith("instrument---")}),
    }
