"""NLP Pipeline & Bayesian Text Classification Module.
Performs:
1. Text paragraph extraction
2. Sentence segmentation (separating sentences)
3. Tokenization & stop words removal
4. Stemming (Porter) & Lemmatization (WordNet)
5. Bayesian text classification (Multinomial Naive Bayes) with posterior estimation
"""

import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Ensure required NLTK resources
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()

# ---------------------------------------------------------------------------
# Training & Benchmark Test Corpus (3 Distinct Topics)
# ---------------------------------------------------------------------------
TRAIN_CORPUS = [
    # Technology & Computing
    ("Neural networks and deep learning models process large data tensors on GPU accelerators.", "Technology"),
    ("Quantum computing chips and algorithmic architectures revolutionize machine intelligence.", "Technology"),
    ("Software developers compile microservices into containerized cloud Kubernetes clusters.", "Technology"),
    ("Artificial intelligence agents optimize code bases and automate complex development tasks.", "Technology"),
    ("Distributed databases replicate transactional records across high-availability network servers.", "Technology"),
    
    # Medical & Healthcare
    ("Clinical trials evaluate vaccine efficacy and immune antibody responses in human patients.", "Medical"),
    ("Cardiovascular therapies lower arterial blood pressure and prevent acute cardiac infarction.", "Medical"),
    ("Physicians diagnose diabetic retinopathy lesions and prescribe therapeutic insulin regimens.", "Medical"),
    ("Neurological scans reveal neural synaptic degeneration in chronic dementia patients.", "Medical"),
    ("Oncologists utilize radiation therapy and targeted chemotherapy to inhibit malignant tumor cells.", "Medical"),
    
    # Climate & Environmental Science
    ("Atmospheric greenhouse emissions accelerate Arctic glacier melting and oceanic warming.", "Climate"),
    ("Renewable solar panels and wind turbine grids reduce carbon atmospheric emissions.", "Climate"),
    ("Tropical rainforest deforestation threatens biodiversity and regional rainfall patterns.", "Climate"),
    ("Ocean acidification impacts coral reef ecosystems and marine marine organisms.", "Climate"),
    ("Extreme weather droughts deplete freshwater aquifers and disrupt agricultural food security.", "Climate"),
]

TEST_CORPUS = [
    {
        "id": "TEST-01",
        "text": "Autonomous AI processors execute machine learning models with high efficiency.",
        "true_class": "Technology"
    },
    {
        "id": "TEST-02",
        "text": "Cloud engineers deploy database clusters across distributed server networks.",
        "true_class": "Technology"
    },
    {
        "id": "TEST-03",
        "text": "Hospitals monitor patient heart rate and administer pharmaceutical antibiotic therapies.",
        "true_class": "Medical"
    },
    {
        "id": "TEST-04",
        "text": "Retinal fundus scans assist doctors in detecting microvascular edema and lesions.",
        "true_class": "Medical"
    },
    {
        "id": "TEST-05",
        "text": "Rising global temperatures accelerate Arctic ice sheet collapse and coastal sea level surges.",
        "true_class": "Climate"
    },
    {
        "id": "TEST-06",
        "text": "Reforestation projects restore soil nutrients and sequester carbon dioxide emissions.",
        "true_class": "Climate"
    },
    {
        "id": "TEST-07",
        "text": "Cybersecurity algorithms detect malicious ransomware intrusion within corporate networks.",
        "true_class": "Technology"
    },
    {
        "id": "TEST-08",
        "text": "Surgical procedures repair fractured knee ligaments and restore musculoskeletal mobility.",
        "true_class": "Medical"
    }
]

CLASSES = ["Technology", "Medical", "Climate"]
CLASS_COLORS = {
    "Technology": "#1e3a8a",
    "Medical": "#164e3d",
    "Climate": "#b45309"
}

# ---------------------------------------------------------------------------
# Core NLP Transformation Functions
# ---------------------------------------------------------------------------
def extract_sentences(paragraph: str) -> list[str]:
    """Segment raw input paragraph into individual sentences."""
    if not paragraph.strip():
        return []
    try:
        return sent_tokenize(paragraph.strip())
    except Exception:
        # Fallback regex segmentation
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraph.strip()) if s.strip()]


def process_sentence_tokens(sentence: str):
    """Tokenize sentence, isolate removed stopwords, and perform Stemming vs Lemmatization."""
    try:
        raw_tokens = word_tokenize(sentence)
    except Exception:
        raw_tokens = re.findall(r"\b\w+\b", sentence)

    tokens_clean = []
    removed_stops = []
    comparison_table = []

    for token in raw_tokens:
        word_lower = token.lower()
        if not token.isalnum():
            continue
        if word_lower in STOP_WORDS:
            removed_stops.append(token)
        else:
            tokens_clean.append(token)
            stemmed = STEMMER.stem(word_lower)
            lemmatized = LEMMATIZER.lemmatize(word_lower)
            comparison_table.append({
                "original": token,
                "stemmed": stemmed,
                "lemmatized": lemmatized
            })

    return {
        "raw_tokens": raw_tokens,
        "removed_stopwords": removed_stops,
        "clean_tokens": tokens_clean,
        "comparison_table": comparison_table
    }


def full_nlp_pipeline(paragraph: str):
    """Execute the complete NLP extraction and processing pipeline on a paragraph."""
    sentences = extract_sentences(paragraph)
    sentence_results = []
    all_clean_tokens = []
    all_stemmed = []
    all_lemmatized = []
    all_stops = []

    for idx, s in enumerate(sentences):
        res = process_sentence_tokens(s)
        sentence_results.append({
            "index": idx + 1,
            "text": s,
            **res
        })
        all_clean_tokens.extend(res["clean_tokens"])
        all_stops.extend(res["removed_stopwords"])
        for comp in res["comparison_table"]:
            all_stemmed.append(comp["stemmed"])
            all_lemmatized.append(comp["lemmatized"])

    return {
        "raw_paragraph": paragraph,
        "sentences_count": len(sentences),
        "sentences": sentence_results,
        "total_raw_tokens": sum(len(s["raw_tokens"]) for s in sentence_results),
        "total_stopwords_removed": len(all_stops),
        "total_clean_tokens": len(all_clean_tokens),
        "lemmatized_text": " ".join(all_lemmatized),
        "stemmed_text": " ".join(all_stemmed)
    }


# ---------------------------------------------------------------------------
# Bayesian Model Training & Evaluation
# ---------------------------------------------------------------------------
def get_trained_bayesian_model():
    """Train Multinomial Naive Bayes classifier on training corpus."""
    def clean_text(t):
        tokens = word_tokenize(t.lower())
        words = [LEMMATIZER.lemmatize(w) for w in tokens if w.isalnum() and w not in STOP_WORDS]
        return " ".join(words)

    train_texts = [clean_text(d[0]) for d in TRAIN_CORPUS]
    train_labels = [d[1] for d in TRAIN_CORPUS]

    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(train_texts)

    model = MultinomialNB(alpha=1.0)
    model.fit(X_train_vec, train_labels)

    return model, vectorizer


def classify_text_bayes(text: str):
    """Classify text using trained Bayesian model and return posterior probabilities."""
    model, vectorizer = get_trained_bayesian_model()
    nlp_res = full_nlp_pipeline(text)
    proc_text = nlp_res["lemmatized_text"]

    if not proc_text.strip():
        return {
            "predicted_class": "Technology",
            "confidence": 0.333,
            "posteriors": {c: 0.333 for c in CLASSES}
        }

    vec = vectorizer.transform([proc_text])
    probs = model.predict_proba(vec)[0]
    classes = list(model.classes_)
    posteriors = {c: float(probs[classes.index(c)]) for c in CLASSES}
    pred_class = max(posteriors, key=posteriors.get)

    return {
        "predicted_class": pred_class,
        "confidence": posteriors[pred_class],
        "posteriors": posteriors
    }


def evaluate_test_corpus():
    """Evaluate benchmark test corpus using the Bayesian classifier."""
    results = []
    y_true = []
    y_pred = []

    for item in TEST_CORPUS:
        clf_res = classify_text_bayes(item["text"])
        nlp_res = full_nlp_pipeline(item["text"])
        pred_c = clf_res["predicted_class"]
        is_correct = (pred_c == item["true_class"])
        y_true.append(item["true_class"])
        y_pred.append(pred_c)

        results.append({
            "id": item["id"],
            "text": item["text"],
            "true_class": item["true_class"],
            "pred_class": pred_c,
            "confidence": clf_res["confidence"],
            "posteriors": clf_res["posteriors"],
            "is_correct": is_correct,
            "sentences_count": nlp_res["sentences_count"],
            "clean_tokens": nlp_res["total_clean_tokens"],
            "stops_removed": nlp_res["total_stopwords_removed"],
        })

    acc = accuracy_score(y_true, y_pred)
    return results, acc


# ---------------------------------------------------------------------------
# Python Source Code for Page Display
# ---------------------------------------------------------------------------
PYTHON_NLP_BAYES_CODE = '''import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# 1. Download required NLTK resources
for pkg in ["punkt", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

# 2. Extract paragraph and separate sentences
paragraph = """
Artificial intelligence algorithms optimize complex neural networks on distributed cloud servers. 
Physicians in clinical hospitals evaluate diabetic macular edema and arterial blood pressure. 
Meanwhile, atmospheric carbon emissions accelerate Arctic glacier melting across polar regions.
"""

sentences = sent_tokenize(paragraph.strip())
print(f"Extracted {len(sentences)} Sentences:")
for i, s in enumerate(sentences, 1):
    print(f"  [{i}] {s}")

# 3. Stop words removal, Stemming & Lemmatization
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

processed_sentences = []
for s in sentences:
    tokens = word_tokenize(s)
    clean_tokens = [t for t in tokens if t.isalnum() and t.lower() not in stop_words]
    stemmed = [stemmer.stem(t.lower()) for t in clean_tokens]
    lemmatized = [lemmatizer.lemmatize(t.lower()) for t in clean_tokens]
    processed_sentences.append(" ".join(lemmatized))
    print(f"\\nOriginal:   {s}")
    print(f"Cleaned:    {clean_tokens}")
    print(f"Stemmed:    {stemmed}")
    print(f"Lemmatized: {lemmatized}")

# 4. Bayesian Classification (Multinomial Naive Bayes)
# Training domain corpus
train_docs = [
    ("Neural networks and deep learning models process large data tensors on GPU accelerators.", "Technology"),
    ("Quantum computing chips and algorithmic architectures revolutionize machine intelligence.", "Technology"),
    ("Clinical trials evaluate vaccine efficacy and immune antibody responses in patients.", "Medical"),
    ("Cardiovascular therapies lower arterial blood pressure and prevent cardiac infarction.", "Medical"),
    ("Atmospheric greenhouse emissions accelerate Arctic glacier melting and oceanic warming.", "Climate"),
    ("Renewable solar panels and wind turbine grids reduce carbon atmospheric emissions.", "Climate"),
]

X_train_text = [" ".join([lemmatizer.lemmatize(w.lower()) for w in word_tokenize(d[0]) if w.isalnum() and w.lower() not in stop_words]) for d in train_docs]
y_train = [d[1] for d in train_docs]

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train_text)

clf = MultinomialNB(alpha=1.0)
clf.fit(X_train_vec, y_train)

# 5. Classify unseen test samples using Bayesian Maximum A Posteriori (MAP)
test_samples = [
    ("Autonomous AI processors execute machine learning models with high efficiency.", "Technology"),
    ("Hospitals monitor patient heart rate and administer pharmaceutical antibiotic therapies.", "Medical"),
    ("Rising global temperatures accelerate Arctic ice sheet collapse and coastal sea level surges.", "Climate")
]

for text, true_label in test_samples:
    clean_t = " ".join([lemmatizer.lemmatize(w.lower()) for w in word_tokenize(text) if w.isalnum() and w.lower() not in stop_words])
    vec = vectorizer.transform([clean_t])
    probs = clf.predict_proba(vec)[0]
    pred = clf.predict(vec)[0]
    
    print(f"\\nTest Sentence: '{text}'")
    print(f"True Class: {true_label} | MAP Predicted: {pred}")
    for c, p in zip(clf.classes_, probs):
        print(f"  P({c}|text) = {p:.4f}")
'''
