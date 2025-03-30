import json
import random
import re
import numpy as np
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def extract_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

best_xgb_params = {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 4}

with open("train_data_classified.json", "r", encoding="utf8") as f:
    train_data = json.load(f)

def prepare_data(sc_dict, positive_key, negative_keys):
    texts, labels = [], []
    for key in negative_keys:
        texts.extend(sc_dict.get(key, []))
        labels.extend([0] * len(sc_dict.get(key, [])))
    texts.extend(sc_dict.get(positive_key, []))
    labels.extend([1] * len(sc_dict.get(positive_key, [])))
    return texts, labels

texts_mal, labels_mal = prepare_data(train_data, "maladaptive-state", ["adaptive-state", "neither-state"])
combined_mal = list(zip(texts_mal, labels_mal))
random.shuffle(combined_mal)
texts_mal, labels_mal = zip(*combined_mal)

vectorizer_mal = TfidfVectorizer()
X_mal = vectorizer_mal.fit_transform(texts_mal)

xgb_model_mal = XGBClassifier(**best_xgb_params,
                              objective='binary:logistic',
                              use_label_encoder=False,
                              eval_metric='logloss',
                              random_state=42)
xgb_model_mal.fit(X_mal, labels_mal)

texts_adapt, labels_adapt = prepare_data(train_data, "adaptive-state", ["maladaptive-state", "neither-state"])
combined_adapt = list(zip(texts_adapt, labels_adapt))
random.shuffle(combined_adapt)
texts_adapt, labels_adapt = zip(*combined_adapt)

vectorizer_adapt = TfidfVectorizer()
X_adapt = vectorizer_adapt.fit_transform(texts_adapt)

lr_model_adapt = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
lr_model_adapt.fit(X_adapt, labels_adapt)

with open("test_predict.json", "r", encoding="utf8") as infile:
    pred_timelines = json.load(infile)

noise_std = 1e-3

for timeline in pred_timelines:
    timeline["post_level"] = {}
    
    for post in timeline.get("posts", []):
        original_post = post.get("post", "")
        post["wellbeing_score"] = 1
        adaptive_evidence = []
        maladaptive_evidence = []
        sentences = extract_sentences(original_post)
        
        for sentence in sentences:
            vec_adapt = vectorizer_adapt.transform([sentence]).toarray()
            preds_adapt = [lr_model_adapt.predict(vec_adapt + np.random.normal(0, noise_std, vec_adapt.shape))[0] for _ in range(50)]
            if np.argmax(np.bincount(preds_adapt)) == 1:
                adaptive_evidence.append(sentence)
            
            vec_mal = vectorizer_mal.transform([sentence]).toarray()
            preds_mal = [xgb_model_mal.predict(vec_mal + np.random.normal(0, noise_std, vec_mal.shape))[0] for _ in range(100)]
            if np.argmax(np.bincount(preds_mal)) == 1:
                maladaptive_evidence.append(sentence)
        
        post["adaptive_evidence"] = adaptive_evidence
        post["maladaptive_evidence"] = maladaptive_evidence
        
        post_id = post.get("post_id")
        if post_id:
            timeline["post_level"][post_id] = {"summary": ""}

submission = {timeline.get("timeline_id"): timeline for timeline in pred_timelines if timeline.get("timeline_id")}

with open("test_submission.json", "w", encoding="utf8") as outfile:
    json.dump(submission, outfile, ensure_ascii=False, indent=2)

