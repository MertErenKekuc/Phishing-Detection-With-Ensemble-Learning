import os
import re
import math
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from urllib.parse import urlparse

app = Flask(__name__)

MODEL_PATH = "models/phishing_model.pkl"
SCALER_PATH = "models/scaler.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("models/phishing_model.pkl veya models/scaler.pkl bulunamadı.")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# THRESHOLDLAR
RISK_THRESH = (0.80, 0.60, 0.40)  # Very high / High / Medium
LEGIT_THRESHOLD = 0.55 

# Consensus parametreleri
CONSENSUS_WEIGHT_PROB = 0.80
CONSENSUS_WEIGHT_PRED = 0.25
CONSENSUS_THRESHOLD = 0.55
USE_CONSENSUS_AS_LABEL = False

SHORT_DOMAINS = {"bit.ly","tinyurl.com","goo.gl","t.co","ow.ly","is.gd","buff.ly","adf.ly"}
BAD_TLDS = {".top", ".tk", ".xyz", ".club", ".info", ".pw"}

def normalize_url_for_analysis(url: str) -> str:
    u = url.strip()
    if not re.match(r'^[a-zA-Z]+://', u):
        u = "http://" + u
    return u

def _is_ip(domain: str) -> bool:
    return bool(re.match(r'^\d+\.\d+\.\d+\.\d+$', domain))

def _sigmoid(x):
    try:
        return 1.0 / (1.0 + math.exp(-float(x)))
    except Exception:
        return None

def _get_phishing_score_from_model(X_scaled):
    # predict_proba öncelik
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_scaled)
            return float(probs[0][1])
    except Exception:
        pass

    # decision_function -> sigmoid
    try:
        if hasattr(model, "decision_function"):
            df = model.decision_function(X_scaled)
            val = df[0] if hasattr(df, "__len__") else df
            return _sigmoid(val)
    except Exception:
        pass

    # Voting/ensemble: base estimatorların ortalaması
    try:
        if hasattr(model, "estimators_"):
            scores = []
            for est in model.estimators_:
                try:
                    if hasattr(est, "predict_proba"):
                        scores.append(float(est.predict_proba(X_scaled)[0][1]))
                    elif hasattr(est, "decision_function"):
                        df = est.decision_function(X_scaled)
                        val = df[0] if hasattr(df, "__len__") else df
                        scores.append(_sigmoid(val))
                except Exception:
                    continue
            scores = [s for s in scores if s is not None]
            if scores:
                return float(sum(scores) / len(scores))
    except Exception:
        pass

    return None

def build_features_from_names(url: str, feature_names):
    u = normalize_url_for_analysis(url)
    p = urlparse(u)
    domain = p.netloc.lower().split(':')[0]
    full = u.lower()

    url_len = len(full)
    sub_count = domain.count('.')
    has_https = (p.scheme or "").lower() == "https"
    ip_present = _is_ip(domain)

    tld = ""
    if "." in domain:
        tld = "." + domain.split(".")[-1]

    feats = {}
    for name in feature_names:
        if name == "index":
            feats[name] = float(url_len)
            continue

        if name == "having_IPhaving_IP_Address":
            feats[name] = 1.0 if ip_present else -1.0
            continue

        if name == "URLURL_Length":
            if url_len < 54:
                feats[name] = -1.0
            elif url_len <= 75:
                feats[name] = 0.0
            else:
                feats[name] = 1.0
            continue

        if name == "Shortining_Service":
            feats[name] = 1.0 if any(sd in domain for sd in SHORT_DOMAINS) else -1.0
            continue

        if name == "having_At_Symbol":
            feats[name] = 1.0 if "@" in full else -1.0
            continue

        if name == "double_slash_redirecting":
            feats[name] = 1.0 if full.count("//") > 1 else -1.0
            continue

        if name == "Prefix_Suffix":
            feats[name] = 1.0 if "-" in domain else -1.0
            continue

        if name == "having_Sub_Domain":
            if sub_count <= 1:
                feats[name] = -1.0
            elif sub_count == 2:
                feats[name] = 0.0
            else:
                feats[name] = 1.0
            continue

        if name == "SSLfinal_State":
            if tld in BAD_TLDS or ("https" in domain) or (not has_https):
                feats[name] = 1.0
            else:
                feats[name] = 0.0
            continue

        if name == "HTTPS_token":
            feats[name] = 1.0 if "https" in domain else -1.0
            continue

        # Port kontrolü
        if name == "port":
            port_parts = p.netloc.split(':')
            if len(port_parts) > 1 and port_parts[1]:
                feats[name] = 1.0  # Custom port
            else:
                feats[name] = -1.0  # Default
            continue

        # Email kontrolü
        if name == "Submitting_to_email":
            email_kw = ["mailto:", "email=", "submit="]
            feats[name] = 1.0 if any(kw in full for kw in email_kw) else -1.0
            continue

        if name == "Redirect":
            redirect_kw = ["redirect=", "goto=", "url=", "next=", "return=", "continue=", "dest="]
            feats[name] = 1.0 if any(kw in full for kw in redirect_kw) else -1.0
            continue

        if name == "Abnormal_URL":
            brand_kw = ["paypal", "amazon", "google", "facebook", "microsoft", "apple", 
                       "netflix", "instagram", "twitter", "linkedin", "ebay"]
            has_brand = any(brand in domain for brand in brand_kw)
            is_suspicious = (tld in BAD_TLDS and has_brand) or (has_brand and sub_count > 2)
            feats[name] = 1.0 if is_suspicious else 0.0
            continue

        if name == "popUpWidnow":
            popup_kw = ["popup=", "window.open", "popunder", "pop=", "modal="]
            feats[name] = 1.0 if any(kw in full for kw in popup_kw) else -1.0
            continue

        if name == "Iframe":
            iframe_kw = ["iframe=", "frame=", "embed="]
            feats[name] = 1.0 if any(kw in full for kw in iframe_kw) else -1.0
            continue

        if name in ("on_mouseover", "RightClick"):
            script_kw = ["script=", "onload=", "onerror=", "onclick=", "onmouseover=", 
                        "oncontextmenu=", "eval(", "javascript:"]
            feats[name] = 1.0 if any(kw in full for kw in script_kw) else -1.0
            continue

        # Pasif
        if name in (
            "Request_URL","URL_of_Anchor","Links_in_tags","SFH",
            "Domain_registeration_length","Favicon","age_of_domain","DNSRecord",
            "web_traffic","Page_Rank","Google_Index","Links_pointing_to_page",
            "Statistical_report"
        ):
            if tld in BAD_TLDS and name in ("Statistical_report", "Request_URL"):
                feats[name] = 1.0
            else:
                feats[name] = 0.0
            continue

        feats[name] = 0.0

    return [float(feats[n]) for n in feature_names]

def adjust_features(feats, required_len):
    if required_len is None:
        return feats
    if len(feats) < required_len:
        feats = feats + [0.0] * (required_len - len(feats))
    elif len(feats) > required_len:
        feats = feats[:required_len]
    return feats

def predict_from_url(url: str):
    feature_names = getattr(scaler, "feature_names_in_", None)
    n_required = getattr(scaler, "n_features_in_", None)
    if n_required is None and hasattr(scaler, "scale_"):
        n_required = scaler.scale_.shape[0]

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_required or 17)]

    feats = build_features_from_names(url, list(feature_names))
    feats = adjust_features(feats, n_required)

    if getattr(scaler, "feature_names_in_", None) is not None:
        X_in = pd.DataFrame([{name: feats[i] for i, name in enumerate(feature_names)}], columns=feature_names)
    else:
        X_in = np.array([feats], dtype=float)

    X_scaled = scaler.transform(X_in)

    # Tahmin
    try:
        raw_pred = model.predict(X_scaled)[0]
        pred = int(raw_pred)
    except Exception:
        pred = None

    phishing_score = _get_phishing_score_from_model(X_scaled)
    
    original_score = phishing_score
    boost_reasons = []
    if phishing_score is not None:
        phishing_score, boost_reasons = apply_subtle_phishing_boost(url, phishing_score)
    
    if phishing_score is None:
        phishing_pct = None
        probs_short = None
    else:
        phishing_pct = f"{(phishing_score * 100):.1f}%"
        probs_short = [f"Phishing: {(phishing_score*100):.1f}%", f"Legit: {((1-phishing_score)*100):.1f}%"]

    consensus_score = None
    if phishing_score is not None and pred is not None:
        consensus_score = phishing_score * CONSENSUS_WEIGHT_PROB + float(pred) * CONSENSUS_WEIGHT_PRED
    elif phishing_score is not None:
        consensus_score = phishing_score
    elif pred is not None:
        consensus_score = float(pred)

    consensus_label = "Phishing" if (consensus_score and consensus_score >= CONSENSUS_THRESHOLD) else "Legitimate"

    if phishing_score is not None:
        label = "Phishing" if phishing_score >= LEGIT_THRESHOLD else "Legitimate"
    elif pred is not None:
        label = "Phishing" if pred == 1 else "Legitimate"
    else:
        label = "Unknown"

    # Risk seviyesi
    risk = "Low"
    if phishing_score is not None:
        if phishing_score >= RISK_THRESH[0]:
            risk = "Very high"
        elif phishing_score >= RISK_THRESH[1]:
            risk = "High"
        elif phishing_score >= RISK_THRESH[2]:
            risk = "Medium"
    else:
        risk = "High" if pred == 1 else "Low"

    # Renk
    if phishing_score is not None:
        if phishing_score >= RISK_THRESH[1]:
            visual_color = "red"
        elif phishing_score >= RISK_THRESH[2]:
            visual_color = "yellow"
        else:
            visual_color = "green"
    else:
        visual_color = "red" if pred == 1 else "green"

    out = {
        "prediction": int(pred) if pred is not None else None,
        "label": label,
        "consensus_label": consensus_label,
        "consensus_score": float(consensus_score) if consensus_score is not None else None,
        "phishing_probability": float(phishing_score) if phishing_score is not None else None,
        "phishing_probability_pct": phishing_pct,
        "probabilities_short": probs_short,
        "risk_level": risk,
        "visual_color": visual_color,
        "features_used": len(feats),
        "expected_features": n_required
    }
    
    return out

def apply_subtle_phishing_boost(url: str, phishing_score: float) -> tuple:
    if phishing_score is None:
        return None, []
    
    u = normalize_url_for_analysis(url)
    p = urlparse(u)
    domain = p.netloc.lower().split(':')[0]
    full = u.lower()
    path = p.path or ""
    
    boost = 0.0
    reasons = []
    
    # Temel URL özelliklerini her zaman göster
    url_len = len(full)
    sub_count = domain.count('.')
    
    # Protocol kontrolü
    if p.scheme == "https":
        reasons.append(f"HTTPS_Protocol")
    else:
        reasons.append(f"HTTP_Protocol")
    
    # Domain uzunluğu
    if len(domain) > 30:
        reasons.append(f"Long_Domain({len(domain)}chars)")
    elif len(domain) < 10:
        reasons.append(f"Short_Domain({len(domain)}chars)")
    
    # Subdomain sayısı
    if sub_count > 2:
        reasons.append(f"Subdomains({sub_count})")
    
    # URL Shortener → +0.18 boost
    if any(sd in domain for sd in SHORT_DOMAINS):
        boost += 0.18
        reasons.append("URL_Shortener")
    
    # Kötü TLD → Base +0.12 boost
    tld = ""
    if "." in domain:
        tld = "." + domain.split(".")[-1]
    
    if tld in BAD_TLDS:
        if p.scheme == "https":
            boost += 0.08  # HTTPS ile kötü TLD → +0.08
            reasons.append(f"Bad_TLD{tld}_HTTPS")
        else:
            boost += 0.15  # HTTP ile kötü TLD → +0.15
            reasons.append(f"Bad_TLD{tld}_HTTP")
    
    # @ Sembolü → +0.25 boost
    if "@" in full:
        boost += 0.25
        reasons.append("At_Symbol")
    
    # IP Adresi → +0.22 boost
    if _is_ip(domain):
        boost += 0.22
        reasons.append("IP_Address")
    
    # HTTPS Token (domain'de "https") → +0.12 boost
    if "https" in domain and p.scheme != "https":
        boost += 0.12
        reasons.append("HTTPS_Token")
    
    # Çok Uzun URL (>120 karakter) → +0.10 boost
    url_len = len(full)
    if url_len > 120:
        boost += 0.10
        reasons.append(f"Very_Long_URL({url_len})")
    
    # HTTP + Login/Password Kelimeleri → +0.15 boost
    if p.scheme == "http":
        sensitive_kw = ["login", "signin", "password", "verify", "account", "banking", "secure"]
        if any(kw in full for kw in sensitive_kw):
            boost += 0.15
            reasons.append("HTTP_Sensitive")
    
    # Marka + Kötü TLD Kombinasyonu → +0.18 boost
    brand_kw = ["paypal", "amazon", "google", "facebook", "microsoft", "apple", 
                "netflix", "bank", "ebay", "instagram"]
    has_brand = any(brand in domain for brand in brand_kw)
    if has_brand and tld in BAD_TLDS:
        boost += 0.18
        reasons.append("Brand_BadTLD")
    
    # Çoklu Subdomain (>4) → +0.08 boost
    sub_count = domain.count('.')
    if sub_count > 4:
        boost += 0.08
        reasons.append(f"Many_Subdomains({sub_count})")
    
    # Çok Derin Path (>5 seviye) → +0.06 boost
    path_depth = len([p for p in path.split('/') if p])
    if path_depth > 5:
        boost += 0.06
        reasons.append(f"Deep_Path({path_depth})")
    
    # Şüpheli Path Tekrarı → +0.08 boost
    if path:
        path_parts = [p for p in path.split('/') if p]
        if len(path_parts) >= 3:
            # Aynı segment 3+ kez tekrar ediyorsa
            from collections import Counter
            counts = Counter(path_parts)
            max_repeat = max(counts.values()) if counts else 0
            if max_repeat >= 3:
                boost += 0.08
                reasons.append(f"Repeated_Path({max_repeat}x)")
    
    boosted_score = min(0.97, phishing_score + boost)
    
    return boosted_score, reasons

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    info = None
    url_value = ""
    if request.method == "POST":
        url_value = request.form.get("url", "").strip()
        if not url_value:
            result = "Hata: Lütfen bir URL girin."
        else:
            try:
                parsed = urlparse(normalize_url_for_analysis(url_value))
                if not parsed.netloc:
                    raise ValueError("Geçersiz URL.")
                info = predict_from_url(url_value)
                result = info["label"]
            except Exception as e:
                result = "Hata: " + str(e)
    return render_template("index.html", result=result, info=info, url_value=url_value)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.json or {}
    url = data.get("url") or data.get("u")
    if not url:
        return jsonify({"error": "url alanı gerekli"}), 400
    try:
        res = predict_from_url(url)
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/debug/features", methods=["POST"])
def debug_features():
    data = request.json or request.form
    url = data.get("url")
    if not url:
        return jsonify({"error":"url gerekli"}), 400
    feature_names = getattr(scaler, "feature_names_in_", None)
    if feature_names is None:
        return jsonify({"error":"scaler feature names yok"}), 500
    feats = build_features_from_names(url, list(feature_names))
    X_in = pd.DataFrame([{name: feats[i] for i,name in enumerate(feature_names)}], columns=feature_names)
    X_scaled = scaler.transform(X_in)

    try:
        pred = int(model.predict(X_scaled)[0])
    except Exception:
        pred = None

    phishing_score = _get_phishing_score_from_model(X_scaled)
    phishing_pct = f"{(phishing_score*100):.2f}%" if phishing_score is not None else None
    probs_short = [f"Phishing: {(phishing_score*100):.2f}%", f"Legit: {((1-phishing_score)*100):.2f}%"] if phishing_score is not None else None

    return jsonify({
        "raw_features": feats,
        "input_df_first_row": X_in.iloc[0].to_dict(),
        "scaled_first_row": X_scaled[0].tolist(),
        "prediction": pred,
        "phishing_probability": phishing_score,
        "phishing_probability_pct": phishing_pct,
        "probabilities_short": probs_short
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
