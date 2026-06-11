"""
cyber_app/backend/phishing_app.py
Phishing URL detection — adapted from fake_url/app1.py (Flask → FastAPI).
Model files are loaded once at startup from models/ directory.
"""

from fastapi import APIRouter
from pydantic import BaseModel
import os, math, unicodedata
import urllib.parse
import ipaddress
from collections import Counter

import joblib
import pandas as pd
import tldextract

router = APIRouter()

# ── Model paths ───────────────────────────────────────────────────────────────
_BASE = os.path.dirname(__file__)
MODEL_PATH  = os.path.join(_BASE, "models", "model.joblib")
SCALER_PATH = os.path.join(_BASE, "models", "scaler.joblib")

_model  = None
_scaler = None

def _load():
    global _model, _scaler
    if _model is None:
        _model  = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)

# ── Domain knowledge ──────────────────────────────────────────────────────────
SUSPICIOUS_TLDS = {
    'xyz','tk','ml','ga','cf','gq','pw','top','club','online',
    'site','live','stream','download','win','loan','click','buzz',
    'rest','casa','icu','vip','work','fun','space','cyou'
}

BRAND_KEYWORDS = [
    'paypal','amazon','google','facebook','apple','microsoft',
    'netflix','instagram','whatsapp','bank','secure','login',
    'verify','account','update','confirm','signin','ebay',
    'wellsfargo','chase','citibank','hdfc','icici','sbi'
]

TRUSTED_DOMAINS = {
    'google','paypal','amazon','apple','microsoft','facebook',
    'netflix','instagram','ebay','chase','wellsfargo','citibank',
    'hdfc','icici','sbi','youtube','twitter','linkedin','github'
}

GOV_KEYWORDS = [
    'centralexcise','excise','incometax','income','govtindia',
    'gov','government','ministry','customs','police','uidai',
    'aadhar','aadhaar','passport','railway','irctc','epfo',
    'nsdl','tds','cbdt','cbic','nagarpalika','municipal',
    'panchayat','mantralaya','tribunal','highcourt','supremecourt',
    'rto','postal','servicetax','gst','gstn','esic','sebi',
    'rbi','drdo','isro','bsnl','ration','licence',
    'nios','cbse','icse','ugc','aicte','ncert','neet','upsc',
    'ssc','rrb','ibps','lic','npci','digilocker','cowin',
    'umang','mygov','digitalindia','ndma','nha','mohfw'
]

TRUSTED_GOV_SUFFIXES  = {'gov.in','nic.in','gov','mil'}
EDU_GOV_KEYWORDS      = ['nios','cbse','upsc','ssc','railway','passport','neet','ugc','aicte','ncert']
TRUSTED_EDU_DOMAINS   = {'nios.ac.in','cbse.gov.in','upsc.gov.in','ssc.nic.in'}

FEATURE_ORDER = [
    'url_len','dom_len','is_ip','tld_len','subdom_cnt',
    'letter_cnt','digit_cnt','special_cnt','eq_cnt','qm_cnt',
    'amp_cnt','dot_cnt','dash_cnt','under_cnt','letter_ratio',
    'digit_ratio','spec_ratio','is_https','slash_cnt','entropy',
    'path_len','query_len','keyword_mismatch'
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def _clean(raw: str) -> str:
    raw = unicodedata.normalize('NFKC', raw)
    raw = ''.join(c for c in raw if c.isprintable() and ord(c) < 128)
    return raw.lstrip('. \t').strip()

def _kw_mismatch(dom, tld, subdomain="", path="") -> int:
    trusted = ['gov.in','nic.in','ac.in','edu.in']
    raw = f"{subdomain} {dom} {path}".lower().replace('-','').replace('.','')
    for kw in EDU_GOV_KEYWORDS:
        if kw in raw and tld not in trusted:
            return 1
    return 0

def _extract(url: str) -> dict:
    parsed   = urllib.parse.urlparse(url)
    url_len  = len(url)
    hostname = parsed.hostname or ""

    if not hostname and not url.startswith(("http://","https://")):
        parsed   = urllib.parse.urlparse("http://" + url)
        hostname = parsed.hostname or ""

    is_ip = 0
    try:
        ipaddress.ip_address(hostname); is_ip = 1
    except ValueError:
        pass

    dom_len = tld_len = subdom_cnt = 0
    ext = None; tld = ""
    if hostname and not is_ip:
        ext        = tldextract.extract(url)
        dom_str    = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
        tld        = ext.suffix or ""
        tld_len    = len(tld)
        dom_len    = len(dom_str)
        subdom_cnt = len(ext.subdomain.split('.')) if ext.subdomain else 0
    elif is_ip:
        dom_len = len(hostname)

    lc  = sum(c.isalpha()  for c in url)
    dc  = sum(c.isdigit()  for c in url)
    sc  = url_len - lc - dc
    entropy = 0.0
    if url_len:
        for cnt in Counter(url).values():
            p = cnt / url_len
            entropy -= p * math.log2(p)

    kw_flag = _kw_mismatch(ext.domain, tld, ext.subdomain, parsed.path) if ext else 0

    return {
        'url_len': url_len, 'dom_len': dom_len, 'is_ip': is_ip,
        'tld_len': tld_len, 'subdom_cnt': subdom_cnt,
        'letter_cnt': lc, 'digit_cnt': dc, 'special_cnt': sc,
        'eq_cnt': url.count('='), 'qm_cnt': url.count('?'),
        'amp_cnt': url.count('&'), 'dot_cnt': url.count('.'),
        'dash_cnt': url.count('-'), 'under_cnt': url.count('_'),
        'letter_ratio': lc/url_len if url_len else 0,
        'digit_ratio':  dc/url_len if url_len else 0,
        'spec_ratio':   sc/url_len if url_len else 0,
        'is_https': 1 if parsed.scheme == 'https' else 0,
        'slash_cnt': url.count('/'), 'entropy': entropy,
        'path_len':  len(parsed.path)  if parsed.path  else 0,
        'query_len': len(parsed.query) if parsed.query else 0,
        'keyword_mismatch': kw_flag,
    }

# ── Request / response models ─────────────────────────────────────────────────
class URLRequest(BaseModel):
    url: str

# ── Route ─────────────────────────────────────────────────────────────────────
@router.post("/predict-url")
async def predict_url(req: URLRequest):
    _load()

    url = _clean(req.url)
    if not url:
        return {"success": False, "error": "URL cannot be empty"}

    proc = url if url.startswith(("http://","https://")) else "http://" + url

    features  = _extract(proc)
    parsed    = urllib.parse.urlparse(proc)
    ext       = tldextract.extract(proc)

    domain_root   = ext.domain.lower()
    subdomain_str = ext.subdomain.lower()
    path_str      = parsed.path.lower()
    full_suffix   = ext.suffix.lower()

    df      = pd.DataFrame([[features[c] for c in FEATURE_ORDER]], columns=FEATURE_ORDER)
    scaled  = _scaler.transform(df)
    pred    = int(_model.predict(scaled)[0])
    proba   = _model.predict_proba(scaled)[0]
    risk    = float(proba[1])

    explanations = []
    if features['is_ip']       == 1: explanations.append("Domain is an IP address — highly typical of malicious links.")
    if features['is_https']    == 0: explanations.append("Link does not use secure HTTPS protocol.")
    if features['url_len']      > 75: explanations.append("URL is unusually long — often used to obscure the real domain.")
    if features['subdom_cnt']  >= 3:  explanations.append("Many subdomains detected — characteristic of phishing infrastructure.")
    if features['special_cnt'] > 15:  explanations.append("High special-character count in URL.")
    if features['entropy']      > 4.5: explanations.append("High character entropy — URL looks randomly generated.")

    # Hard override rules (same logic as original app1.py)
    if features['is_ip']:
        pred = 1; risk = max(risk, 0.95)
        explanations.append("RULE 1: IP address as domain — strong phishing indicator.")

    if full_suffix in SUSPICIOUS_TLDS:
        pred = 1; risk = max(risk, 0.85)
        explanations.append("RULE 2: High-risk TLD commonly used in phishing.")

    if domain_root not in TRUSTED_DOMAINS:
        check = subdomain_str + " " + domain_root + " " + path_str
        for brand in BRAND_KEYWORDS:
            if brand in check:
                pred = 1; risk = max(risk, 0.90)
                explanations.append("RULE 3: Brand keyword outside trusted domain — possible impersonation.")
                break

    if features['subdom_cnt'] >= 3 and features['is_https'] == 0:
        pred = 1; risk = max(risk, 0.80)
        explanations.append("RULE 4: Multiple subdomains with no HTTPS — strong phishing pattern.")

    if features['dash_cnt'] >= 4:
        pred = 1; risk = max(risk, 0.75)
        explanations.append("RULE 5: Excessive dashes — common in typosquatted domains.")

    if full_suffix not in TRUSTED_GOV_SUFFIXES:
        raw_dom = (subdomain_str + domain_root).replace('-','').replace('.','')
        for kw in GOV_KEYWORDS:
            if kw in raw_dom:
                pred = 1; risk = max(risk, 0.85)
                explanations.append("RULE 6: Government keyword found but NOT on .gov.in/.nic.in — likely impersonating an official site.")
                break

    full_dom = f"{domain_root}.{full_suffix}"
    raw_chk  = (subdomain_str + domain_root + path_str).replace('-','').replace('.','').replace('/','')
    for kw in EDU_GOV_KEYWORDS:
        if kw in raw_chk and full_dom not in TRUSTED_EDU_DOMAINS:
            pred = 1; risk = max(risk, 0.85)
            explanations.append("RULE 7: Educational keyword in non-official domain — possible impersonation.")
            break

    if features['keyword_mismatch']:
        pred = 1; risk = max(risk, 0.85)
        explanations.append("RULE 8: Government/educational keyword mismatch — strong phishing indicator.")

    # De-duplicate explanations
    seen, unique_exp = set(), []
    for e in explanations:
        if e not in seen:
            seen.add(e); unique_exp.append(e)

    return {
        "success":      True,
        "url":          url,
        "label":        "Phishing (Fraudulent)" if pred == 1 else "Legitimate",
        "is_phishing":  pred == 1,
        "risk_score":   round(risk * 100, 2),
        "features":     features,
        "explanations": unique_exp,
    }
