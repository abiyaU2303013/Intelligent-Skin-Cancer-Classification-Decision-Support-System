"""
app.py — XDERM CDSS (Clinical Decision Support System)
Skin Lesion Screening Web App — SINGLE-FILE VERSION

Everything (routes, HTML templates, CSS) lives in this one file, so you
can copy/paste it as-is and run it with nothing else needed except
your model file.

Features:
  - Auto-opens in your default browser (e.g. Chrome) — you never have
    to click a link in the terminal.
  - Register page: Name (letters + spaces only), Password (letters +
    numbers + at least one special symbol), Phone (exactly 10 digits).
  - Login page: same validation rules, plus invalid-credential checks.
  - Live red inline validation messages (JavaScript) + server-side
    re-validation with flash messages as a backstop.
  - Dark-blue -> light-blue themed UI, Times New Roman throughout.
  - Dermatologist / hospital referral directory page.
  - Grad-CAM skin lesion screening using your trained EfficientNetV2-S
    model.

IMPORTANT: Educational screening demo only, not a medical device.

Run:
    pip install -r requirements.txt   (flask, torch, torchvision, pillow,
                                        numpy, matplotlib, werkzeug)
    python app.py

Your browser opens automatically at http://127.0.0.1:5000/login
"""

import os
import io
import re
import json
import base64
import secrets
import webbrowser
import threading

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
from werkzeug.security import generate_password_hash, check_password_hash

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash
)
from jinja2 import DictLoader


# =========================================================
# 1. PATHS  (adjust MODEL_FILE if your models/ folder lives elsewhere)
# =========================================================

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)  # assumes this file sits inside skin_app/, next to models/
MODEL_FILE = os.path.join(PROJECT_ROOT, "models", "efficientnetv2b3_attention_best.pth")
USERS_FILE = os.path.join(APP_DIR, "users.json")

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp"}


# =========================================================
# 2. USER STORE  (demo only — a JSON file stands in for a real DB)
# =========================================================
# Two built-in demo accounts so you can log straight in without
# registering first:
#   admin  / Admin@123
#   doctor / Doctor@2026

DEFAULT_USERS = {
    "admin": {"password": generate_password_hash("Admin@123"), "phone": "9999999999"},
    "doctor": {"password": generate_password_hash("Doctor@2026"), "phone": "8888888888"},
}


def load_users():
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return dict(DEFAULT_USERS)


def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


# In-memory "last screening result" per logged-in user, so the uploaded
# image and prediction stay visible when the user switches to the
# Dermatologists tab and comes back — cleared only on logout.
USER_RESULTS = {}


# =========================================================
# 3. VALIDATION RULES
# =========================================================
# Name      -> letters and spaces only
# Password  -> letters, digits and special symbols; must contain at
#              least one of each, 8 characters minimum
# Phone     -> exactly 10 digits

NAME_RE = re.compile(r"^[A-Za-z][A-Za-z ]{1,49}$")
PHONE_RE = re.compile(r"^\d{10}$")
PASSWORD_ALLOWED_RE = re.compile(r"^[A-Za-z0-9!@#$%^&*()_+\-=\[\]{}|;:',.<>/?~`]{8,64}$")
PASSWORD_HAS_LETTER = re.compile(r"[A-Za-z]")
PASSWORD_HAS_DIGIT = re.compile(r"\d")
PASSWORD_HAS_SPECIAL = re.compile(r"[!@#$%^&*()_+\-=\[\]{}|;:',.<>/?~`]")


def validate_name(name):
    if not name:
        return "Name is required."
    if not NAME_RE.match(name):
        return "Name must contain only letters and spaces (2-50 characters)."
    return None


def validate_phone(phone):
    if not phone:
        return "Phone number is required."
    if not PHONE_RE.match(phone):
        return "Phone number must be exactly 10 digits, numbers only."
    return None


def validate_password(password):
    if not password:
        return "Password is required."
    if not PASSWORD_ALLOWED_RE.match(password):
        return ("Password must be 8-64 characters using only letters, numbers "
                "and the special symbols ! @ # $ % ^ & * ( ) _ + - = etc. (no spaces).")
    if not PASSWORD_HAS_LETTER.search(password):
        return "Password must contain at least one letter."
    if not PASSWORD_HAS_DIGIT.search(password):
        return "Password must contain at least one number."
    if not PASSWORD_HAS_SPECIAL.search(password):
        return "Password must contain at least one special symbol (e.g. @ # $ % &)."
    return None


# =========================================================
# 4. CLASSES + DISEASE INFO  (same as your prediction script)
# =========================================================

class_names = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NUM_CLASSES = 7

DISEASE_INFO = {
    "akiec": {
        "full_name": "Actinic Keratosis / Intraepithelial Carcinoma (Bowen's disease)",
        "description": "A rough, scaly patch caused by years of sun exposure. Considered precancerous to early-stage cancerous, and can progress to squamous cell carcinoma if untreated.",
        "symptoms": ["Rough, dry, scaly patch, often pink/red", "Feels like sandpaper", "Usually on sun-exposed skin", "May itch or burn"],
        "risk_level": "Moderate",
        "risk_class": "risk-moderate",
        "next_steps": ["See a dermatologist for biopsy", "Use SPF 30+ daily", "Do not self-treat or remove it"],
    },
    "bcc": {
        "full_name": "Basal Cell Carcinoma",
        "description": "The most common skin cancer. Grows slowly and rarely spreads, but can invade nearby tissue if left untreated.",
        "symptoms": ["Pearly or waxy bump", "Flat, scar-like lesion", "A sore that bleeds and returns", "Slow growth over months"],
        "risk_level": "Moderate-High",
        "risk_class": "risk-high",
        "next_steps": ["See a dermatologist promptly", "Typically treated with minor surgery", "Do not delay treatment"],
    },
    "bkl": {
        "full_name": "Benign Keratosis-like Lesion",
        "description": "A common, non-cancerous growth (e.g. seborrheic keratosis). Can look dark and irregular but is not dangerous.",
        "symptoms": ["Waxy, 'stuck-on' look", "Brown/black/tan color", "Slightly raised, rough surface", "Painless, slow-growing"],
        "risk_level": "Low",
        "risk_class": "risk-low",
        "next_steps": ["Usually no treatment needed", "Have a doctor confirm if it changes", "Removal is optional/cosmetic"],
    },
    "df": {
        "full_name": "Dermatofibroma",
        "description": "A common, benign, firm nodule, often forming after a minor injury such as an insect bite.",
        "symptoms": ["Small, firm, raised bump", "Brown, pink or reddish", "Dimples inward when pinched", "Usually painless"],
        "risk_level": "Low",
        "risk_class": "risk-low",
        "next_steps": ["Generally no treatment needed", "See a doctor if it grows or bleeds", "Can be removed if bothersome"],
    },
    "mel": {
        "full_name": "Melanoma",
        "description": "The most serious skin cancer. Can spread to other organs if not caught early.",
        "symptoms": ["Asymmetric, irregular border", "Multiple colors in one lesion", "Diameter over 6mm", "Evolving in size/shape/color"],
        "risk_level": "High",
        "risk_class": "risk-high",
        "next_steps": ["See a dermatologist or oncologist ASAP", "Request a dermoscopic exam and biopsy", "Do not wait and watch"],
    },
    "nv": {
        "full_name": "Melanocytic Nevus (common mole)",
        "description": "An ordinary mole made of pigment cell clusters. The vast majority are completely harmless.",
        "symptoms": ["Round/oval, even border", "Uniform color", "Usually under 6mm, stable", "Flat or slightly raised"],
        "risk_level": "Low",
        "risk_class": "risk-low",
        "next_steps": ["No action needed for stable moles", "Watch for ABCDE warning signs", "Annual skin check if many moles"],
    },
    "vasc": {
        "full_name": "Vascular Lesion",
        "description": "A lesion formed from blood vessels (e.g. hemangioma, angioma). Usually benign.",
        "symptoms": ["Red, purple, or blue coloring", "May blanch when pressed", "Can appear suddenly", "Usually painless"],
        "risk_level": "Low",
        "risk_class": "risk-low",
        "next_steps": ["Usually no treatment needed", "See a doctor if it bleeds or grows fast", "Laser removal is optional"],
    },
}


# =========================================================
# 5. SAMPLE DERMATOLOGY REFERRAL DIRECTORY
# =========================================================
# NOTE: These are hospital-level general contact numbers only
# (public switchboard / enquiry lines), not individual doctors'
# personal phone numbers. Confirm the on-duty dermatologist and
# extension with each hospital's reception, and update this list
# with your own verified contacts before relying on it clinically.

DOCTORS = [
    {
        "hospital": "Aster Medcity",
        "location": "Ernakulam",
        "department": "Dermatology (Clinical & Cosmetic)",
        "phone": "0484-6699999",
        "note": "Ask reception to connect you to the Dermatology OPD desk.",
    },
    {
        "hospital": "Rajagiri Hospital",
        "location": " Aluva, Ernakulam",
        "department": "Dermatology OPD",
        "phone": "0484-2905000",
        "note": "Enquiry / appointments line — request the Dermatology department.",
    },
    {
        "hospital": "Lisie Hospital",
        "location": "Lisie Hospital Road, North Kaloor, Ernakulam",
        "department": "Dermatology OPD",
        "phone": "0484-2402044",
        "note": "General hospital line — ask to be routed to Dermatology.",
    },
]


# =========================================================
# 6. FLASK APP
# =========================================================

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload cap


# =========================================================
# 7. EMBEDDED TEMPLATES + CSS  (no separate template/static files)
# =========================================================

STYLE_CSS = """
:root {
  --navy-deep:#061a30; --navy:#0b2b4a; --blue-mid:#14588a; --blue-bright:#2c86c9;
  --blue-light:#a9d6ef; --blue-pale:#e7f3fb; --white:#ffffff; --ink:#0b2b4a;
  --danger:#b3261e; --danger-bg:#fdeceb; --success:#1c6e3c; --success-bg:#e7f6ec;
  --risk-low:#1c6e3c; --risk-mod:#b06a00; --risk-high:#b3261e;
  --shadow:0 14px 40px rgba(6,26,48,0.35);
}
* { box-sizing: border-box; }
body {
  margin:0; min-height:100vh; font-family:"Times New Roman", Times, Georgia, serif;
  color:var(--ink);
  background:linear-gradient(160deg, var(--navy-deep) 0%, var(--navy) 32%, var(--blue-mid) 68%, var(--blue-bright) 100%);
  background-attachment:fixed; display:flex; flex-direction:column;
}
.brandbar { padding:22px 28px; text-align:center; color:var(--white);
  border-bottom:3px solid var(--blue-light);
  background:linear-gradient(90deg, var(--navy-deep), var(--blue-mid));
  box-shadow:0 4px 18px rgba(0,0,0,0.35); }
.brandbar h1 { margin:0; font-size:2.1rem; letter-spacing:1.5px; font-weight:700;
  text-shadow:0 2px 6px rgba(0,0,0,0.4); }
.brandbar h1 .accent { color:var(--blue-light); }
.brandbar p.tagline { margin:6px 0 0; font-size:1.05rem; font-style:italic;
  color:var(--blue-pale); letter-spacing:0.5px; }
nav.mainnav { background:var(--navy); padding:10px 24px; display:flex; gap:22px;
  justify-content:center; flex-wrap:wrap; border-bottom:2px solid var(--blue-bright); }
nav.mainnav a { color:var(--blue-pale); text-decoration:none; font-size:1.05rem;
  padding:6px 14px; border-radius:20px; transition:background .2s ease,color .2s ease; }
nav.mainnav a:hover, nav.mainnav a.active { background:var(--blue-bright); color:var(--white); }
.logout-fab { position:fixed; top:16px; left:16px; width:42px; height:42px; border-radius:50%;
  background:rgba(255,255,255,0.12); border:2px solid var(--blue-light); color:var(--white);
  display:flex; align-items:center; justify-content:center; text-decoration:none; z-index:1000;
  box-shadow:0 4px 14px rgba(0,0,0,0.35); transition:background .2s ease, transform .15s ease; }
.logout-fab:hover { background:var(--danger); border-color:var(--danger); transform:scale(1.08); }
.logout-fab svg { width:20px; height:20px; }
main.page { flex:1; display:flex; align-items:flex-start; justify-content:center;
  padding:40px 18px 60px; }
.card { background:var(--white); border-radius:16px; box-shadow:var(--shadow);
  padding:34px 38px; width:100%; max-width:460px; border:1px solid var(--blue-light); }
.card.wide { max-width:900px; }
.card.xwide { max-width:1080px; }
.card h2 { margin-top:0; color:var(--navy); text-align:center;
  border-bottom:2px solid var(--blue-light); padding-bottom:14px; letter-spacing:0.5px; }
.card .subtitle { text-align:center; color:var(--blue-mid); margin-top:-8px;
  margin-bottom:22px; font-style:italic; }
.field { margin-bottom:18px; }
.field label { display:block; font-weight:bold; margin-bottom:6px; color:var(--navy); }
.field input { width:100%; padding:11px 13px; font-size:1rem; font-family:inherit;
  border:2px solid var(--blue-light); border-radius:8px; background:var(--blue-pale);
  color:var(--ink); transition:border-color .2s ease, box-shadow .2s ease; }
.field input:focus { outline:none; border-color:var(--blue-bright);
  box-shadow:0 0 0 3px rgba(44,134,201,0.25); background:var(--white); }
.field input.invalid { border-color:var(--danger); background:var(--danger-bg); }
.field .hint { font-size:0.85rem; color:var(--blue-mid); margin-top:4px; }
.field .field-error { font-size:0.85rem; color:var(--danger); margin-top:4px;
  font-weight:bold; display:none; }
.field .field-error.show { display:block; }
button.btn, a.btn { display:inline-block; width:100%; padding:12px 18px; font-size:1.05rem;
  font-family:inherit; font-weight:bold; letter-spacing:0.5px; color:var(--white);
  background:linear-gradient(90deg, var(--navy), var(--blue-bright)); border:none;
  border-radius:9px; cursor:pointer; text-align:center; text-decoration:none;
  box-shadow:0 6px 16px rgba(11,43,74,0.35); transition:transform .15s ease, box-shadow .15s ease; }
button.btn:hover, a.btn:hover { transform:translateY(-2px); box-shadow:0 10px 22px rgba(11,43,74,0.45); }
button.btn.secondary, a.btn.secondary { background:var(--white); color:var(--navy);
  border:2px solid var(--blue-bright); }
.flash-list { list-style:none; padding:0; margin:0 0 18px; }
.flash-list li { background:var(--danger-bg); color:var(--danger); border:1px solid var(--danger);
  border-radius:8px; padding:10px 14px; margin-bottom:8px; font-weight:bold; }
.flash-list li.success { background:var(--success-bg); color:var(--success); border-color:var(--success); }
.form-footer { text-align:center; margin-top:18px; color:var(--navy); }
.form-footer a { color:var(--blue-bright); font-weight:bold; text-decoration:none; }
.form-footer a:hover { text-decoration:underline; }
.dropzone { border:3px dashed var(--blue-bright); border-radius:12px; padding:34px 18px;
  text-align:center; background:var(--blue-pale); margin-bottom:18px; }
.dropzone p { margin:6px 0; color:var(--navy); }
.dropzone .filename { font-weight:bold; color:var(--blue-mid); }
.model-warning { background:var(--danger-bg); border:1px solid var(--danger); color:var(--danger);
  border-radius:8px; padding:12px 16px; margin-bottom:20px; font-weight:bold; text-align:center; }
.result-grid { display:grid; grid-template-columns:1.1fr 1fr; gap:26px; }
@media (max-width:760px) { .result-grid { grid-template-columns:1fr; } }
.pred-banner { background:linear-gradient(90deg, var(--navy), var(--blue-mid)); color:var(--white);
  border-radius:12px; padding:16px 20px; margin-bottom:20px; text-align:center; }
.pred-banner .pred-name { font-size:1.6rem; font-weight:bold; }
.pred-banner .pred-conf { font-size:1.05rem; color:var(--blue-light); }
.imgs-row { display:flex; gap:12px; flex-wrap:wrap; justify-content:center; margin-bottom:18px; }
.imgs-row figure { margin:0; text-align:center; }
.imgs-row img { width:160px; height:160px; object-fit:cover; border-radius:10px;
  border:2px solid var(--blue-light); }
.imgs-row figcaption { font-size:0.85rem; color:var(--blue-mid); margin-top:4px; }
.risk-badge { display:inline-block; padding:4px 14px; border-radius:20px; font-weight:bold; color:var(--white); }
.risk-low { background:var(--risk-low); }
.risk-moderate { background:var(--risk-mod); }
.risk-high { background:var(--risk-high); }
.prob-bar-row { display:flex; align-items:center; gap:10px; margin-bottom:8px; }
.prob-bar-row .label { width:60px; text-transform:uppercase; font-size:0.85rem; color:var(--navy); font-weight:bold; }
.prob-bar-track { flex:1; background:var(--blue-pale); border-radius:20px; overflow:hidden;
  height:16px; border:1px solid var(--blue-light); }
.prob-bar-fill { height:100%; background:linear-gradient(90deg, var(--blue-mid), var(--blue-bright)); }
.prob-bar-row .pct { width:54px; text-align:right; font-size:0.85rem; color:var(--navy); }
ul.symptom-list, ul.step-list { margin:8px 0 0; padding-left:20px; }
.disclaimer { margin-top:22px; font-size:0.85rem; color:var(--blue-mid);
  border-top:1px solid var(--blue-light); padding-top:12px; text-align:center; }
.doctor-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(260px,1fr)); gap:20px; }
.doctor-card { background:var(--blue-pale); border:2px solid var(--blue-light); border-radius:14px;
  padding:20px; text-align:center; transition:transform .15s ease; }
.doctor-card:hover { transform:translateY(-4px); }
.doctor-card .icon { font-size:2.4rem; margin-bottom:6px; }
.doctor-card h3 { margin:4px 0 2px; color:var(--navy); }
.doctor-card .dept { color:var(--blue-mid); font-style:italic; margin-bottom:10px; }
.doctor-card .phone { font-size:1.1rem; font-weight:bold; color:var(--navy); }
.doctor-card .location { font-size:0.85rem; color:var(--blue-mid); margin-top:6px; }
.doctor-card .note { font-size:0.8rem; color:#555; margin-top:8px; }
.doctors-note { background:var(--danger-bg); border:1px solid var(--danger); color:var(--danger);
  border-radius:8px; padding:12px 16px; margin-bottom:20px; font-size:0.9rem; }
.top-actions { display:flex; justify-content:space-between; align-items:center;
  margin-bottom:20px; flex-wrap:wrap; gap:10px; }
footer.sitefooter { text-align:center; padding:16px; color:var(--blue-pale); font-size:0.85rem; }
"""

BASE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{% block title %}XDERM CDSS{% endblock %}</title>
<style>""" + STYLE_CSS + """</style>
</head>
<body>

{% if session.get('username') %}
<a href="{{ url_for('logout') }}" class="logout-fab" title="Logout" aria-label="Logout">
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"></path>
    <polyline points="16 17 21 12 16 7"></polyline>
    <line x1="21" y1="12" x2="9" y2="12"></line>
  </svg>
</a>
{% endif %}

<div class="brandbar">
  <h1>XDERM <span class="accent">CDSS</span></h1>
  <p class="tagline">Clinical Decision Support System &mdash; Skin Lesion Screening</p>
</div>

{% if session.get('username') %}
<nav class="mainnav">
  <a href="{{ url_for('upload') }}" class="{{ 'active' if request.endpoint == 'upload' }}">Upload &amp; Screen</a>
  <a href="{{ url_for('doctors') }}" class="{{ 'active' if request.endpoint == 'doctors' }}">Dermatologists</a>
</nav>
{% endif %}

<main class="page">
  <div style="width:100%; display:flex; justify-content:center;">
    {% block content %}{% endblock %}
  </div>
</main>



</body>
</html>
"""

LOGIN_HTML = """{% extends "base.html" %}
{% block title %}Login - XDERM CDSS{% endblock %}
{% block content %}
<div class="card">
  <h2>Login</h2>
  <p class="subtitle">Sign in to access the screening tool</p>

  {% with messages = get_flashed_messages(with_categories=true) %}
    {% if messages %}
    <ul class="flash-list">
      {% for category, msg in messages %}
        <li class="{{ category }}">{{ msg }}</li>
      {% endfor %}
    </ul>
    {% endif %}
  {% endwith %}

  <form method="POST" action="{{ url_for('login') }}" id="loginForm" novalidate>
    <div class="field">
      <label for="username">Name</label>
      <input type="text" id="username" name="username" value="{{ name }}"
             placeholder="e.g. John Kurian" autocomplete="name" required>
      <div class="hint">Letters and spaces only.</div>
      <div class="field-error" id="username-error">Only letters and spaces are accepted for the name.</div>
    </div>

    <div class="field">
      <label for="password">Password</label>
      <input type="password" id="password" name="password"
             placeholder="Your password" autocomplete="current-password" required>
      <div class="hint">Letters, numbers and special symbols (min. 8 characters).</div>
      <div class="field-error" id="password-error">Password must include letters, numbers and a special symbol.</div>
    </div>

    <button type="submit" class="btn">Login</button>
  </form>

  <div class="form-footer">
    New here? <a href="{{ url_for('register') }}">Create an account</a>
  </div>
</div>

<script>
const nameRegex = /^[A-Za-z][A-Za-z ]{1,49}$/;
const passwordRegex = /^(?=.*[A-Za-z])(?=.*\\d)(?=.*[^A-Za-z0-9\\s]).{8,64}$/;

function wireField(inputId, errorId, regex) {
  const input = document.getElementById(inputId);
  const error = document.getElementById(errorId);
  function check() {
    const ok = regex.test(input.value);
    input.classList.toggle('invalid', input.value.length > 0 && !ok);
    error.classList.toggle('show', input.value.length > 0 && !ok);
    return ok;
  }
  input.addEventListener('input', check);
  return check;
}

const checkName = wireField('username', 'username-error', nameRegex);
const checkPassword = wireField('password', 'password-error', passwordRegex);

document.getElementById('loginForm').addEventListener('submit', function (e) {
  const nameOk = checkName();
  const passOk = checkPassword();
  if (!nameOk || !passOk) { e.preventDefault(); }
});
</script>
{% endblock %}
"""

REGISTER_HTML = """{% extends "base.html" %}
{% block title %}Register - XDERM CDSS{% endblock %}
{% block content %}
<div class="card">
  <h2>Create Account</h2>
  <p class="subtitle">Register to use the screening tool</p>

  {% with messages = get_flashed_messages(with_categories=true) %}
    {% if messages %}
    <ul class="flash-list">
      {% for category, msg in messages %}
        <li class="{{ category }}">{{ msg }}</li>
      {% endfor %}
    </ul>
    {% endif %}
  {% endwith %}

  <form method="POST" action="{{ url_for('register') }}" id="registerForm" novalidate>
    <div class="field">
      <label for="name">Name</label>
      <input type="text" id="name" name="name" value="{{ name }}"
             placeholder="e.g. Anjali Varma" autocomplete="name" required>
      <div class="hint">Letters and spaces only, 2-50 characters.</div>
      <div class="field-error" id="name-error">Only letters and spaces are accepted for the name.</div>
    </div>

    <div class="field">
      <label for="phone">Phone Number</label>
      <input type="text" id="phone" name="phone" value="{{ phone }}"
             placeholder="10-digit mobile number" autocomplete="tel" inputmode="numeric" maxlength="10" required>
      <div class="hint">Exactly 10 digits, numbers only.</div>
      <div class="field-error" id="phone-error">Phone number must be exactly 10 digits.</div>
    </div>

    <div class="field">
      <label for="password">Password</label>
      <input type="password" id="password" name="password"
             placeholder="Create a password" autocomplete="new-password" required>
      <div class="hint">8-64 characters: letters + numbers + at least one special symbol (@ # $ % &amp; etc.), no spaces.</div>
      <div class="field-error" id="password-error">Password must include letters, numbers and a special symbol.</div>
    </div>

    <div class="field">
      <label for="confirm_password">Confirm Password</label>
      <input type="password" id="confirm_password" name="confirm_password"
             placeholder="Re-type your password" autocomplete="new-password" required>
      <div class="field-error" id="confirm-error">Passwords do not match.</div>
    </div>

    <button type="submit" class="btn">Register</button>
  </form>

  <div class="form-footer">
    Already have an account? <a href="{{ url_for('login') }}">Login</a>
  </div>
</div>

<script>
const nameRegex = /^[A-Za-z][A-Za-z ]{1,49}$/;
const phoneRegex = /^\\d{10}$/;
const passwordRegex = /^(?=.*[A-Za-z])(?=.*\\d)(?=.*[^A-Za-z0-9\\s]).{8,64}$/;

function wireField(inputId, errorId, testFn) {
  const input = document.getElementById(inputId);
  const error = document.getElementById(errorId);
  function check() {
    const ok = testFn(input.value);
    input.classList.toggle('invalid', input.value.length > 0 && !ok);
    error.classList.toggle('show', input.value.length > 0 && !ok);
    return ok;
  }
  input.addEventListener('input', check);
  return check;
}

const checkName = wireField('name', 'name-error', v => nameRegex.test(v));
const checkPhone = wireField('phone', 'phone-error', v => phoneRegex.test(v));
const checkPassword = wireField('password', 'password-error', v => passwordRegex.test(v));

const confirmInput = document.getElementById('confirm_password');
const confirmError = document.getElementById('confirm-error');
function checkConfirm() {
  const ok = confirmInput.value === document.getElementById('password').value && confirmInput.value.length > 0;
  confirmInput.classList.toggle('invalid', confirmInput.value.length > 0 && !ok);
  confirmError.classList.toggle('show', confirmInput.value.length > 0 && !ok);
  return ok;
}
confirmInput.addEventListener('input', checkConfirm);
document.getElementById('password').addEventListener('input', checkConfirm);

document.getElementById('phone').addEventListener('input', function () {
  this.value = this.value.replace(/\\D/g, '').slice(0, 10);
});

document.getElementById('registerForm').addEventListener('submit', function (e) {
  const ok = checkName() && checkPhone() && checkPassword() && checkConfirm();
  if (!ok) { e.preventDefault(); }
});
</script>
{% endblock %}
"""

UPLOAD_HTML = """{% extends "base.html" %}
{% block title %}Upload - XDERM CDSS{% endblock %}
{% block content %}
<div class="card xwide">
  <h2>Skin Lesion Screening</h2>
  <p class="subtitle">Upload a clear, well-lit photo of the lesion for AI-assisted screening</p>

  {% with messages = get_flashed_messages(with_categories=true) %}
    {% if messages %}
    <ul class="flash-list">
      {% for category, msg in messages %}
        <li class="{{ category }}">{{ msg }}</li>
      {% endfor %}
    </ul>
    {% endif %}
  {% endwith %}

  {% if not model_ready %}
  <div class="model-warning">
    The AI model file was not found on the server, so predictions are disabled.
    Check MODEL_FILE in app.py.
  </div>
  {% endif %}

  <form method="POST" action="{{ url_for('upload') }}" enctype="multipart/form-data" id="uploadForm">
    <div class="dropzone">
      <p>Choose a JPG, JPEG, PNG or BMP image (max 10 MB)</p>
      <input type="file" id="image" name="image" accept=".jpg,.jpeg,.png,.bmp" required
             style="margin-top:10px;">
      <p class="filename" id="filenameLabel"></p>
    </div>
    <button type="submit" class="btn" {{ 'disabled' if not model_ready }}>Run Screening</button>
  </form>

  {% if result %}
  <div class="pred-banner">
    <div class="pred-name">{{ result.info.full_name }}</div>
    <div class="pred-conf">Confidence: {{ result.confidence_pct }}%
      &middot; <span class="risk-badge {{ result.info.risk_class }}">{{ result.info.risk_level }} risk</span>
    </div>
  </div>

  <p class="subtitle" style="margin-bottom:14px;">File analyzed: {{ filename }}</p>

  <div class="imgs-row">
    <figure>
      <img src="data:image/png;base64,{{ result.images.original }}" alt="Original image">
      <figcaption>Original</figcaption>
    </figure>
    <figure>
      <img src="data:image/png;base64,{{ result.images.heatmap }}" alt="Grad-CAM heatmap">
      <figcaption>Grad-CAM Heatmap</figcaption>
    </figure>
    <figure>
      <img src="data:image/png;base64,{{ result.images.overlay }}" alt="Overlay">
      <figcaption>Overlay</figcaption>
    </figure>
  </div>

  <div class="result-grid">
    <div>
      <h3>About this finding</h3>
      <p>{{ result.info.description }}</p>

      <h3>Typical Symptoms</h3>
      <ul class="symptom-list">
        {% for s in result.info.symptoms %}
        <li>{{ s }}</li>
        {% endfor %}
      </ul>

      <h3>Recommended Next Steps</h3>
      <ul class="step-list">
        {% for step in result.info.next_steps %}
        <li>{{ step }}</li>
        {% endfor %}
      </ul>
    </div>

    <div>
      <h3>Confidence Across All Classes</h3>
      {% for p in result.all_probs %}
      <div class="prob-bar-row">
        <div class="label">{{ p.name }}</div>
        <div class="prob-bar-track">
          <div class="prob-bar-fill" style="width: {{ p.pct }}%;"></div>
        </div>
        <div class="pct">{{ p.pct }}%</div>
      </div>
      {% endfor %}
    </div>
  </div>

  <div class="top-actions" style="margin-top:24px; justify-content:center;">
    <a href="{{ url_for('doctors') }}" class="btn" style="width:auto;">Find a Dermatologist</a>
  </div>
  {% endif %}
</div>

<script>
document.getElementById('image').addEventListener('change', function () {
  const label = document.getElementById('filenameLabel');
  label.textContent = this.files.length ? 'Selected: ' + this.files[0].name : '';
});
</script>
{% endblock %}
"""

DOCTORS_HTML = """{% extends "base.html" %}
{% block title %}Dermatologists - XDERM CDSS{% endblock %}
{% block content %}
<div class="card wide">
  <h2>Dermatology Referral Directory</h2>
  <p class="subtitle">Hospitals in Ernakulam with dermatology departments</p>

  <div class="doctor-grid">
    {% for d in doctors %}
    <div class="doctor-card">
      <div class="icon">&#129657;</div>
      <h3>{{ d.hospital }}</h3>
      <div class="dept">{{ d.department }}</div>
      <div class="phone">{{ d.phone }}</div>
      <div class="location">{{ d.location }}</div>
      <div class="note">{{ d.note }}</div>
    </div>
    {% endfor %}
  </div>

  <div class="top-actions" style="margin-top:24px; justify-content:center;">
    <a href="{{ url_for('upload') }}" class="btn secondary" style="width:auto;">Back to Screening</a>
  </div>
</div>
{% endblock %}
"""

app.jinja_loader = DictLoader({
    "base.html": BASE_HTML,
    "login.html": LOGIN_HTML,
    "register.html": REGISTER_HTML,
    "upload.html": UPLOAD_HTML,
    "doctors.html": DOCTORS_HTML,
})


# =========================================================
# 8. MODEL + GRAD-CAM  (loaded once at startup)
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor, class_index):
        self.model.zero_grad()
        output = self.model(input_tensor)
        output[0, class_index].backward()

        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=(1, 2))

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.cpu().numpy()


def load_model():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")
    model = efficientnet_v2_s(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model = model.to(device)
    state_dict = torch.load(MODEL_FILE, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


MODEL = None
GRAD_CAM = None
MODEL_LOAD_ERROR = None

try:
    print("Loading model on", device, "...")
    MODEL = load_model()
    GRAD_CAM = GradCAM(MODEL, MODEL.features[-1])
    print("Model ready.")
except Exception as exc:  # noqa: BLE001 — surfaced to the user on the upload page
    MODEL_LOAD_ERROR = str(exc)
    print("WARNING: model did not load —", MODEL_LOAD_ERROR)
    print("The site will still run; the Upload page will show this warning "
          "until MODEL_FILE points to a real .pth file.")


def image_to_base64(pil_or_array_image, is_array=False):
    """Encode a PIL image or a 0-1 float RGB numpy array as a base64 PNG string."""
    if is_array:
        img = Image.fromarray(np.uint8(np.clip(pil_or_array_image, 0, 1) * 255))
    else:
        img = pil_or_array_image
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def run_prediction(pil_image):
    tensor = inference_transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = MODEL(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    confidence, predicted_idx = torch.max(probabilities, dim=0)
    predicted_idx = predicted_idx.item()
    predicted_class = class_names[predicted_idx]
    confidence_pct = confidence.item() * 100

    all_probs = [
        {"name": class_names[i], "pct": round(probabilities[i].item() * 100, 2)}
        for i in range(NUM_CLASSES)
    ]
    all_probs.sort(key=lambda x: x["pct"], reverse=True)

    # Grad-CAM (re-run forward+backward with gradients enabled)
    tensor_grad = inference_transform(pil_image).unsqueeze(0).to(device)
    cam = GRAD_CAM.generate(tensor_grad, predicted_idx)

    cam_resized = np.array(
        Image.fromarray(np.uint8(cam * 255)).resize(pil_image.size, resample=Image.BILINEAR)
    ) / 255.0
    heatmap_rgb = plt.get_cmap("jet")(cam_resized)[:, :, :3]
    base_arr = np.array(pil_image.convert("RGB")) / 255.0
    overlay = np.clip(0.55 * base_arr + 0.45 * heatmap_rgb, 0, 1)

    images_b64 = {
        "original": image_to_base64(pil_image.convert("RGB")),
        "heatmap": image_to_base64(heatmap_rgb, is_array=True),
        "overlay": image_to_base64(overlay, is_array=True),
    }

    return {
        "predicted_class": predicted_class,
        "confidence_pct": round(confidence_pct, 2),
        "all_probs": all_probs,
        "info": DISEASE_INFO[predicted_class],
        "images": images_b64,
    }


# =========================================================
# 9. AUTH HELPERS
# =========================================================

def login_required(view_func):
    def wrapper(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)
    wrapper.__name__ = view_func.__name__
    return wrapper


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# =========================================================
# 10. ROUTES
# =========================================================

@app.route("/")
def index():
    if "username" in session:
        return redirect(url_for("upload"))
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        phone = request.form.get("phone", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        errors = []
        name_error = validate_name(name)
        phone_error = validate_phone(phone)
        password_error = validate_password(password)

        if name_error:
            errors.append(name_error)
        if phone_error:
            errors.append(phone_error)
        if password_error:
            errors.append(password_error)
        if not password_error and password != confirm:
            errors.append("Password and confirm password do not match.")

        users = load_users()
        if not name_error and name in users:
            errors.append("That name is already registered. Please log in instead.")

        if errors:
            for e in errors:
                flash(e)
            return render_template("register.html", name=name, phone=phone)

        users[name] = {
            "password": generate_password_hash(password),
            "phone": phone,
        }
        save_users(users)
        flash("Registration successful. Please log in below.", "success")
        return redirect(url_for("login"))

    return render_template("register.html", name="", phone="")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        name = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        errors = []
        name_error = validate_name(name)
        password_error = validate_password(password)
        if name_error:
            errors.append(name_error)
        if password_error:
            errors.append(password_error)

        if errors:
            for e in errors:
                flash(e)
            return render_template("login.html", name=name)

        users = load_users()
        record = users.get(name)
        if record and check_password_hash(record["password"], password):
            session["username"] = name
            return redirect(url_for("upload"))

        flash("Invalid name or password.")
        return render_template("login.html", name=name)

    return render_template("login.html", name="")


@app.route("/logout")
def logout():
    username = session.get("username")
    if username:
        USER_RESULTS.pop(username, None)
    session.clear()
    return redirect(url_for("login"))


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    username = session["username"]

    if request.method == "POST":
        if MODEL is None:
            flash("The prediction model is not loaded on this server "
                  "(MODEL_FILE not found). Ask the administrator to check "
                  "the models/ folder.")
            return redirect(url_for("upload"))

        file = request.files.get("image")
        if not file or file.filename == "":
            flash("Please choose an image file.")
            return redirect(url_for("upload"))
        if not allowed_file(file.filename):
            flash("Unsupported file type. Use JPG, JPEG, PNG or BMP.")
            return redirect(url_for("upload"))

        try:
            pil_image = Image.open(file.stream).convert("RGB")
        except Exception:
            flash("Could not read that image file.")
            return redirect(url_for("upload"))

        result = run_prediction(pil_image)
        # Keep this on the server for the duration of the session, so it
        # stays visible when the user leaves for the Dermatologists tab
        # and comes back, until they log out.
        USER_RESULTS[username] = {"result": result, "filename": file.filename}
        return render_template("upload.html", model_ready=True,
                                result=result, filename=file.filename)

    stored = USER_RESULTS.get(username)
    return render_template(
        "upload.html",
        model_ready=(MODEL is not None),
        result=stored["result"] if stored else None,
        filename=stored["filename"] if stored else None,
    )


@app.route("/doctors")
@login_required
def doctors():
    return render_template("doctors.html", doctors=DOCTORS)


# =========================================================
# 11. AUTO-OPEN BROWSER + RUN
# =========================================================

def _open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/login")


if __name__ == "__main__":
    # Only open the browser once, even with the debug auto-reloader
    # (which re-executes this file in a child process).
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        threading.Timer(1.25, _open_browser).start()
    app.run(debug=True)