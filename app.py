from flask import Flask,render_template,request, jsonify, send_file
from flask_cors import CORS
import random
import pickle
import numpy as np
import io
import os
import time
import requests
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.ensemble import RandomForestClassifier
import base64
GENAI_SDK_V2 = False
genai = None
types = None
legacy_genai = None
try:
    from google import genai
    from google.genai import types
    GENAI_SDK_V2 = True
except Exception:
    try:
        import google.generativeai as legacy_genai
    except Exception:
        pass
from gtts import gTTS

load_dotenv()

# Paths and globals for CSV-backed knowledge base and models
BASE_DIR = Path(__file__).resolve().parent
KCC_CSV_PATH = BASE_DIR / "KCC_raw_data.csv"
CROP_CSV_PATH = BASE_DIR / "Crop_recommendation.csv"

_kcc_vectorizer = None
_kcc_matrix = None
_kcc_questions = None
_kcc_answers = None

_crop_model = None
_crop_feature_names = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def _load_kcc_knowledgebase():
    """
    Load the KCC_raw_data.csv questions/answers into memory and
    build a TF-IDF index for fast similarity search.
    """
    global _kcc_vectorizer, _kcc_matrix, _kcc_questions, _kcc_answers
    try:
        if not KCC_CSV_PATH.exists():
            print(f"[KCC KB] CSV not found at {KCC_CSV_PATH}")
            return

        df = pd.read_csv(KCC_CSV_PATH)
        if "questions" not in df.columns or "answers" not in df.columns:
            print("[KCC KB] Expected 'questions' and 'answers' columns not found")
            return

        df = df.dropna(subset=["questions", "answers"])
        questions = df["questions"].astype(str).str.lower().tolist()
        answers = df["answers"].astype(str).tolist()

        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(questions)

        _kcc_vectorizer = vectorizer
        _kcc_matrix = matrix
        _kcc_questions = questions
        _kcc_answers = answers

        print(f"[KCC KB] Loaded {len(questions)} Q&A pairs into knowledge base.")
    except Exception as e:
        print(f"[KCC KB] Failed to load knowledge base: {e}")


def _get_kcc_context(user_query, top_k=3, min_similarity=0.15):
    """
    Given a farmer's question, find the most similar entries from KCC_raw_data.csv.
    Returns a formatted text block with top matches, or None if nothing is suitable.
    """
    if not user_query:
        return None
    if _kcc_vectorizer is None or _kcc_matrix is None:
        return None

    try:
        query_vec = _kcc_vectorizer.transform([user_query.lower()])
        sims = linear_kernel(query_vec, _kcc_matrix).flatten()

        if sims.size == 0:
            return None

        top_indices = sims.argsort()[::-1][:top_k]
        filtered = [(idx, sims[idx]) for idx in top_indices if sims[idx] >= min_similarity]
        if not filtered:
            return None

        lines = []
        for rank, (idx, score) in enumerate(filtered, start=1):
            q = _kcc_questions[idx]
            a = _kcc_answers[idx]
            lines.append(
                f"{rank}. Farmer question: {q}\n"
                f"   Expert answer: {a}"
            )

        return "\n".join(lines)
    except Exception as e:
        print(f"[KCC KB] Retrieval error: {e}")
        return None


def _load_crop_model():
    """
    Train an in-memory crop recommendation model using Crop_recommendation.csv.
    """
    global _crop_model
    try:
        if not CROP_CSV_PATH.exists():
            print(f"[CROP MODEL] CSV not found at {CROP_CSV_PATH}")
            return

        df = pd.read_csv(CROP_CSV_PATH)
        missing = [c for c in _crop_feature_names + ["label"] if c not in df.columns]
        if missing:
            print(f"[CROP MODEL] Missing expected columns: {missing}")
            return

        X = df[_crop_feature_names]
        y = df["label"]

        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X, y)
        _crop_model = model
        print(f"[CROP MODEL] Trained crop recommendation model on {len(df)} samples.")
    except Exception as e:
        print(f"[CROP MODEL] Failed to train model: {e}")


def _recommend_crop(features):
    """
    Predict the best crop label for given feature list
    [N, P, K, temperature, humidity, ph, rainfall].
    """
    if _crop_model is None:
        return None
    try:
        arr = np.array([features], dtype=float)
        pred = _crop_model.predict(arr)
        return str(pred[0])
    except Exception as e:
        print(f"[CROP MODEL] Prediction error: {e}")
        return None


_load_kcc_knowledgebase()
_load_crop_model()

_CLASSIFIER_PATH = BASE_DIR / "classifier1.pkl"
with open(_CLASSIFIER_PATH, "rb") as _f:
    model = pickle.load(_f)
GEMINI_API_KEY = (
    os.environ.get("GEMINI_API_KEY")
    or os.environ.get("GOOGLE_API_KEY")
)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "").strip() or None
gemini_client = None
if GEMINI_API_KEY:
    if GENAI_SDK_V2 and genai is not None:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    elif legacy_genai is not None:
        legacy_genai.configure(api_key=GEMINI_API_KEY)
        gemini_client = legacy_genai.GenerativeModel("gemini-1.5-flash")
    else:
        print("[GEMINI] No compatible Gemini SDK installed.")
else:
    print("[GEMINI] Missing GEMINI_API_KEY/GOOGLE_API_KEY in environment.")

app = Flask(__name__)
CORS(app)


def _api_manifest():
    """Machine-readable list of JSON/binary APIs (for SPA/mobile/any client)."""
    base = request.url_root.rstrip("/")
    return {
        "service": "FarmAssist Chatura API",
        "docs": f"{base}/api",
        "endpoints": [
            {"path": "/api", "method": "GET", "json": True, "description": "This discovery document."},
            {"path": "/health", "method": "GET", "json": True, "aliases": ["/api/health"]},
            {"path": "/chat", "method": "POST", "json": True, "aliases": ["/api/chat"], "body": {"message": "str", "lang": "optional", "lang_label": "optional"}},
            {"path": "/api/weather", "method": "GET", "json": True, "query": ["city", "lat", "lon"]},
            {"path": "/api/fertilizer-predict", "method": "POST", "json": True, "body": {"nitrogen": "float", "phosphorous": "float", "potassium": "float"}},
            {"path": "/api/fertilizer-calculator", "method": "POST", "json": True, "body": {"n_ratio_percent": "float", "p_ratio_percent": "float", "k_ratio_percent": "float", "required_n": "float", "required_p": "float", "required_k": "float", "area": "float"}},
            {"path": "/api/crop-recommendation", "method": "POST", "json": True, "body": {"N": "float", "P": "float", "K": "float", "temperature": "float", "humidity": "float", "ph": "float", "rainfall": "float"}},
            {"path": "/api/disease-detection", "method": "POST", "multipart": True, "field": "image"},
            {"path": "/transcribe", "method": "POST", "multipart": True, "aliases": ["/api/transcribe"], "field": "audio"},
            {"path": "/text-to-speech", "method": "POST", "json": True, "aliases": ["/api/text-to-speech"], "body": {"text": "str"}, "note": "Also GET ?text=&lang=en|hi|te"},
        ],
    }


@app.route("/api", methods=["GET"])
def api_discovery():
    return jsonify(_api_manifest())


@app.route('/')
def index():
     return render_template('main.html')

@app.route('/bot')
def botfun():
     return render_template('bot.html')

@app.route('/weather')
def weatherfun():
     return render_template('weather.html')

@app.route('/fcalculator')
def fcalculatorfun():
     return render_template('fcalculator.html')

@app.route('/voicechat')
def voicechatfun():
     return render_template('voicechat.html')

@app.route("/fpredictor")
def recommender():
    # Add your logic for the recommender app here
    # Return the appropriate template or data
    return render_template("fpredictor.html")

@app.route('/predict',methods=['POST'])
def predict():
    try:
        Nitrogen = float(request.form.get('Nitrogen', '0') or '0')
        Potassium = float(request.form.get('Potassium', '0') or '0')
        Phosphorous = float(request.form.get('Phosphorous', '0') or '0')
    except ValueError:
        return render_template('fpredictor.html', result='Invalid numeric input')

    # prediction
    result = model.predict(np.array([[Nitrogen,Potassium,Phosphorous]]))
    
    # Map numeric prediction to fertilizer name
    if result[0] == 0:
        result = 'TEN-TWENTY SIX-TWENTY SIX'
    elif result[0] == 1:
        result = 'Fourteen-Thirty Five-Fourteen'
    elif result[0] == 2:
        result = 'Seventeen-Seventeen-Seventeen'
    elif result[0] == 3:
        result = 'TWENTY-TWENTY'
    elif result[0] == 4:
        result = 'TWENTY EIGHT-TWENTY EIGHT'
    elif result[0] == 5:
        result = 'DAP'
    else:
        result = 'UREA'

    return render_template('fpredictor.html',result=str(result))


def _map_fertilizer_label(prediction_value):
    if prediction_value == 0:
        return 'TEN-TWENTY SIX-TWENTY SIX'
    if prediction_value == 1:
        return 'Fourteen-Thirty Five-Fourteen'
    if prediction_value == 2:
        return 'Seventeen-Seventeen-Seventeen'
    if prediction_value == 3:
        return 'TWENTY-TWENTY'
    if prediction_value == 4:
        return 'TWENTY EIGHT-TWENTY EIGHT'
    if prediction_value == 5:
        return 'DAP'
    return 'UREA'


def _build_weather_advice(temp, humidity, condition):
    tips = []
    c = (condition or "").lower()
    if temp is not None and temp > 35:
        tips.append("Increase irrigation frequency and mulch exposed soil.")
        tips.append("Avoid pesticide spraying during peak afternoon heat.")
    elif temp is not None and temp < 12:
        tips.append("Protect sensitive crops from low-temperature stress.")
        tips.append("Delay early-morning field work if frost risk is present.")

    if "rain" in c or "drizzle" in c:
        tips.append("Pause chemical sprays and ensure field drainage is clear.")
        tips.append("Good moisture conditions for transplanting activities.")
    elif "clear" in c:
        tips.append("Favorable conditions for harvesting and post-harvest drying.")

    if humidity is not None and humidity > 80:
        tips.append("Monitor crops for fungal disease in dense canopy areas.")

    if not tips:
        tips.append("Weather is stable. Continue regular farm operations.")
    return tips


def _fetch_openweather_current(city=None, lat=None, lon=None):
    if not OPENWEATHER_API_KEY:
        raise RuntimeError("OPENWEATHER_API_KEY is not configured.")
    base = "https://api.openweathermap.org/data/2.5/weather"
    params = {"units": "metric", "appid": OPENWEATHER_API_KEY}
    if city:
        params["q"] = city
    elif lat is not None and lon is not None:
        params["lat"] = lat
        params["lon"] = lon
    else:
        raise ValueError("Provide city or lat/lon")
    response = requests.get(base, params=params, timeout=20)
    if response.status_code >= 400:
        raise RuntimeError(f"Weather API error {response.status_code}: {response.text}")
    return response.json()


def _fetch_openweather_forecast(lat, lon):
    base = "https://api.openweathermap.org/data/2.5/forecast"
    response = requests.get(
        base,
        params={"lat": lat, "lon": lon, "units": "metric", "appid": OPENWEATHER_API_KEY},
        timeout=20,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Forecast API error {response.status_code}: {response.text}")
    data = response.json()
    grouped = {}
    for entry in data.get("list", []):
        dt = pd.to_datetime(entry.get("dt", 0), unit="s")
        day_key = dt.strftime("%Y-%m-%d")
        day_name = dt.strftime("%a")
        min_t = float(entry["main"]["temp_min"])
        max_t = float(entry["main"]["temp_max"])
        icon = entry["weather"][0]["icon"]
        if day_key not in grouped:
            grouped[day_key] = {
                "date": day_key,
                "day": day_name,
                "min": min_t,
                "max": max_t,
                "icon": icon,
            }
        else:
            grouped[day_key]["min"] = min(grouped[day_key]["min"], min_t)
            grouped[day_key]["max"] = max(grouped[day_key]["max"], max_t)
    days = sorted(grouped.values(), key=lambda x: x["date"])
    return days[:5]


@app.route('/api/weather', methods=['GET'])
def weather_api():
    try:
        city = request.args.get('city', '').strip() or None
        lat = request.args.get('lat')
        lon = request.args.get('lon')
        lat_val = float(lat) if lat not in (None, '') else None
        lon_val = float(lon) if lon not in (None, '') else None

        current = _fetch_openweather_current(city=city, lat=lat_val, lon=lon_val)
        coords = current.get("coord", {})
        f_lat = coords.get("lat", lat_val)
        f_lon = coords.get("lon", lon_val)
        forecast = _fetch_openweather_forecast(f_lat, f_lon) if f_lat is not None and f_lon is not None else []

        temp = float(current["main"]["temp"])
        humidity = int(current["main"]["humidity"])
        condition = str(current["weather"][0]["description"])
        advice = _build_weather_advice(temp, humidity, condition)
        warning = None
        if temp > 35 or temp < 5:
            warning = "Extreme temperature conditions may affect crops. Take necessary precautions."

        return jsonify({
            "location": {
                "city": current.get("name"),
                "country": current.get("sys", {}).get("country"),
                "lat": f_lat,
                "lon": f_lon,
            },
            "current": {
                "temp": temp,
                "feels_like": float(current["main"]["feels_like"]),
                "humidity": humidity,
                "pressure": int(current["main"]["pressure"]),
                "wind": float(current.get("wind", {}).get("speed", 0.0)),
                "condition": condition,
                "icon": current["weather"][0]["icon"],
            },
            "forecast": forecast,
            "advice": advice,
            "warning": warning,
        })
    except Exception as e:
        print(f"[WEATHER API] Error: {e}")
        return jsonify({"error": str(e)}), 400


@app.route('/api/fertilizer-predict', methods=['POST'])
def fertilizer_predict_api():
    try:
        data = request.get_json(force=True)
        nitrogen = float(data.get('nitrogen', 0))
        potassium = float(data.get('potassium', 0))
        phosphorous = float(data.get('phosphorous', 0))
        raw_result = model.predict(np.array([[nitrogen, potassium, phosphorous]]))
        fertilizer = _map_fertilizer_label(int(raw_result[0]))
        return jsonify({'fertilizer': fertilizer})
    except Exception as e:
        print(f"[FERTILIZER API] Error: {e}")
        return jsonify({'error': 'Unable to predict fertilizer'}), 400


@app.route("/api/fertilizer-calculator", methods=["POST"])
def fertilizer_calculator_api():
    """
    Compute fertilizer product amounts (kg) from NPK mix percentages (e.g. 19-19-19 => 19 each),
    per-unit nutrient requirements, and field area. Same math as the legacy HTML calculator.
    """
    try:
        d = request.get_json(force=True)
        n_pct = float(d.get("n_ratio_percent", 0)) / 100.0
        p_pct = float(d.get("p_ratio_percent", 0)) / 100.0
        k_pct = float(d.get("k_ratio_percent", 0)) / 100.0
        rn = float(d.get("required_n", 0))
        rp = float(d.get("required_p", 0))
        rk = float(d.get("required_k", 0))
        area = float(d.get("area", 0))
        if n_pct <= 0 or p_pct <= 0 or k_pct <= 0 or area <= 0:
            return jsonify({"error": "n_ratio_percent, p_ratio_percent, k_ratio_percent, and area must be > 0"}), 400
        n_kg = int((rn / n_pct) * area)
        p_kg = int((rp / p_pct) * area)
        k_kg = int((rk / k_pct) * area)
        return jsonify({"n_kg": n_kg, "p_kg": p_kg, "k_kg": k_kg})
    except Exception as e:
        print(f"[FERTILIZER CALC API] Error: {e}")
        return jsonify({"error": "Invalid input; send JSON with n_ratio_percent, p_ratio_percent, k_ratio_percent, required_n, required_p, required_k, area"}), 400


@app.route("/crop-recommendation", methods=["GET", "POST"])
def crop_recommendation():
    recommendation = None
    error = None

    if request.method == "POST":
        try:
            N = float(request.form.get("N", "0") or "0")
            P = float(request.form.get("P", "0") or "0")
            K = float(request.form.get("K", "0") or "0")
            temperature = float(request.form.get("temperature", "0") or "0")
            humidity = float(request.form.get("humidity", "0") or "0")
            ph = float(request.form.get("ph", "0") or "0")
            rainfall = float(request.form.get("rainfall", "0") or "0")

            recommendation = _recommend_crop(
                [N, P, K, temperature, humidity, ph, rainfall]
            )
            if recommendation is None:
                error = "Crop recommendation model is not available. Please try again later."
        except ValueError:
            error = "Please enter valid numeric values in all fields."

    return render_template(
        "crop_recommendation.html",
        recommendation=recommendation,
        error=error,
    )


@app.route("/api/crop-recommendation", methods=["POST"])
def crop_recommendation_api():
    try:
        data = request.get_json(force=True)
        features = [
            float(data.get("N", 0)),
            float(data.get("P", 0)),
            float(data.get("K", 0)),
            float(data.get("temperature", 0)),
            float(data.get("humidity", 0)),
            float(data.get("ph", 0)),
            float(data.get("rainfall", 0)),
        ]
        recommendation = _recommend_crop(features)
        if recommendation is None:
            return jsonify({"error": "Model unavailable"}), 503
        return jsonify({"recommendation": recommendation})
    except Exception as e:
        print(f"[CROP API] Error: {e}")
        return jsonify({"error": "Invalid input data"}), 400

def _detect_query_language(text):
    """Detect if user query is primarily in Telugu or Hindi (Devanagari). Returns 'te', 'hi', or 'en'."""
    if not text or not text.strip():
        return 'en'
    telugu_count = sum(1 for c in text if '\u0C00' <= c <= '\u0C7F')
    hindi_count = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    total_relevant = len([c for c in text if c.strip()])
    if total_relevant == 0:
        return 'en'
    # If a significant portion is in Telugu or Hindi, use that language
    if telugu_count >= 2 and telugu_count >= hindi_count:
        return 'te'
    if hindi_count >= 2 and hindi_count >= telugu_count:
        return 'hi'
    return 'en'


def _gemini_text(parts, temperature=0.3, max_output_tokens=1000, model_name="gemini-2.5-flash", retries=0):
    """Safely call Gemini model and return response text."""
    if gemini_client is None:
        raise RuntimeError("Gemini is not configured. Set a valid GEMINI_API_KEY in .env")

    last_error = None
    for attempt in range(retries + 1):
        try:
            if GENAI_SDK_V2:
                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents=parts,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                    ),
                )
                return (getattr(response, "text", "") or "").strip()
            response = gemini_client.generate_content(
                parts,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                },
            )
            return (getattr(response, "text", "") or "").strip()
        except Exception as e:
            last_error = e
            message = str(e)
            if ("503" in message or "UNAVAILABLE" in message) and attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
    raise last_error


def _groq_chat_fallback(system_prompt, user_message, kcc_context=None, temperature=0.3, max_tokens=1000):
    """Fallback chat generation using Groq (OpenAI-compatible endpoint)."""
    if not GROQ_API_KEY:
        raise RuntimeError("Groq fallback not configured. Set GROQ_API_KEY in .env")

    messages = [{"role": "system", "content": system_prompt}]
    if kcc_context:
        kb_instruction = (
            "You also have access to trusted reference answers from an agricultural "
            "knowledge base (Kisan Call Centre). Use these as your primary factual source, "
            "then rewrite, combine and enhance them into a single clear answer for the farmer.\n\n"
            "[REFERENCE_ANSWERS]\n"
            f"{kcc_context}"
        )
        messages.append({"role": "system", "content": kb_instruction})
    messages.append({"role": "user", "content": user_message})

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": GROQ_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=45,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Groq fallback error {response.status_code}: {response.text}")
    data = response.json()
    return (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()


@app.route("/disease-detection", methods=["GET", "POST"])
def disease_detection():
    diagnosis = None
    error = None
    uploaded_image_data = None

    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            error = "Please upload a plant or leaf photo."
        else:
            try:
                image_bytes = file.read()
                if not image_bytes:
                    error = "Uploaded file is empty. Please try again with a valid image."
                else:
                    mime_type = file.mimetype or "image/jpeg"
                    b64_image = base64.b64encode(image_bytes).decode("utf-8")
                    uploaded_image_data = f"data:{mime_type};base64,{b64_image}"

                    system_prompt = (
                        "You are an experienced plant pathologist helping small farmers. "
                        "Look carefully at the plant/leaf image and diagnose any visible disease, "
                        "nutrient deficiency, or pest damage. If the plant looks mostly healthy, say so.\n\n"
                        "Your answer must be clear, safe, and practical. Avoid giving medicine-like advice for humans.\n\n"
                        "FORMAT (respond in English):\n"
                        "- Likely problem (or say 'No obvious disease')\n"
                        "- Main visual clues you used\n"
                        "- Immediate actions for the farmer\n"
                        "- Long-term prevention tips"
                    )

                    diagnosis_prompt = (
                        f"{system_prompt}\n\n"
                        "Please analyze this plant image and give your diagnosis and advice."
                    )
                    if not GENAI_SDK_V2 or types is None:
                        raise RuntimeError("Image diagnosis currently requires google-genai SDK.")
                    diagnosis = _gemini_text(
                        [
                            diagnosis_prompt,
                            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                        ],
                        temperature=0.2,
                        max_output_tokens=600,
                    )
            except Exception as e:
                print(f"[DISEASE DETECTION] Error: {e}")
                error = "Something went wrong while analyzing the image. Please try again."

    return render_template(
        "disease_detection.html",
        diagnosis=diagnosis,
        error=error,
        image_data=uploaded_image_data,
    )


@app.route("/api/disease-detection", methods=["POST"])
def disease_detection_api():
    try:
        file = request.files.get("image")
        if not file or file.filename == "":
            return jsonify({"error": "Please upload a plant or leaf photo."}), 400
        image_bytes = file.read()
        if not image_bytes:
            return jsonify({"error": "Uploaded file is empty."}), 400

        mime_type = file.mimetype or "image/jpeg"
        system_prompt = (
            "You are an experienced plant pathologist helping small farmers. "
            "Look carefully at the plant/leaf image and diagnose any visible disease, "
            "nutrient deficiency, or pest damage. If the plant looks mostly healthy, say so.\n\n"
            "Your answer must be clear, safe, and practical."
        )
        diagnosis = _gemini_text(
            [
                system_prompt,
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ],
            temperature=0.2,
            max_output_tokens=600,
        )
        return jsonify({"diagnosis": diagnosis})
    except Exception as e:
        print(f"[DISEASE API] Error: {e}")
        return jsonify({"error": "Unable to analyze image right now"}), 500


@app.route('/chat', methods=['POST'])
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data['message']
        override_lang = data.get("lang")
        override_lang_label = data.get("lang_label")

        # Use explicit language from client when provided, otherwise auto-detect
        lang = override_lang or _detect_query_language(user_message)

        # Retrieve relevant Q&A entries from KCC CSV (if available)
        kcc_context = _get_kcc_context(user_message)

        # Language instruction for response
        if lang in ('en', 'te', 'hi'):
            lang_instruction = {
                'en': 'Respond ONLY in English.',
                'te': 'Respond ONLY in Telugu (తెలుగు). Use Telugu script for the entire response.',
                'hi': 'Respond ONLY in Hindi (हिन्दी). Use Devanagari script for the entire response.',
            }[lang]
        else:
            # For other languages (e.g., Bengali, Tamil, etc.) use a generic instruction
            display_name = override_lang_label or lang or "the farmer\'s language"
            lang_instruction = (
                f"Respond ONLY in {display_name}. Use the native script of {display_name} "
                f"for the entire response. Do not mix in other languages."
            )

        # Create a farming-focused prompt that gives detailed practical answers.
        system_prompt = f"""You are a knowledgeable farming assistant.
        {lang_instruction}

        STRICT RULES:
        - Answer only questions about agriculture, farming, crops, and soil.
        - Keep responses practical, accurate, and farmer-friendly.
        - Use clear, simple language that farmers can easily understand.
        - Do not give irrelevant generic answers.

        DETAIL REQUIREMENTS:
        - Give a detailed answer with enough depth to be useful in real farming decisions.
        - Include "why" and "how", not just one-line advice.
        - Prefer region-aware guidance for India when the question is generic.
        - If exact values depend on local conditions, state assumptions clearly.

        FORMAT:
        - Use bullet points (• or -) only, no long paragraphs.
        - Provide 8-14 bullets when possible.
        - Organize using these section labels as bullets:
          - Problem understanding
          - Recommended crops/practices
          - Step-by-step action plan
          - Input quantity/dose/range (seed, fertilizer, water) when applicable
          - Best time/season
          - Risks and precautions
          - Low-cost alternatives
          - Expected result and timeline
        - End with 1 short follow-up bullet asking for missing local details
          (district/state, soil type, season, irrigation availability) to refine advice."""

        prompt_blocks = [system_prompt]

        # If we found relevant rows in KCC_raw_data.csv, feed them as reference material
        if kcc_context:
            kb_instruction = (
                "You also have access to trusted reference answers from an agricultural "
                "knowledge base (Kisan Call Centre). Use these as your primary factual source, "
                "then rewrite, combine and enhance them into a single clear answer for the farmer, "
                "following all the rules above.\n\n"
                "[REFERENCE_ANSWERS]\n"
                f"{kcc_context}"
            )
            prompt_blocks.append(kb_instruction)

        prompt_blocks.append(f"[FARMER_QUESTION]\n{user_message}")
        try:
            bot_text = _gemini_text(prompt_blocks, temperature=0.3, max_output_tokens=1000, retries=2)
        except Exception as gemini_error:
            print(f"[CHAT] Gemini failed, trying Groq fallback: {gemini_error}")
            bot_text = _groq_chat_fallback(
                system_prompt=system_prompt,
                user_message=user_message,
                kcc_context=kcc_context,
                temperature=0.3,
                max_tokens=1000,
            )

        return jsonify({
            'response': bot_text,
            'image_url': ""
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'response': "I apologize, but I encountered an error. Please try again."})

@app.route('/favicon.ico')
def favicon():
    return '', 204


@app.route('/health', methods=['GET'])
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/transcribe', methods=['POST'])
@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    temp_path = None
    try:
        if gemini_client is None:
            return jsonify({'error': 'Gemini API key is missing/invalid. Set GEMINI_API_KEY in .env and restart app.'}), 400
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        language = request.form.get('language', '')  # Optional hint from client; Whisper will auto-detect
        
        # Use a unique filename to avoid permission issues
        temp_path = f'temp_{random.randint(1000, 9999)}.webm'
        audio_file.save(temp_path)
        
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) < 100:
            return jsonify({'error': 'Audio too short or empty'}), 400
        
        # Transcribe audio with Gemini (multimodal input).
        with open(temp_path, 'rb') as audio_data:
            audio_bytes = audio_data.read()
        if not GENAI_SDK_V2 or types is None:
            return jsonify({'error': 'Audio transcription requires google-genai SDK. Please install/upgrade dependencies.'}), 400
        transcript_text = _gemini_text(
            [
                types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type=audio_file.mimetype or "audio/webm",
                ),
                (
                    "Transcribe this farmer audio message into plain text only. "
                    "Do not add explanations. If unclear, provide your best clean transcript. "
                    f"Language hint: {language or 'auto-detect'}."
                ),
            ],
            temperature=0.0,
            max_output_tokens=500,
            model_name="gemini-2.5-flash",
            retries=2,
        )
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return jsonify({'text': transcript_text, 'language': language or None})
        
    except Exception as e:
        print(f"Transcription Error: {str(e)}")
        if temp_path and os.path.exists(temp_path):
            try: os.remove(temp_path)
            except: pass
        msg = str(e)
        if "API_KEY_INVALID" in msg or "API key not valid" in msg:
            return jsonify({'error': 'Gemini API key is invalid. Update GEMINI_API_KEY in .env and restart app.'}), 400
        if "503" in msg or "UNAVAILABLE" in msg:
            return jsonify({'error': 'Gemini is temporarily overloaded. Please retry in 20-60 seconds.'}), 503
        return jsonify({'error': msg}), 500

def _synthesize_tts_audio(text, lang_override=None):
    if not text or not str(text).strip():
        return None, 'No text provided'
    text = str(text).strip()
    if lang_override:
        tts_lang = lang_override.lower()[:2]
        if tts_lang not in ("en", "hi", "te"):
            tts_lang = "en"
    else:
        lang = _detect_query_language(text)
        tts_lang = {"en": "en", "hi": "hi", "te": "te"}.get(lang, "en")
    audio_bytes = io.BytesIO()
    gTTS(text=text, lang=tts_lang).write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes, None


@app.route('/text-to-speech', methods=['POST', 'GET'])
@app.route('/api/text-to-speech', methods=['POST', 'GET'])
def text_to_speech():
    try:
        lang_override = None
        if request.method == 'GET':
            text = request.args.get('text', '')
            lang_override = request.args.get('lang') or request.args.get('language')
        else:
            data = request.get_json(silent=True) or {}
            text = data.get('text', '')
            lang_override = data.get('lang') or data.get('language')

        audio_bytes, err = _synthesize_tts_audio(text, lang_override=lang_override)
        if err:
            return jsonify({'error': err}), 400

        return send_file(
            audio_bytes,
            mimetype='audio/mpeg',
            as_attachment=False,
            download_name='response.mp3',
        )

    except Exception as e:
        print(f"TTS Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true", "yes")
    app.run(host="0.0.0.0", port=port, debug=debug)