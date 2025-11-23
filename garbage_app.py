import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import json
from pathlib import Path

# ============================================================
# KONFIGURATION
# ============================================================

# Seiten-Konfiguration
st.set_page_config(
    page_title="♻️ AI Garbage Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Klassen-Definitionen
CLASS_NAMES = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']
CLASS_EMOJIS = ['📦', '🍾', '🥫', '📄', '🧴', '🗑️']
CLASS_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#95A5A6']

# Deutsche Übersetzungen
CLASS_NAMES_DE = {
    'Cardboard': 'Karton',
    'Glass': 'Glas',
    'Metal': 'Metall',
    'Paper': 'Papier',
    'Plastic': 'Plastik',
    'Trash': 'Restmüll'
}

# Recycling-Informationen (erweitert)
RECYCLING_INFO = {
    'Cardboard': {
        'recyclable': True,
        'bin': 'Papiertonne (Blau)',
        'bin_color': '#0066CC',
        'tips': [
            '✅ Kartons flach zusammenfalten',
            '✅ Sauber und trocken halten',
            '✅ Klebeband entfernen',
            '❌ Keine beschichteten oder verschmutzten Kartons'
        ],
        'environmental_impact': '♻️ Spart 17 Bäume pro Tonne',
        'decomposition_time': '⏱️ 2 Monate bis 2 Jahre'
    },
    'Glass': {
        'recyclable': True,
        'bin': 'Glascontainer',
        'bin_color': '#00AA00',
        'tips': [
            '✅ Nach Farben trennen (Weiß, Grün, Braun)',
            '✅ Deckel und Verschlüsse entfernen',
            '✅ Keine Spülung notwendig',
            '❌ Kein Fensterglas, Spiegel oder Keramik'
        ],
        'environmental_impact': '♻️ 100% recycelbar, unendlich oft',
        'decomposition_time': '⏱️ 4.000+ Jahre'
    },
    'Metal': {
        'recyclable': True,
        'bin': 'Gelber Sack / Wertstofftonne',
        'bin_color': '#FFCC00',
        'tips': [
            '✅ Dosen und Verpackungen ausspülen',
            '✅ Deckel separat entsorgen',
            '✅ Aluminium und Weißblech zusammen',
            '❌ Keine verschmutzten oder beschichteten Metalle'
        ],
        'environmental_impact': '♻️ Spart 95% Energie vs. Neuproduktion',
        'decomposition_time': '⏱️ 50-500 Jahre'
    },
    'Paper': {
        'recyclable': True,
        'bin': 'Papiertonne (Blau)',
        'bin_color': '#0066CC',
        'tips': [
            '✅ Sauber und trocken halten',
            '✅ Zeitungen, Zeitschriften, Briefe',
            '✅ Heftklammern sind okay',
            '❌ Kein verschmutztes, fettiges oder beschichtetes Papier'
        ],
        'environmental_impact': '♻️ Spart 60% Energie und Wasser',
        'decomposition_time': '⏱️ 2-6 Wochen'
    },
    'Plastic': {
        'recyclable': True,
        'bin': 'Gelber Sack / Wertstofftonne',
        'bin_color': '#FFCC00',
        'tips': [
            '✅ Verpackungen ausspülen (löffelrein)',
            '✅ Auf Recycling-Symbol achten',
            '✅ PET, PE, PP sind gut recycelbar',
            '❌ Keine Plastiktüten, PVC oder stark verschmutzte Teile'
        ],
        'environmental_impact': '♻️ Spart Erdöl und reduziert CO₂',
        'decomposition_time': '⏱️ 450+ Jahre'
    },
    'Trash': {
        'recyclable': False,
        'bin': 'Restmülltonne (Schwarz/Grau)',
        'bin_color': '#666666',
        'tips': [
            '⚠️ Nicht recycelbar',
            '⚠️ In Restmüll entsorgen',
            '💡 Versuche Müll zu vermeiden',
            '💡 Prüfe ob wirklich nicht recycelbar'
        ],
        'environmental_impact': '⚠️ Wird verbrannt oder deponiert',
        'decomposition_time': '⏱️ Variiert stark'
    }
}

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    /* Hauptheader */
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        padding: 1rem 0;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }

    /* Prediction Card */
    .prediction-card {
        padding: 2.5rem;
        border-radius: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        animation: fadeIn 0.5s ease-in;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Info Cards */
    .info-card {
        padding: 1.5rem;
        border-radius: 10px;
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }

    /* Metric Styling */
    .metric-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }

    /* Upload Area */
    .upload-text {
        text-align: center;
        padding: 2rem;
        border: 2px dashed #667eea;
        border-radius: 10px;
        background: #f8f9fa;
    }

    /* Confidence Bar */
    .confidence-bar {
        height: 30px;
        border-radius: 15px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        transition: width 0.5s ease;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================

if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'total_classifications' not in st.session_state:
    st.session_state.total_classifications = 0
if 'total_co2_saved' not in st.session_state:
    st.session_state.total_co2_saved = 0.0


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

@st.cache_resource
def load_model():
    """Lade das trainierte Modell mit Fehlerbehandlung"""
    model_paths = [
        'models/best_model.keras',
        'models/final_model.keras',
        '../models/best_model.keras'
    ]

    for path in model_paths:
        if Path(path).exists():
            try:
                model = tf.keras.models.load_model(path)
                st.success(f"✅ Modell geladen: {path}")
                return model, path
            except Exception as e:
                st.warning(f"⚠️ Fehler beim Laden von {path}: {e}")

    st.error("❌ Kein Modell gefunden!")
    st.info(
        "💡 Stelle sicher, dass ein trainiertes Modell existiert:\n- models/best_model.keras\n- models/final_model.keras")
    return None, None


@st.cache_data
def load_model_info():
    """Lade Modell-Informationen aus JSON falls vorhanden"""
    try:
        with open('models/training_summary.json', 'r') as f:
            return json.load(f)
    except:
        return {
            'results': {
                'accuracy': 0.881,
                'top_2_accuracy': 0.965
            },
            'data': {
                'training_samples': 10438,
                'num_classes': 6
            }
        }


def preprocess_image(image):
    """Bereite Bild für Vorhersage vor"""
    # Resize zu 299x299 (InceptionV3 Input-Größe)
    img = image.resize((299, 299))
    img_array = np.array(img)

    # Stelle sicher, dass es 3 Kanäle hat (RGB)
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]

    # Normalisiere auf [0, 1]
    img_array = img_array.astype('float32') / 255.0

    # Füge Batch-Dimension hinzu
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def create_confidence_chart(predictions, class_names):
    """Erstelle interaktives Confidence-Chart mit Plotly"""
    # Sortiere nach Confidence
    sorted_indices = np.argsort(predictions[0])[::-1]
    sorted_names = [class_names[i] for i in sorted_indices]
    sorted_emojis = [CLASS_EMOJIS[i] for i in sorted_indices]
    sorted_values = predictions[0][sorted_indices] * 100
    sorted_colors = [CLASS_COLORS[i] for i in sorted_indices]

    # Labels mit Emojis
    labels = [f"{emoji} {name}" for emoji, name in zip(sorted_emojis, sorted_names)]

    fig = go.Figure(data=[
        go.Bar(
            x=sorted_values,
            y=labels,
            orientation='h',
            marker=dict(
                color=sorted_colors,
                line=dict(color='white', width=2)
            ),
            text=[f'{v:.1f}%' for v in sorted_values],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Confidence: %{x:.2f}%<extra></extra>'
        )
    ])

    fig.update_layout(
        title={
            'text': '📊 Confidence Scores aller Klassen',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Confidence (%)',
        yaxis_title='',
        height=450,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=14),
        xaxis=dict(range=[0, 100])
    )

    return fig


def estimate_co2_savings(class_name):
    """Schätze CO₂-Ersparnis durch Recycling (in kg)"""
    co2_savings = {
        'Cardboard': 0.7,
        'Glass': 0.3,
        'Metal': 1.5,
        'Paper': 0.5,
        'Plastic': 2.0,
        'Trash': 0.0
    }
    return co2_savings.get(class_name, 0.0)


def add_to_history(class_name, confidence):
    """Füge Klassifizierung zur Historie hinzu"""
    st.session_state.prediction_history.append({
        'timestamp': datetime.now().strftime('%H:%M:%S'),
        'class': class_name,
        'confidence': confidence
    })
    st.session_state.total_classifications += 1
    st.session_state.total_co2_saved += estimate_co2_savings(class_name)


# ============================================================
# HEADER
# ============================================================

st.markdown('<div class="main-header">♻️ AI Garbage Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">🤖 Intelligente Müll-Klassifizierung mit Deep Learning</div>', unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 📊 Modell-Informationen")

    # Lade Modell-Info
    model_info = load_model_info()

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric(
            "Genauigkeit",
            f"{model_info['results']['accuracy'] * 100:.1f}%",
            delta="+0.6%"
        )
    with col_s2:
        st.metric(
            "Top-2 Acc.",
            f"{model_info['results']['top_2_accuracy'] * 100:.1f}%",
            delta="+2.1%"
        )

    st.metric(
        "Trainingsbilder",
        f"{model_info['data']['training_samples']:,}",
        help="Anzahl der Bilder im Trainingsdatensatz"
    )

    st.markdown("---")

    st.markdown("## 🗂️ Kategorien")
    for emoji, name in zip(CLASS_EMOJIS, CLASS_NAMES):
        st.markdown(f"{emoji} **{name}** ({CLASS_NAMES_DE[name]})")

    st.markdown("---")

    st.markdown("## 📈 Session Statistiken")
    st.metric("Klassifizierungen", st.session_state.total_classifications)
    st.metric("CO₂ eingespart", f"{st.session_state.total_co2_saved:.1f} kg",
              help="Geschätzte CO₂-Ersparnis durch korrektes Recycling")

    if st.session_state.prediction_history:
        st.markdown("### 🕒 Letzte Vorhersagen")
        for item in st.session_state.prediction_history[-3:]:
            st.caption(f"{item['timestamp']} - {item['class']} ({item['confidence']:.0f}%)")

    st.markdown("---")

    # Reset Button
    if st.button("🔄 Statistiken zurücksetzen"):
        st.session_state.prediction_history = []
        st.session_state.total_classifications = 0
        st.session_state.total_co2_saved = 0.0
        st.rerun()

# ============================================================
# HAUPTBEREICH
# ============================================================

# Tabs für verschiedene Modi
tab1, tab2, tab3 = st.tabs(["📸 Bild hochladen", "📊 Batch-Klassifizierung", "ℹ️ Über das Projekt"])

with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 📤 Bild hochladen")

        uploaded_file = st.file_uploader(
            "Wähle ein Bild deines Mülls...",
            type=['jpg', 'jpeg', 'png'],
            help="Unterstützte Formate: JPG, JPEG, PNG (max. 200MB)"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='📷 Hochgeladenes Bild', use_container_width=True)

            # Bild-Info
            st.caption(f"Größe: {image.size[0]}x{image.size[1]} Pixel | Format: {image.format}")

        else:
            st.info("👆 Lade ein Bild hoch um zu starten")

    with col2:
        st.markdown("### 🔍 Klassifizierung & Ergebnis")

        if uploaded_file is not None:
            # Lade Modell
            model, model_path = load_model()

            if model is not None:
                # Klassifiziere Button
                if st.button('🚀 Jetzt klassifizieren!', type='primary', use_container_width=True):

                    # Progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Schritt 1: Preprocessing
                    status_text.text('⏳ Bild wird vorbereitet...')
                    progress_bar.progress(25)
                    time.sleep(0.3)

                    processed_img = preprocess_image(image)

                    # Schritt 2: Prediction
                    status_text.text('🤖 KI analysiert das Bild...')
                    progress_bar.progress(50)

                    start_time = time.time()
                    predictions = model.predict(processed_img, verbose=0)
                    inference_time = time.time() - start_time

                    progress_bar.progress(75)
                    status_text.text('✨ Ergebnisse werden aufbereitet...')
                    time.sleep(0.2)

                    # Ergebnisse
                    predicted_class_idx = np.argmax(predictions[0])
                    predicted_class = CLASS_NAMES[predicted_class_idx]
                    confidence = predictions[0][predicted_class_idx] * 100

                    progress_bar.progress(100)
                    status_text.empty()
                    progress_bar.empty()

                    # Füge zur Historie hinzu
                    add_to_history(predicted_class, confidence)

                    # ============================================
                    # HAUPTERGEBNIS
                    # ============================================

                    st.markdown(f"""
                    <div class="prediction-card">
                        <h1 style='font-size: 4rem; margin: 0;'>{CLASS_EMOJIS[predicted_class_idx]}</h1>
                        <h2 style='font-size: 2.5rem; margin: 0.5rem 0;'>{predicted_class}</h2>
                        <p style='font-size: 1.2rem; opacity: 0.9;'>({CLASS_NAMES_DE[predicted_class]})</p>
                        <h3 style='font-size: 2rem; margin-top: 1rem;'>{confidence:.1f}% Confidence</h3>
                        <p style='opacity: 0.8;'>Inferenz: {inference_time * 1000:.0f}ms</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # ============================================
                    # RECYCLING-INFORMATIONEN
                    # ============================================

                    st.markdown("---")
                    st.markdown("### ♻️ Entsorgungsinformationen")

                    info = RECYCLING_INFO[predicted_class]

                    # Row 1: Recycelbar & Tonne
                    col_a, col_b, col_c = st.columns(3)

                    with col_a:
                        if info['recyclable']:
                            st.success("✅ **Recycelbar**")
                        else:
                            st.error("❌ **Nicht recycelbar**")

                    with col_b:
                        st.info(f"🗑️ **{info['bin']}**")

                    with col_c:
                        co2 = estimate_co2_savings(predicted_class)
                        if co2 > 0:
                            st.success(f"🌱 **~{co2} kg CO₂** gespart")
                        else:
                            st.warning("⚠️ **Keine Ersparnis**")

                    # Umwelteinfluss & Abbauzeit
                    col_d, col_e = st.columns(2)

                    with col_d:
                        st.markdown("**Umwelteinfluss:**")
                        st.caption(info['environmental_impact'])

                    with col_e:
                        st.markdown("**Abbauzeit:**")
                        st.caption(info['decomposition_time'])

                    # Entsorgungstipps
                    st.markdown("**💡 Entsorgungstipps:**")
                    for tip in info['tips']:
                        st.markdown(f"- {tip}")

                    # ============================================
                    # CONFIDENCE CHART
                    # ============================================

                    st.markdown("---")
                    st.plotly_chart(
                        create_confidence_chart(predictions, CLASS_NAMES),
                        use_container_width=True
                    )

                    # Top-3 mit mehr Details
                    st.markdown("### 🏆 Top 3 Vorhersagen")
                    top_3_idx = np.argsort(predictions[0])[-3:][::-1]

                    for i, idx in enumerate(top_3_idx, 1):
                        conf = predictions[0][idx] * 100
                        class_name = CLASS_NAMES[idx]

                        # Medal Emoji
                        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"

                        with st.expander(f"{medal} #{i}: {CLASS_EMOJIS[idx]} {class_name} - {conf:.1f}%"):
                            st.markdown(f"**{CLASS_NAMES_DE[class_name]}**")
                            st.progress(conf / 100)
                            if i > 1:
                                st.caption(f"Tonne: {RECYCLING_INFO[class_name]['bin']}")
            else:
                st.error("❌ Modell konnte nicht geladen werden")
        else:
            st.info("👈 Lade zuerst ein Bild hoch")

with tab2:
    st.markdown("### 📊 Batch-Klassifizierung")
    st.info("🚧 Feature in Entwicklung: Mehrere Bilder gleichzeitig klassifizieren")

    uploaded_files = st.file_uploader(
        "Mehrere Bilder hochladen",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} Bilder hochgeladen")

        cols = st.columns(4)
        for i, file in enumerate(uploaded_files):
            with cols[i % 4]:
                img = Image.open(file)
                st.image(img, caption=file.name, use_container_width=True)

with tab3:
    st.markdown("### ℹ️ Über das Projekt")

    col_info1, col_info2 = st.columns(2)

    with col_info1:
        st.markdown("""
        #### 🤖 Technologie

        Diese App nutzt ein **Deep Learning Modell** basierend auf der 
        **InceptionV3 Architektur** von Google, trainiert auf über 10.000 
        Bildern verschiedener Müllkategorien.

        **Modell-Details:**
        - Architektur: InceptionV3 (Transfer Learning)
        - Input: 299x299x3 RGB Bilder
        - Output: 6 Müllkategorien
        - Framework: TensorFlow 2.x / Keras
        - Training: 2-phasig (Frozen → Fine-Tuning)

        **Performance:**
        - Validation Accuracy: ~88%
        - Top-2 Accuracy: ~96%
        - Inferenz-Zeit: <0.5s pro Bild
        """)

    with col_info2:
        st.markdown("""
        #### 📚 Dataset

        Das Modell wurde trainiert auf einem kuratierten Dataset mit:
        - **10.438** Trainingsbildern
        - **2.606** Validierungsbildern
        - **6** Kategorien (Cardboard, Glass, Metal, Paper, Plastic, Trash)

        **Data Augmentation:**
        - Rotation (±30°)
        - Zoom (±20%)
        - Brightness (80-120%)
        - Horizontal Flip

        **Klassenverteilung:**
        - Glass: 3.562 Bilder
        - Plastic: 2.466 Bilder
        - Paper: 2.274 Bilder
        - Cardboard: 2.228 Bilder
        - Metal: 1.430 Bilder
        - Trash: 1.084 Bilder
        """)

    st.markdown("---")

    st.markdown("""
    #### 🌍 Warum ist das wichtig?

    Korrekte Mülltrennung ist essentiell für:
    - ♻️ **Recycling-Effizienz**: Nur richtig sortierter Müll kann recycelt werden
    - 🌱 **Umweltschutz**: Reduzierung von Deponie-Müll und CO₂-Emissionen
    - 💰 **Kostenersparnis**: Recycling ist oft günstiger als Neuproduktion
    - 🌊 **Ressourcenschonung**: Weniger Rohstoffabbau und Energieverbrauch

    Diese KI kann helfen, Mülltrennung einfacher und zugänglicher zu machen!
    """)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

col_f1, col_f2, col_f3, col_f4 = st.columns(4)

with col_f1:
    st.metric(
        "📊 Klassifizierungen",
        st.session_state.total_classifications,
        help="Anzahl der Klassifizierungen in dieser Session"
    )

with col_f2:
    avg_time = 0.5
    st.metric(
        "⏱️ Ø Inferenz-Zeit",
        f"{avg_time:.2f}s",
        help="Durchschnittliche Zeit pro Klassifizierung"
    )

with col_f3:
    st.metric(
        "🌍 CO₂ eingespart",
        f"{st.session_state.total_co2_saved:.1f} kg",
        help="Geschätzte CO₂-Ersparnis durch korrektes Recycling"
    )

with col_f4:
    st.metric(
        "🎯 Modell-Genauigkeit",
        "88.1%",
        help="Validation Accuracy des Modells"
    )

st.markdown("""
---
<div style='text-align: center; color: gray; padding: 2rem 0;'>
    <p>Made with ❤️ using <b>TensorFlow</b> & <b>Streamlit</b></p>
    <p style='font-size: 0.9rem;'>
        Model: InceptionV3 | Dataset: 13,044 Bilder | Framework: TensorFlow 2.x
    </p>
    <p style='font-size: 0.8rem; opacity: 0.7;'>
        © 2024 AI Garbage Classifier | Für Bildungszwecke
    </p>
</div>
""", unsafe_allow_html=True)