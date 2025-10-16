import streamlit as st
from PIL import Image
import time
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import os
from huggingface_hub import hf_hub_download

# config de la page
st.set_page_config(
    page_title="Meat Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# déco css
st.markdown("""
<style>
    /* ===== FOND PRINCIPAL ===== */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-attachment: fixed;
    }
    
    /* Supprimer TOUTES les barres blanches */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
        display: none !important;
    }
    
    .main .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    
    .st-emotion-cache-z5fcl4 {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    
    .st-emotion-cache-18ni7ap {
        display: none !important;
    }
    
    .css-18e3th9, .css-1dp5vir {
        display: none !important;
    }
    
    /* ===== TITRE PRINCIPAL ===== */
    .main-title {
        text-align: center;
        color: #FFFFFF;
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
        border: 3px solid rgba(255, 255, 255, 0.2);
        text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.5);
        letter-spacing: 2px;
    }
    
    .subtitle {
        text-align: center;
        color: #FFFFFF;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 2.5rem;
        background: rgba(30, 60, 114, 0.95);
        padding: 1.2rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.3);
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.4);
    }
    
    /* ===== BOÎTES DE PRÉDICTION ===== */
    .prediction-fresh {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 20px;
        color: #FFFFFF;
        font-size: 2rem;
        font-weight: 800;
        text-align: center;
        margin: 1.5rem 0;
        border: 4px solid #0a7a6e;
        box-shadow: 0 8px 25px rgba(17, 153, 142, 0.5);
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        animation: pulse-fresh 2s ease-in-out infinite;
    }
    
    @keyframes pulse-fresh {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    .prediction-spoiled {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 2rem;
        border-radius: 20px;
        color: #FFFFFF;
        font-size: 2rem;
        font-weight: 800;
        text-align: center;
        margin: 1.5rem 0;
        border: 4px solid #c0262b;
        box-shadow: 0 8px 25px rgba(235, 51, 73, 0.5);
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        animation: pulse-spoiled 2s ease-in-out infinite;
    }
    
    @keyframes pulse-spoiled {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    /* ===== CONTENEURS PRINCIPAUX ===== */
    .container {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        margin: 2rem 0;
        border: 2px solid rgba(255, 255, 255, 0.5);
    }
    
    .section-title {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: #FFFFFF;
        font-size: 1.8rem;
        font-weight: 700;
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        border-left: 6px solid #38ef7d;
        box-shadow: 0 5px 15px rgba(30, 60, 114, 0.4);
    }
    
    /* ===== INSTRUCTIONS ===== */
    .instructions {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        border: 3px solid #ff9a76;
        box-shadow: 0 5px 20px rgba(252, 182, 159, 0.4);
        color: #2c3e50;
        font-weight: 600;
        font-size: 1.1rem;
        line-height: 1.8;
    }
    
    .instructions h4 {
        color: #e74c3c;
        font-weight: 800;
        margin-bottom: 1rem;
    }
    
    /* ===== INFO BLOCKS ===== */
    .info-block {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: #FFFFFF;
        padding: 2rem;
        border-radius: 15px;
        border-left: 6px solid #3498db;
        margin: 1.5rem 0;
        box-shadow: 0 5px 20px rgba(44, 62, 80, 0.4);
        font-size: 1.05rem;
        line-height: 1.8;
    }
    
    /* ===== UPLOAD SECTION ===== */
    .upload-section {
        background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 100%);
        padding: 2.5rem;
        border-radius: 20px;
        border: 4px dashed #00acc1;
        margin: 2rem 0;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 5px 20px rgba(0, 172, 193, 0.3);
    }
    
    .upload-section:hover {
        border-color: #0097a7;
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0, 172, 193, 0.5);
    }
    
    /* ===== SIDEBAR ===== */
    section[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }
    
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] h4, 
    section[data-testid="stSidebar"] h5, 
    section[data-testid="stSidebar"] h6 {
        color: #FFFFFF !important;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] li {
        color: #E8F4F8 !important;
        font-size: 1rem;
        line-height: 1.7;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #FFFFFF !important;
    }
    
    /* ===== PROGRESS BAR ===== */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%) !important;
        border-radius: 10px;
    }
    
    .stProgress > div > div > div {
        background-color: rgba(255, 255, 255, 0.3) !important;
        border-radius: 10px;
    }
    
    .stProgress {
        height: 30px !important;
        border-radius: 10px !important;
        overflow: hidden !important;
        box-shadow: inset 0 2px 5px rgba(0, 0, 0, 0.2);
    }
    
    /* ===== FOOTER ===== */
    .footer {
        text-align: center;
        color: #FFFFFF;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-top: 3rem;
        font-size: 1rem;
        font-weight: 600;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.3);
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.3);
    }
    
    /* ===== STREAMLIT ELEMENTS ===== */
    .stAlert {
        border-radius: 15px;
        border-width: 2px;
        font-weight: 600;
        font-size: 1.05rem;
        padding: 1.2rem;
    }
    
    /* Success messages */
    .stSuccess {
        background-color: #d4edda !important;
        border-color: #28a745 !important;
        color: #155724 !important;
    }
    
    /* Error messages */
    .stError {
        background-color: #f8d7da !important;
        border-color: #dc3545 !important;
        color: #721c24 !important;
    }
    
    /* Warning messages */
    .stWarning {
        background-color: #fff3cd !important;
        border-color: #ffc107 !important;
        color: #856404 !important;
    }
    
    /* Info messages */
    .stInfo {
        background-color: #d1ecf1 !important;
        border-color: #17a2b8 !important;
        color: #0c5460 !important;
    }
    
    /* ===== HEADINGS IN MAIN ===== */
    .main h1, .main h2, .main h3 {
        color: #1e3c72;
        font-weight: 800;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
    }
    
    /* ===== SPINNER ===== */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
</style>
""", unsafe_allow_html=True)

# dl le modele depuis huggingface
@st.cache_resource
def download_model_from_huggingface():
    """Télécharge le modèle depuis Hugging Face Hub"""
    os.makedirs('models', exist_ok=True)
    model_path = 'models/meat_classifier_model.keras'
    
    # verif if le modèle existe déjà localement
    if not os.path.exists(model_path):
        with st.spinner('🔄 Téléchargement du modèle depuis Hugging Face...'):
            try:
                downloaded_model_path = hf_hub_download(
                    repo_id="soooro/meat-classifier",
                    filename="meat_classifier_model.keras",
                    cache_dir="models"
                )
                
                if downloaded_model_path != model_path:
                    import shutil
                    shutil.copy(downloaded_model_path, model_path)
                
                return model_path
            except Exception as e:
                st.error(f"❌ Erreur lors du téléchargement du modèle depuis Hugging Face: {e}")
                import traceback
                st.error(traceback.format_exc())
                raise
    else:
        return model_path

#  modèle de secours compatible
def create_fallback_model():
    """Crée un modèle simple compatible en cas d'échec du téléchargement"""
    from tensorflow import keras
    
    model = keras.Sequential([
        keras.layers.Input(shape=(224, 224, 3)),
        keras.layers.Rescaling(1./255),
        keras.layers.Conv2D(32, 3, activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(128, 3, activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Charge le modèle depuis higginface
@st.cache_resource
def load_classification_model():
    try:
        model_path = download_model_from_huggingface()
        st.info(f"📦 Chargement du modèle depuis {model_path}...")
        
        try:
            model = load_model(model_path)
            st.success("✅ Modèle chargé avec succès!")
            return model
        except Exception as e1:
            st.warning(f"⚠️ Échec du chargement standard, tentative sans compilation...")
            try:
                model = load_model(model_path, compile=False)
                model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                st.success("✅ Modèle chargé avec succès (sans compilation)!")
                return model
            except Exception as e2:
                st.error(f"❌ Échec de toutes les méthodes de chargement du modèle téléchargé")
                st.warning("🔄 Création d'un modèle de démonstration temporaire...")
                st.info("⚠️ ATTENTION: Ce modèle n'est pas entraîné et donnera des résultats aléatoires!")
                st.info("📝 Pour obtenir des prédictions réelles, veuillez uploader le modèle converti sur Hugging Face")
                
                fallback_model = create_fallback_model()
                st.warning("✓ Modèle de démonstration créé (prédictions non fiables)")
                return fallback_model
                
    except Exception as e:
        st.error(f"❌ Erreur lors du téléchargement du modèle: {e}")
        st.warning("🔄 Création d'un modèle de démonstration temporaire...")
        st.info("⚠️ ATTENTION: Ce modèle n'est pas entraîné et donnera des résultats aléatoires!")
        
        fallback_model = create_fallback_model()
        st.warning("✓ Modèle de démonstration créé (prédictions non fiables)")
        return fallback_model


# barre latérale
with st.sidebar:
    st.image("https://institut-agro-dijon.fr/fileadmin/user_upload/INSTITUT-DIJON-MARQUE-ETAT.svg", width=200)
    st.markdown("## 📊 À propos")
    st.info("""
    Cette application utilise un modèle de **Deep Learning** pour déterminer si une viande est fraîche ou avariée à partir d'une simple image.
    
    🔬 Chargez une image de viande pour obtenir une analyse instantanée.
            
    💻 L'ensemble du code est disponible en open-source sur GitHub : https://github.com/gtom-pandas
    """)
    
    st.markdown("### 🚀 Comment ça marche")
    st.markdown("""
    1. 📤 **Téléchargez** une image de viande
    2. 🤖 **Notre modèle d'IA** analyse l'image
    3. ✅ **Recevez le résultat**: Fraîche ou Avariée
    """)
    
    st.markdown("### 🛠️ Développé avec")
    st.markdown("""
    - 🧠 **TensorFlow** - Deep Learning
    - 🎨 **Streamlit** - Interface Web
    - 🐍 **Python** - Langage de programmation
    - 🤗 **Hugging Face** - Hébergement du modèle
    """)

# Contenu principal
st.markdown('<h1 class="main-title">🥩 MEAT ANALYZER</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">🔬 Système d\'analyse avancé pour déterminer la fraîcheur de la viande par Intelligence Artificielle</p>', unsafe_allow_html=True)

try:
    st.text("⏳ Initialisation du modèle...")
    model = load_classification_model()
    if model is not None:
        model_loaded = True
    else:
        model_loaded = False
        st.error("❌ Le modèle n'a pas pu être chargé (valeur None retournée).")
except Exception as e:
    st.error(f"❌ Exception lors du chargement du modèle: {e}")
    model = None
    model_loaded = False

# section principale
st.markdown('<div class="container">', unsafe_allow_html=True)
st.markdown("## 🔍 Analysez votre échantillon de viande")
st.markdown('<div class="instructions">', unsafe_allow_html=True)
st.markdown("""
#### 📋 Instructions:
- 📸 **Prenez une photo claire** de votre échantillon de viande (bœuf, porc)
- ✨ **Assurez-vous** que l'image est la plus nette possible
- ⬆️ **Téléchargez l'image** ci-dessous pour l'analyse
""")
st.markdown('</div>', unsafe_allow_html=True)

# pour DL les photos à analyser
uploaded_file = st.file_uploader("📁 Choisissez une image de viande à analyser...", type=["jpg", "jpeg", "png"])

# Pour le traitement de l'image et la prédiction
if uploaded_file is not None and model_loaded and model is not None:
    # display de l'image + analyse
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🖼️ Image analysée")
        st.image(uploaded_file, caption='📷 Échantillon de viande', use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Résultats de l'analyse")

        # Animation pendant le traitement
        with st.spinner("🔄 Analyse en cours..."):
            time.sleep(1)
            
            # prepare l'image pour la prédiction
            img = image.load_img(uploaded_file, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # verif à nouveau que model n'est pas None
            if model is not None:
                prediction = model.predict(img_array)
                confidence = prediction[0][0]
                
                # display du résultat avec une barre de confiance
                is_spoiled = confidence >= 0.5
                confidence_pct = confidence * 100 if is_spoiled else (1 - confidence) * 100
                
                if is_spoiled:
                    st.markdown('<div class="prediction-spoiled">⚠️ VIANDE AVARIÉE</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="prediction-fresh">✅ VIANDE FRAÎCHE</div>', unsafe_allow_html=True)
                
                st.markdown(f"### 📈 Niveau de confiance: {confidence_pct:.1f}%")
                st.progress(float(confidence_pct/100))
                
                # Afficher des reco basées sur le résultat
                if is_spoiled:
                    st.error("""
                    **⚠️ Recommandation**: Cette viande présente des signes de détérioration et **ne devrait pas être consommée**.
                    
                    🗑️ Veuillez la jeter de manière appropriée pour éviter tout risque sanitaire.
                    """)
                else:
                    st.success("""
                    **✅ Recommandation**: Cette viande semble être **fraîche et propre à la consommation**.
                    
                    ❄️ N'oubliez pas de la conserver correctement et de la cuisiner à une température adéquate (>70°C).
                    """)
            else:
                st.error("❌ Le modèle n'est pas disponible pour faire des prédictions.")
elif uploaded_file is not None and not model_loaded:
    st.error("❌ Le modèle n'a pas pu être chargé. Veuillez réessayer plus tard.")

# infos complémentaires
if not uploaded_file:
    st.markdown("### 📸 Exemples de classification")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image("https://i.imgur.com/EsSwBjC.jpg", 
                 caption="✅ Exemple: Viande fraîche",
                 use_container_width=True) 
        st.success("✅ Cette viande serait classifiée comme **fraîche**")
    
    with col2:
        st.image("https://i.imgur.com/yaSv0M0.jpg", 
                 caption="⚠️ Exemple: Viande avariée",
                 use_container_width=True)  
        st.error("⚠️ Cette viande serait classifiée comme **avariée**")

st.markdown('</div>', unsafe_allow_html=True)

# section infos
st.markdown('<div class="container">', unsafe_allow_html=True)
st.markdown("## 🧪 Comment reconnaître une viande avariée?")
st.markdown("""
La viande avariée peut présenter certains **signes distinctifs** à surveiller:

1. 🎨 **Changement de couleur**: La viande devient grisâtre, brunâtre ou verdâtre
2. 👃 **Odeur désagréable**: Une odeur aigre, acide ou putride
3. ✋ **Texture visqueuse ou collante**: La surface devient glissante au toucher
4. 🦠 **Moisissures**: Présence de taches de moisissure blanche, verte ou noire
5. 📅 **Date de péremption dépassée**: Toujours vérifier la date limite de consommation

⚠️ **En cas de doute, ne prenez aucun risque** : il vaut mieux jeter une viande douteuse que de risquer une intoxication alimentaire.
""")
st.markdown('</div>', unsafe_allow_html=True)

# en bas de la page
st.markdown('<p class="footer">© 2025 Meat Analyzer 🥩 | Développé par Tom GRACI 👨‍💻 | Système à des fins éducatives 🎓</p>', unsafe_allow_html=True)
