import streamlit as st
from PIL import Image
import time
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# config de la page
st.set_page_config(
    page_title="Meat Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# déco
st.markdown("""
<style>
    /* Fond principal avec motifs */
    .stApp {
        background-color: #A182A8;
        background-image: 
            radial-gradient(circle at 20% 20%, rgba(255, 255, 255, 0.15) 0%, rgba(255, 255, 255, 0) 20%),
            radial-gradient(circle at 80% 60%, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0) 25%),
            linear-gradient(60deg, rgba(126, 87, 140, 0.5) 25%, transparent 25.5%),
            linear-gradient(120deg, rgba(160, 124, 175, 0.5) 25%, transparent 25.5%),
            linear-gradient(180deg, rgba(142, 107, 158, 0.5) 25%, transparent 25.5%),
            linear-gradient(240deg, rgba(179, 147, 191, 0.5) 25%, transparent 25.5%);
        background-size: 100%, 100%, 120px 120px, 120px 120px, 120px 120px, 120px 120px;
        background-position: 0 0, 0 0, 10px 0, 10px 0, 0 0, 0 0;
    }
    
    /* Style du tableau de bord */
    .main-title {
        text-align: center;
        color: white;
        font-size: 3rem;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #7A5F82 0%, #9370DB 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        border: 2px solid rgba(255, 255, 255, 0.1);
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .subtitle {
        text-align: center;
        color: #F0E6F5;
        font-size: 1.3rem;
        margin-bottom: 2rem;
        background-color: rgba(80, 45, 90, 0.7);
        padding: 0.8rem;
        border-radius: 10px;
    }
    
    .prediction-fresh {
        background: linear-gradient(135deg, #1E8449 0%, #27AE60 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        font-size: 1.5rem;
        text-align: center;
        margin: 1rem 0;
        border: 2px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
    }
    
    .prediction-spoiled {
        background: linear-gradient(135deg, #922B21 0%, #E74C3C 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        font-size: 1.5rem;
        text-align: center;
        margin: 1rem 0;
        border: 2px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
    }
    
    .container {
        background-color: white;
        background-image: 
            radial-gradient(circle at 100% 100%, rgba(161, 130, 168, 0.1) 0%, transparent 25%),
            radial-gradient(circle at 0% 0%, rgba(161, 130, 168, 0.1) 0%, transparent 25%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
        margin: 1.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .section-title {
        background: linear-gradient(135deg, #614385 0%, #516395 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.2);
        border-left: 5px solid #BFA5C3;
    }
    
    .instructions {
        background-color: #F0E6F5;
        background-image: 
            linear-gradient(45deg, rgba(161, 130, 168, 0.1) 25%, transparent 25%, transparent 50%, 
            rgba(161, 130, 168, 0.1) 50%, rgba(161, 130, 168, 0.1) 75%, transparent 75%);
        background-size: 20px 20px;
        padding: 1.8rem;
        border-radius: 10px;
        margin: 1.2rem 0;
        border: 1px solid #D5C5DB;
        box-shadow: inset 0 2px 5px rgba(0, 0, 0, 0.05);
    }
    
    .info-block {
        background-color: rgba(240, 230, 245, 0.9);
        background-image: 
            repeating-linear-gradient(45deg, rgba(161, 130, 168, 0.05) 0px, rgba(161, 130, 168, 0.05) 2px,
            transparent 2px, transparent 4px);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #9370DB;
        margin: 1rem 0;
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.1);
    }
    
    .upload-section {
        background-color: rgba(240, 235, 245, 0.8);
        background-image: 
            linear-gradient(-45deg, rgba(161, 130, 168, 0.1) 25%, transparent 25%, 
            transparent 50%, rgba(161, 130, 168, 0.1) 50%, rgba(161, 130, 168, 0.1) 75%, transparent 75%);
        background-size: 16px 16px;
        padding: 1.8rem;
        border-radius: 15px;
        border: 2px dashed #8E6B9E;
        margin: 1.2rem 0;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: #614385;
        box-shadow: 0 0 15px rgba(97, 67, 133, 0.3);
    }
    
    .footer {
        text-align: center;
        color: white;
        background-color: rgba(80, 45, 90, 0.7);
        padding: 1rem;
        border-radius: 10px;
        margin-top: 2rem;
        font-size: 0.9rem;
    }
    
    /* Personnalisation des composants Streamlit */
    .stButton button {
        background: linear-gradient(135deg, #7A5F82 0%, #9370DB 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    
    .stProgress .st-bo {
        background-color: #9370DB;
    }
    
    .stTextInput input, .stSelectbox select {
        border-radius: 8px;
        border: 1px solid #D5C5DB;
    }
</style>
""", unsafe_allow_html=True)

# barre latérale
with st.sidebar:
    st.image("https://institut-agro-dijon.fr/fileadmin/user_upload/INSTITUT-DIJON-MARQUE-ETAT.svg", width=200)
    st.markdown("## À propos")
    st.info("""
    Cette application utilise un modèle de Deep Learning pour déterminer si une viande est fraîche ou avariée à partir d'une simple image.
    
    Chargez une image de viande pour obtenir une analyse instantanée.
            
    L'ensemble du code est disponible en open-source sur GitHub : https://github.com/gtom-pandas
    """)
    
    st.markdown("### Comment ça marche")
    st.markdown("""
    1. Téléchargez une image de viande
    2. Notre modèle d'IA analyse l'image
    3. Recevez le résultat: Fraîche ou Avariée
    """)
    
    st.markdown("### Développé avec")
    st.markdown("- TensorFlow")
    st.markdown("- Streamlit")
    st.markdown("- Python")
    
# Contenu principal
st.markdown('<h1 class="main-title">Meat Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Système d\'analyse avancé pour déterminer la fraîcheur de la viande</p>', unsafe_allow_html=True)

# charge le modèle
@st.cache_resource
def load_classification_model():
    return load_model('meat_fresh_model.keras')

model = load_classification_model()

# sectionn principale
st.markdown('<div class="container">', unsafe_allow_html=True)
st.markdown("## Analysez votre échantillon de viande")
st.markdown('<div class="instructions">', unsafe_allow_html=True)
st.markdown("""
#### Instructions:
- Prenez une photo claire de votre échantillon de viande (boeuf, porc, poulet)
- Assurez-vous que l'image est la plus nette possible
- Téléchargez l'image ci-dessous pour l'analyse
""")
st.markdown('</div>', unsafe_allow_html=True)

# pour DL les photos à analyser
uploaded_file = st.file_uploader("Choisissez une image de viande à analyser...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # display de l'image + analyse
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Image analysée")
        st.image(uploaded_file, caption='Échantillon de viande', use_container_width=True)
    
    with col2:
        st.markdown("### Résultats de l'analyse")

        # Animation pendant le traitement
        with st.spinner("Analyse en cours..."):
            # simul de délai pour montrer le spinner :)
            time.sleep(1)
            
            # Prépare l'image pour la prédiction
            img = image.load_img(uploaded_file, target_size=(128, 128))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            prediction = model.predict(img_array)
            confidence = prediction[0][0]
            
            # Affichage du résultat avec une barre de confiance
            is_spoiled = confidence >= 0.5
            confidence_pct = confidence * 100 if is_spoiled else (1 - confidence) * 100
            
            if is_spoiled:
                st.markdown('<div class="prediction-spoiled">⚠️ VIANDE AVARIÉE</div>', unsafe_allow_html=True)
                bar_color = "#E74C3C"
            else:
                st.markdown('<div class="prediction-fresh">✅ VIANDE FRAÎCHE</div>', unsafe_allow_html=True)
                bar_color = "#27AE60"
            
            st.markdown(f"### Niveau de confiance: {confidence_pct:.1f}%")
            st.progress(float(confidence_pct/100))
            
            # Afficher des recommandations basées sur le résultat
            if is_spoiled:
                st.error("""
                **Recommandation**: Cette viande présente des signes de détérioration et ne devrait pas être consommée.
                
                Veuillez la jeter de manière appropriée pour éviter tout risque sanitaire.
                """)
            else:
                st.success("""
                **Recommandation**: Cette viande semble être fraîche et propre à la consommation.
                
                N'oubliez pas de la conserver correctement et de la cuisiner à une température adéquate.
                """)

# infos complémentaires
if not uploaded_file:
    # exemple
    st.markdown("### Exemples de classification")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image("C:\\Users\\UTGR0501\\Python_Pandas\\images_dataset_meat\\Fresh\\test_20171016_104321D.jpg", 
                 caption="Exemple: Viande fraîche")
        st.success("Cette viande serait classifiée comme fraîche")
    
    with col2:
        st.image("C:\\Users\\UTGR0501\\Python_Pandas\\images_dataset_meat\\Spoiled\\test_20171019_030921D.jpg", 
                 caption="Exemple: Viande avariée")
        st.error("Cette viande serait classifiée comme avariée")

st.markdown('</div>', unsafe_allow_html=True)

# section infos
st.markdown('<div class="container">', unsafe_allow_html=True)
st.markdown("## Comment reconnaître une viande avariée?")
st.markdown("""
La viande avariée peut présenter certains signes distinctifs:

1. **Changement de couleur**: La viande devient grisâtre, brunâtre ou verdâtre
2. **Odeur désagréable**: Une odeur aigre ou putride
3. **Texture visqueuse ou collante**: La surface devient visqueuse au toucher
4. **Moisissures**: Présence de taches de moisissure blanche, verte ou noire
5. **Date de péremption dépassée**: Toujours vérifier la date limite de consommation
""")
st.markdown('</div>', unsafe_allow_html=True)

# en bas de la page
st.markdown('<p class="footer">© 2025 Meat Analyzer - Système développé à des fins éducatives par Tom GRACI</p>', unsafe_allow_html=True)