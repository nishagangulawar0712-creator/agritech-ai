import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AgriTech AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #f5f8f5;
    }

    /* Header */
    .main-title {
        font-size: 48px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 20px;
        margin-top: 0px;
        margin-bottom: 30px;
    }

    /* Cards */
    .card {
        background: white;
        padding: 25px;
        border-radius: 18px;
        box-shadow: 0px 4px 18px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }

    .prediction-card {
        background: white;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0px 5px 25px rgba(0,0,0,0.10);
        border-left: 6px solid #2e7d32;
    }

    .disease-name {
        font-size: 30px;
        font-weight: 700;
    }

    .confidence {
        font-size: 24px;
        font-weight: 600;
    }

    .section-title {
        font-size: 25px;
        font-weight: 700;
        margin-top: 10px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        height: 50px;
        font-size: 17px;
        font-weight: 600;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 25px;
        margin-top: 40px;
        font-size: 14px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# DISEASE INFORMATION
# ============================================================

disease_info = {

    "Pepper__bell___healthy": {
        "name": "Pepper Bell Healthy",
        "description":
            "The pepper bell leaf appears healthy with no major visible disease symptoms.",
        "action":
            "Continue regular watering, nutrition, and monitoring."
    },

    "Pepper__bell___Bacterial_spot": {
        "name": "Pepper Bell Bacterial Spot",
        "description":
            "Bacterial spot can cause small dark lesions on leaves and fruit.",
        "action":
            "Remove severely infected leaves and avoid overhead watering."
    },

    "Potato___Early_blight": {
        "name": "Potato Early Blight",
        "description":
            "Early blight commonly produces dark circular lesions with concentric rings.",
        "action":
            "Remove infected foliage and maintain good air circulation."
    },

    "Potato___Late_blight": {
        "name": "Potato Late Blight",
        "description":
            "Late blight can rapidly damage potato leaves under favorable humid conditions.",
        "action":
            "Remove infected material and avoid prolonged leaf wetness."
    },

    "Potato___healthy": {
        "name": "Potato Healthy",
        "description":
            "The potato leaf appears healthy without detected disease symptoms.",
        "action":
            "Continue normal crop monitoring and plant care."
    },

    "Tomato_Bacterial_spot": {
        "name": "Tomato Bacterial Spot",
        "description":
            "Bacterial spot causes small dark spots on tomato leaves and can affect fruit.",
        "action":
            "Remove affected foliage and minimize water splashing between plants."
    },

    "Tomato_Early_blight": {
        "name": "Tomato Early Blight",
        "description":
            "Early blight produces brown lesions, often with characteristic concentric rings.",
        "action":
            "Remove affected leaves and improve airflow around plants."
    },

    "Tomato_Late_blight": {
        "name": "Tomato Late Blight",
        "description":
            "Late blight is a serious disease that can rapidly spread through tomato crops.",
        "action":
            "Remove infected plant material and avoid wet foliage."
    },

    "Tomato_Leaf_Mold": {
        "name": "Tomato Leaf Mold",
        "description":
            "Leaf mold commonly appears as yellow areas on the upper leaf surface with mold growth underneath.",
        "action":
            "Improve ventilation and reduce humidity around plants."
    },

    "Tomato_Septoria_leaf_spot": {
        "name": "Tomato Septoria Leaf Spot",
        "description":
            "Septoria leaf spot produces numerous small circular spots on tomato leaves.",
        "action":
            "Remove affected leaves and avoid overhead irrigation."
    },

    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "name": "Tomato Spider Mites",
        "description":
            "Spider mites are tiny pests that can cause stippling, yellowing, and leaf damage.",
        "action":
            "Inspect leaves carefully and consider appropriate pest-management methods."
    },

    "Tomato__Target_Spot": {
        "name": "Tomato Target Spot",
        "description":
            "Target spot produces circular lesions that may develop concentric patterns.",
        "action":
            "Remove affected foliage and improve plant ventilation."
    },

    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "name": "Tomato Yellow Leaf Curl Virus",
        "description":
            "This viral disease can cause yellowing, curling leaves, and stunted plant growth.",
        "action":
            "Control insect vectors such as whiteflies and remove severely affected plants."
    },

    "Tomato__Tomato_mosaic_virus": {
        "name": "Tomato Mosaic Virus",
        "description":
            "Tomato mosaic virus can cause mottled leaf patterns and reduced plant growth.",
        "action":
            "Remove infected plants and sanitize tools used around the crop."
    },

    "Tomato_healthy": {
        "name": "Tomato Healthy",
        "description":
            "The tomato leaf appears healthy with no detected disease symptoms.",
        "action":
            "Continue regular monitoring, watering, and crop management."
    }
}


# ============================================================
# MODEL CONFIGURATION
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    checkpoint = torch.load(
        "models/resnet18_plant_disease.pth",
        map_location=device
    )

    class_names = checkpoint["class_names"]

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        len(class_names)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    model.eval()

    return model, class_names


model, class_names = load_model()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🌱 AgriTech AI")

    st.markdown("---")

    st.markdown("### 🧠 AI Model")

    st.write("Fine-Tuned ResNet18")

    st.markdown("### 📊 Model Performance")

    st.metric(
        "Validation Accuracy",
        "98.93%"
    )

    st.metric(
        "Training Accuracy",
        "99.44%"
    )

    st.markdown("---")

    st.markdown("### 📚 Dataset")

    st.write("PlantVillage")

    st.write("15 disease classes")

    st.write("20,638 images")

    st.markdown("---")

    st.markdown(
        "Developed as an AI-powered plant disease detection system."
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🌱 AgriTech AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent Plant Disease Detection using Deep Learning'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# TOP INFORMATION CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "🧠 AI Model",
        "ResNet18"
    )

with col2:
    st.metric(
        "🎯 Accuracy",
        "98.93%"
    )

with col3:
    st.metric(
        "🌿 Classes",
        "15"
    )

with col4:
    st.metric(
        "🖼️ Dataset",
        "20,638"
    )


st.markdown("---")


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">📷 Upload Plant Leaf</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload a clear image of a plant leaf",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG and PNG"
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    left, right = st.columns([1, 1])

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    with left:

        st.markdown("### 📷 Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    with right:

        st.markdown("### 🔍 AI Analysis")

        if st.button(
            "🚀 Analyze Plant Leaf",
            use_container_width=True
        ):

            with st.spinner(
                "AI is analyzing the leaf..."
            ):

                image_tensor = transform(
                    image
                )

                image_tensor = (
                    image_tensor
                    .unsqueeze(0)
                    .to(device)
                )

                with torch.no_grad():

                    outputs = model(
                        image_tensor
                    )

                    probabilities = F.softmax(
                        outputs,
                        dim=1
                    )

                    top_probabilities, top_indices = torch.topk(
                        probabilities,
                        3
                    )

                predicted_index = top_indices[0][0].item()

                predicted_class = class_names[
                    predicted_index
                ]

                confidence = (
                    top_probabilities[0][0].item()
                    * 100
                )

                info = disease_info.get(
                    predicted_class,
                    {
                        "name": predicted_class,
                        "description":
                            "No additional information available.",
                        "action":
                            "Please consult an agricultural expert."
                    }
                )

                # ------------------------------------------------
                # MAIN RESULT
                # ------------------------------------------------

                st.markdown(
                    '<div class="prediction-card">',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="disease-name">'
                    f'🌿 {info["name"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="confidence">'
                    f'Confidence: {confidence:.2f}%'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.progress(
                    min(confidence / 100, 1.0)
                )

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )

    # ========================================================
    # ADDITIONAL INFORMATION
    # ========================================================

    if "predicted_class" in locals():

        st.markdown("---")

        info_col1, info_col2 = st.columns(2)

        with info_col1:

            st.markdown(
                "### 💡 About the Detection"
            )

            st.write(
                info["description"]
            )

        with info_col2:

            st.markdown(
                "### 🌾 Recommended Action"
            )

            st.write(
                info["action"]
            )

        # ====================================================
        # TOP 3 PREDICTIONS
        # ====================================================

        st.markdown("---")

        st.markdown(
            "### 🏆 Top 3 AI Predictions"
        )

        for rank in range(3):

            index = top_indices[
                0
            ][rank].item()

            probability = (
                top_probabilities[
                    0
                ][rank].item() * 100
            )

            class_name = class_names[
                index
            ]

            display_name = disease_info.get(
                class_name,
                {"name": class_name}
            )["name"]

            st.write(
                f"**{rank + 1}. {display_name}**"
            )

            st.progress(
                min(probability / 100, 1.0)
            )

            st.caption(
                f"Confidence: {probability:.2f}%"
            )


# ============================================================
# ABOUT SECTION
# ============================================================

st.markdown("---")

st.markdown(
    "## 🌱 About AgriTech AI"
)

about_col1, about_col2 = st.columns(2)

with about_col1:

    st.markdown(
        """
        ### 🎯 Objective

        AgriTech AI is an AI-powered plant disease
        detection system designed to identify diseases
        from plant leaf images.

        The system uses a fine-tuned ResNet18 deep
        learning model trained on the PlantVillage dataset.
        """
    )


with about_col2:

    st.markdown(
        """
        ### ⚙️ Technology

        **Deep Learning:** ResNet18

        **Framework:** PyTorch

        **Frontend:** Streamlit

        **Dataset:** PlantVillage

        **Classes:** 15

        **Validation Accuracy:** 98.93%
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="footer">'
    '🌱 <b>AgriTech AI</b> | '
    'Plant Disease Detection using Deep Learning'
    '</div>',
    unsafe_allow_html=True
)