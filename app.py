import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="PCOS & PCOD Health Assistant",
    page_icon="💜",
    layout="centered"
)

# ---------------- CSS ---------------- #
st.markdown("""
            
<style>
.stDownloadButton button{
    background: #C4B5FD !important;   /* Light Purple */
    color: #4C1D95 !important;        /* Dark Purple Text */
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
}

.stDownloadButton button:hover{
    background: #A78BFA !important;
    color: white !important;
}
.stApp{
    background-color: #FFE4EC;
}
[data-testid="stChatInput"] textarea {
    color: white !important;
}
/* Main text */
html, body, [class*="css"] {
    color: black !important;
}

/* Streamlit markdown text */
p, div, span, label {
    color: black !important;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    color: black !important;
}

/* Input text */
textarea,
input {
    color: black !important;
}

/* Sidebar text */
[data-testid="stSidebar"] * {
    color: black !important;
}

            .main-title{
    text-align:center;
    font-size:55px;
    font-weight:900;
    color:#9333EA;
}

.sub-title{
    text-align:center;
    font-size:22px;
    color:#DB2777;
}
</style>

""", unsafe_allow_html=True)

st.markdown("""
<h1 style='
text-align:center;
font-size:35px;
font-weight:500;
color:#9333EA;
margin-bottom:0px;
'>
💜 PCOS & PCOD Health Assistant
</h1>

<p style='
text-align:center;
font-size:22px;
color:#DB2777;
margin-top:-10px;
margin-bottom:30px;
'>
🌸 AI-Powered Women's Health Companion 🌸
</p>
""", unsafe_allow_html=True)
#------------ FAQ DATA ---------------- #

faq_data = pd.DataFrame({

"Question":[

"What is PCOS?",
"What is PCOD?",
"Are PCOS and PCOD same?",
"What causes PCOS?",
"What are symptoms of PCOS?",
"Can PCOS cause weight gain?",
"Can PCOS cause irregular periods?",
"Can PCOS cause infertility?",
"Can PCOS be cured?",
"Can exercise help with PCOS?",

"What foods are good for PCOS?",
"PCOS diet",
"Diet for PCOS",
"Best diet for PCOS",
"Foods for PCOS",
"What should I eat in PCOS?",

"What foods should be avoided in PCOS?",
"Can stress worsen PCOS?",
"Is PCOS genetic?",
"Can PCOS cause acne?",
"Can PCOS cause hair loss?",
"Can PCOS cause facial hair?",
"What is insulin resistance in PCOS?",
"Can PCOS increase diabetes risk?",
"Can PCOS affect mental health?",
"How is PCOS diagnosed?",
"What is treatment for PCOS?",
"Can weight loss improve PCOS?",
"Can teenagers have PCOS?",
"Can PCOS affect sleep?",
"Can PCOS cause mood swings?",
"Is yoga good for PCOS?",
"Can PCOS affect pregnancy?",
"Can PCOS cause dark skin patches?",
"How can I manage PCOS naturally?"

],

"Answer":[

"PCOS stands for Polycystic Ovary Syndrome, a hormonal disorder affecting women of reproductive age.",

"PCOD stands for Polycystic Ovarian Disease where ovaries produce immature eggs.",

"PCOS and PCOD are related conditions, but PCOS generally involves more severe hormonal imbalance.",

"PCOS may be caused by genetics, insulin resistance, hormonal imbalance and lifestyle factors.",

"Common symptoms include irregular periods, weight gain, acne, facial hair growth and hair loss.",

"Yes. Hormonal imbalance and insulin resistance can contribute to weight gain.",

"Yes. Irregular or missed periods are one of the most common symptoms.",

"PCOS can make conception difficult but treatment can improve fertility.",

"There is no permanent cure but symptoms can be effectively managed.",

"Yes. Regular exercise improves insulin sensitivity and overall health.",

"Whole grains, vegetables, fruits, lean proteins and healthy fats are beneficial for PCOS.",

"Whole grains, vegetables, fruits, lean proteins and healthy fats are beneficial for PCOS.",

"Whole grains, vegetables, fruits, lean proteins and healthy fats are beneficial for PCOS.",

"Whole grains, vegetables, fruits, lean proteins and healthy fats are beneficial for PCOS.",

"Whole grains, vegetables, fruits, lean proteins and healthy fats are beneficial for PCOS.",

"Whole grains, vegetables, fruits, lean proteins and healthy fats are beneficial for PCOS.",

"Limit sugary foods, processed foods and refined carbohydrates.",

"Yes. Stress can worsen hormonal imbalance and symptoms.",

"Yes. PCOS often runs in families.",

"Yes. Increased androgen levels can cause acne.",

"Yes. Hormonal imbalance may lead to hair thinning or hair loss.",

"Yes. Excess androgen hormones may cause unwanted facial hair growth.",

"Insulin resistance means the body cannot effectively use insulin.",

"Yes. Women with PCOS have a higher risk of Type 2 Diabetes.",

"Yes. Anxiety, depression and low self-esteem may occur.",

"Doctors diagnose PCOS using symptoms, blood tests, ultrasound and medical history.",

"Treatment may include lifestyle changes, medication and hormone therapy.",

"Even modest weight loss can improve symptoms and hormone balance.",

"Yes. PCOS can develop during teenage years.",

"Yes. PCOS may contribute to sleep disturbances.",

"Yes. Hormonal changes may lead to mood swings.",

"Yes. Yoga can help reduce stress and support hormonal balance.",

"Women with PCOS can have healthy pregnancies with proper care.",

"Yes. Dark skin patches may occur due to insulin resistance.",

"Healthy eating, regular exercise, stress management and adequate sleep can help."

]

})

questions = faq_data["Question"]
answers = faq_data["Answer"]

# ---------------- AI MODEL ---------------- #

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

question_embeddings = model.encode(
    questions.tolist(),
    convert_to_tensor=False
)


# ---------------- SIDEBAR ---------------- #


# ---------------- CHAT HISTORY ---------------- #

if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": """
👋 Welcome to the PCOS & PCOD Health Assistant!

I can help you with:

💜 PCOS Symptoms
🌸 PCOD Management
🥗 Diet & Nutrition
🏃 Exercise & Fitness
⚖️ Weight Loss
🤰 Fertility & Pregnancy
💊 Treatment Options

Ask me anything!
"""
        }
    ]

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
# ---------------- USER INPUT ---------------- #

user_question = st.chat_input(
    "Ask your question..."
)

if user_question:

    st.session_state.messages.append(
        {
            "role":"user",
            "content":user_question
        }
    )

    with st.chat_message("user"):
        st.write(user_question)

    user_embedding = model.encode(
        [user_question],
        convert_to_tensor=False
    )

    similarity_scores = cosine_similarity(
        user_embedding,
        question_embeddings
    )

    best_match_index = np.argmax(
        similarity_scores
    )

    best_score = similarity_scores[
        0
    ][best_match_index]

    if best_score > 0.45:

        response = answers.iloc[
            best_match_index
        ]

    else:

        response = """
❓ I couldn't find a relevant answer.

Try asking about:

• PCOS symptoms
• Diet
• Weight loss
• Exercise
• Fertility
• Pregnancy
• Insulin resistance

💜 I'm here to help.
"""

    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":response
        }
    )

    with st.chat_message("assistant"):
        st.write(response)

# ---------------- DOWNLOAD CHAT ---------------- #

if st.session_state.messages:

    chat_history = ""

    for msg in st.session_state.messages:

        chat_history += (
            f"{msg['role'].upper()}: "
            f"{msg['content']}\n\n"
        )

    st.download_button(
        "📥 Download Chat History",
        chat_history,
        file_name="pcos_chat_history.txt"
    )
st.markdown("<br><br>", unsafe_allow_html=True)

st.divider()

st.caption(
    "💜 Developed by Anusha Syed | CodeAlpha AI Internship"
)