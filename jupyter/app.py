import streamlit as st
import pickle
import re
import os
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Spam Detection TikTok",
    page_icon="🛑",
    layout="centered"
)

# ======================
# INIT SASTRAWI
# ======================
stemmer = StemmerFactory().create_stemmer()
stopword = StopWordRemoverFactory().create_stop_word_remover()

# ======================
# CLEANING FUNCTIONS
# ======================
def clean_text_rule(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_text_ml(text):
    text = clean_text_rule(text)
    text = stemmer.stem(text)
    text = stopword.remove(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ======================
# LOAD MODEL & TF-IDF
# ======================
BASE_DIR = os.path.dirname(__file__)

with open(os.path.join(BASE_DIR, "model_lr.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "tfidf.pkl"), "rb") as f:
    tfidf = pickle.load(f)

# ======================
# KEYWORDS
# ======================
SPAM_PROMOSI = [
    "klik", "link", "bio", "dm", "wa", "whatsapp",
    "gratis", "promo", "diskon", "jual",
    "saldo", "cuan", "follow", "subscribe", "hubungi"
]

KATA_KASAR = [
    "anjing", "bangsat", "goblok", "tolol", "babi",
    "ngentot", "jancuk", "kampret", "tai", "brengsek"
]

PROVOKASI = [
    "bunuh", "bakar", "serang",
    "ganyang", "hancurkan", "perang"
]

OPINI_KONSTRUKTIF = [
    "harusnya", "sebaiknya", "menurut saya",
    "menurutku", "musyawarah", "kebijakan",
    "masyarakat", "pemerintah", "evaluasi", "solusi"
]

# ======================
# STREAMLIT UI
# ======================
st.title("🛑 Deteksi Spam Komentar TikTok")
st.write("Masukkan komentar untuk mendeteksi **Spam** atau **Bukan Spam**")

comment = st.text_area("Komentar:", height=150)

if st.button("Deteksi"):
    if comment.strip() == "":
        st.warning("Komentar tidak boleh kosong.")
    else:
        # ======================
        # PREPROCESSING
        # ======================
        clean_rule = clean_text_rule(comment)
        clean_ml = clean_text_ml(comment)
        vector = tfidf.transform([clean_ml])

        # ======================
        # DECISION LOGIC
        # ======================
        is_spam = False
        label = ""
        spam_category = None

        # 1️⃣ PROVOKASI
        if any(k in clean_rule for k in PROVOKASI):
            is_spam = True
            spam_category = "Provokasi / Hasutan Keras"
            label = "🔥 SPAM - PROVOKASI"

        # 2️⃣ TOXIC
        elif any(k in clean_rule for k in KATA_KASAR):
            is_spam = True
            spam_category = "Toxic / Kata Kasar"
            label = "⚠️ SPAM - TOXIC"

        # 3️⃣ SPAM PROMOSI
        elif any(k in clean_rule for k in SPAM_PROMOSI):
            is_spam = True
            spam_category = "Spam Promosi"
            label = "🚨 SPAM - PROMOSI"

        # 4️⃣ OPINI KONSTRUKTIF
        elif any(k in clean_rule for k in OPINI_KONSTRUKTIF):
            is_spam = False
            label = "✅ BUKAN SPAM (opini / kritik)"

        # 5️⃣ MACHINE LEARNING
        else:
            prob = model.predict_proba(vector)[0][1]
            if prob > 0.75:
                is_spam = True
                spam_category = "Spam (Prediksi Machine Learning)"
                label = "🚨 SPAM (ML)"
            else:
                is_spam = False
                label = "✅ BUKAN SPAM"

        # ======================
        # OUTPUT
        # ======================
        st.subheader("Hasil Deteksi")
        st.write(label)

        if is_spam:
            st.error("Komentar terdeteksi sebagai SPAM")
            st.write("📌 Kategori Spam:")
            st.write(f"**{spam_category}**")
            st.info("Komentar disembunyikan karena terdeteksi spam")
        else:
            st.success("Komentar NON-SPAM")
            st.subheader("Komentar Ditampilkan")
            st.write(comment)

        # ======================
        # DEBUG (OPSIONAL)
        # ======================
        # st.subheader("Preprocessing (Rule-Based)")
        # st.code(clean_rule)

        # st.subheader("Preprocessing (Machine Learning)")
        # st.code(clean_ml)
