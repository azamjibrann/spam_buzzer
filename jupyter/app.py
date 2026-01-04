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

# Untuk RULE-BASED (tanpa stemming & stopword)
def clean_text_rule(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Untuk MACHINE LEARNING (HARUS SAMA DENGAN TRAINING)
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
    "saldo", "cuan", "follow", "subscribe" ,"hubungi","web kami"
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
    "menurutku", "bermusyawarah", "musyawarah",
    "kebijakan", "masyarakat", "pemerintah",
    "presiden", "evaluasi", "solusi", "pendapat","menurut kita"
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
        # preprocessing
        clean_rule = clean_text_rule(comment)
        clean_ml = clean_text_ml(comment)
        vector = tfidf.transform([clean_ml])

        # ======================
        # DECISION LOGIC (URUTAN WAJIB)
        # ======================
        # 1️⃣ PROVOKASI KERAS (PALING BERBAHAYA)
        if any(k in clean_rule for k in PROVOKASI):
            st.error("🔥 PROVOKASI / HASUTAN KERAS")

        # 2️⃣ KATA KASAR / TOXIC
        elif any(k in clean_rule for k in KATA_KASAR):
            st.warning("⚠️ TOXIC / KATA KASAR")

        # 3️⃣ SPAM PROMOSI
        elif any(k in clean_rule for k in SPAM_PROMOSI):
            st.error("🚨 SPAM PROMOSI")

        # 4️⃣ OPINI / KRITIK KONSTRUKTIF
        elif any(k in clean_rule for k in OPINI_KONSTRUKTIF):
            st.success("✅ BUKAN SPAM (opini / kritik)")

        # 5️⃣ MACHINE LEARNING (FALLBACK)
        else:
            prob = model.predict_proba(vector)[0][1]
            if prob > 0.75:
                st.error("🚨 SPAM (berdasarkan model ML)")
            else:
                st.success("✅ BUKAN SPAM")

        # # 1️⃣ OPINI / KRITIK KONSTRUKTIF
        # if any(k in clean_rule for k in OPINI_KONSTRUKTIF):
        #     st.success("✅ BUKAN SPAM (opini / kritik)")

        # # 2️⃣ SPAM KERAS
        # elif any(k in clean_rule for k in SPAM_PROMOSI + KATA_KASAR + PROVOKASI):
        #     st.error("🚨 SPAM")

        # # 3️⃣ MACHINE LEARNING
        # else:
        #     prob = model.predict_proba(vector)[0][1]
        #     if prob > 0.75:
        #         st.error("🚨 SPAM (berdasarkan model)")
        #     else:
        #         st.success("✅ BUKAN SPAM")

        # ======================
        # DEBUG OUTPUT
        # ======================
        st.subheader("Preprocessing (Rule-Based)")
        st.code(clean_rule)

        st.subheader("Preprocessing (Machine Learning)")
        st.code(clean_ml)
