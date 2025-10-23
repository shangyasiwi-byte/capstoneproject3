# app.py
import base64
import os
import streamlit as st
import langdetect
from modules.agent import agent

# ---- Konfigurasi halaman ----
st.set_page_config(page_title="🎬 IMDB RAG Chatbot", page_icon="🎥", layout="centered")

# CSS Customization
import os
image_path = os.path.join(os.path.dirname(__file__), "images", "cinema.jpg")

# Encode gambar jadi base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

base64_image = get_base64_image(image_path)

page_bg = f"""
<style>
/* 🌆 Background image */
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpg;base64,{base64_image}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    background-repeat: no-repeat;
    position: relative;
}}

[data-testid="stAppViewContainer"]::before {{
    content: "";
    position: absolute;
    inset: 0;
    background: rgba(0, 0, 0, 0); /* transparan penuh */
    z-index: 0;
}}

/* 🧭 Header & toolbar transparan */
[data-testid="stHeader"], [data-testid="stToolbar"] {{
    background: rgba(0, 0, 0, 0);
}}

/* 🎞️ Title dan teks */
.title-container {{
    text-align: center;
    padding-top: 6rem;
    padding-bottom: 3rem;
    z-index: 1;
}}

.title-container h1 {{
    color: #fff;
    text-shadow: 2px 2px 6px rgba(0,0,0,0.6);
    font-size: 2.8rem;
}}

.title-container p {{
    color: #f0f0f0;
    font-size: 1.1rem;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.5);
}}
/* Chat bubble */
.chat-bubble {{
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    margin-bottom: 1rem;
    animation: fadeIn 0.4s ease-in-out;
}}

@keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(10px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

.chat-bubble .avatar {{
    font-size: 1.5rem;
    flex-shrink: 0;
}}

.chat-bubble .bubble {{
    padding: 1rem 1.2rem;
    border-radius: 18px;
    line-height: 1.5;
    max-width: 75%;
    box-shadow: 0 3px 10px rgba(0,0,0,0.25);
}}

.chat-bubble.user {{
    justify-content: flex-end;
}}

.chat-bubble.user .bubble {{
    background: linear-gradient(135deg, #ff5f6d, #ffc371);
    color: white;
    border-bottom-right-radius: 5px;
    text-align: right;
}}

.chat-bubble.assistant .bubble {{
    background: rgba(255,255,255,0.9);
    color: #222;
    border-bottom-left-radius: 5px;
    backdrop-filter: blur(6px);
}}

/* Tombol Reset Chat */
.reset-button {{
    position: fixed;
    top: 20px;
    right: 30px;
    background: rgba(255, 77, 77, 0.9);
    color: white;
    border-radius: 25px;
    padding: 0.5rem 1.1rem;
    font-size: 0.9rem;
    border: none;
    cursor: pointer;
    transition: 0.2s ease;
    z-index: 999;
}}
.reset-button:hover {{
    background: rgba(255, 30, 30, 1);
    transform: scale(1.05);
}}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# =========================
# 💬 App Header
# =========================

st.markdown(
    """
    <div class="title-container">
        <h1>🎬 IMDB Movie RAG Chatbot</h1>
        <p>Tanyakan apa saja tentang film — powered by OpenAI + Qdrant RAG</p>
    </div>
    """,
    unsafe_allow_html=True
)

#memory
if "agent" not in st.session_state:
    st.session_state.agent = agent 
# ---- Chat Section ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tombol Reset Chat
if st.button("🔄 Reset Chat", key="reset", help="Mulai ulang percakapan", type="secondary"):
    st.session_state.messages = []
    st.rerun()

# Tampilkan chat history
for msg in st.session_state.messages:
    role = msg["role"]
    emoji = "🎥" if role == "assistant" else "🧑‍💬"
    bubble_class = "assistant" if role == "assistant" else "user"
    st.markdown(
        f"""
        <div class="chat-bubble {bubble_class}">
            <div class="avatar">{emoji}</div>
            <div class="bubble">{msg['content']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Input pengguna
if prompt := st.chat_input("Tulis pertanyaan tentang film..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

# Proses agent (pakai memory)
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    prompt = st.session_state.messages[-1]["content"]

    with st.spinner("Sedang mencari jawaban... 🎬"):
        try:
            # Gabungkan konteks chat sebelumnya
            context_text = "\n".join(
                [f"{m['role']}: {m['content']}" for m in st.session_state.messages[:-1]]
            )
            full_prompt = f"{context_text}\nuser: {prompt}\nassistant:"

            # 🧠 Deteksi bahasa pengguna
            try:
                lang = langdetect.detect(prompt)
                lang_instruction = "Jawab dalam Bahasa Indonesia." if lang == "id" else "Answer in English."
            except:
                lang_instruction = ""

            # Tambahkan instruksi bahasa ke prompt
            final_prompt = f"{full_prompt}\n\n{lang_instruction}"

            # 🚀 Panggil agent
            result = st.session_state.agent.invoke({"input": final_prompt})
            answer = result["output"]

        except Exception as e:
            answer = f"⚠️ Terjadi error: {e}"

    # Simpan jawaban ke session
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()
