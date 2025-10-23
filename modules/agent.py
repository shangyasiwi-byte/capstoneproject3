# modules/agent.py
# “Otak utama” yang mengatur interaksi antara retriever (Qdrant), prompt RAG, dan LLM OpenAI.

from langchain.agents import initialize_agent, Tool
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage

from modules.rag_tool import rag_chain  # pastikan rag_tool.py punya fungsi ini

# 1️⃣ Inisialisasi model LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",   # atau "gpt-4o" kalau ingin lebih pintar
    temperature=0.4        # sedikit dinaikkan agar lebih natural saat berbahasa
)

# 2️⃣ Definisikan RAG tool
tools = [
    Tool(
        name="IMDB RAG Search",
        func=rag_chain,
        description="Gunakan untuk menjawab pertanyaan tentang film IMDB, seperti rating, genre, atau plot."
    )
]

# 3️⃣ Tambahkan memory untuk menyimpan konteks chat
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 4️⃣ Sistem prompt multilingual + konteks percakapan
system_prompt = SystemMessage(content="""
Kamu adalah asisten film IMDB yang ramah dan cerdas, bisa berbahasa apa pun. Penting: Selalu balas dalam bahasa yang sama dengan pengguna, apa pun yang terjadi.
Jika pengguna berbicara dalam Bahasa Indonesia, jawablah sepenuhnya dalam Bahasa Indonesia.
Jika pengguna berbicara dalam bahasa lain, gunakan bahasa itu juga.
Tugasmu:
- Mendeteksi bahasa yang digunakan user dan membalas menggunakan bahasa yang sama dengan user.
- Menggunakan frasa yang natural dan mudah dimengerti.
- Saat membutuhkan data eksternal, gunakan tools dan pertahankan format internal reasoning (Action / Action Input / Final Answer).
- Ingat konteks percakapan dan judul film sebelumnya; untuk pertanyaan lanjutan seperti "ratingnya gimana?", gunakan film yang terakhir dibahas.
- Pastikan balasan sesuai bahasa user, singkat, bermanfaat, dan sinematik.
""")

# 5️⃣ Buat agent dengan memory dan pengaturan aman dari infinite loop
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type="chat-conversational-react-description",  # cocok untuk chat multi-turn
    verbose=True,
    handle_parsing_errors=True,
    memory=memory,
    max_iterations=6,            # batasi iterasi agar tidak infinite loop
    max_execution_time=60,       # batasi waktu eksekusi (detik)
    early_stopping_method="force",
    agent_kwargs={
        # -> Penting: letakkan system_prompt sebagai system_message agar menjadi instruksi utama
        "system_message": system_prompt,
        # -> sisipkan chat history setelah system_message
        "extra_prompt_messages": [
            system_prompt,
            MessagesPlaceholder(variable_name="chat_history"),
        ]
    }
)

# 6️⃣ Tes agent
if __name__ == "__main__":
    print("🎬 Testing context memory multilingual...")
    print(agent.invoke({"input": "Siapa sutradara film The Witch?"})["output"])
    print(agent.invoke({"input": "Ratingnya gimana?"})["output"])
