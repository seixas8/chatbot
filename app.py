import os
import streamlit as st
from src.chatbot_llm import (
    create_openrouter_client,
    load_embeddings_and_metadata,
    responder,
)

st.set_page_config(page_title="Chatbot Restaurante", page_icon="🍽")

st.title("Chatbot do Restaurante")
st.write("Fala com o nosso assistente virtual de restaurante.")

@st.cache_resource
def init_rag():
    client = create_openrouter_client()
    embeddings, metadata = load_embeddings_and_metadata()
    return client, embeddings, metadata

try:
    client, embeddings, metadata = init_rag()
except Exception as e:
    st.error("A app não conseguiu iniciar. Vê o erro abaixo:")
    st.exception(e)
    st.stop()

if "history" not in st.session_state:
    st.session_state["history"] = [] 

if "llm_history" not in st.session_state:
    st.session_state["llm_history"] = []  
user_input = st.text_input("Cliente:")

if st.button("Enviar") and user_input.strip():
    try:
        resposta, fotos = responder(
            client,
            user_input,
            st.session_state["llm_history"],
            embeddings,
            metadata,
        )

        st.session_state["llm_history"].append(
            {"role": "user", "content": user_input}
        )
        st.session_state["llm_history"].append(
            {"role": "assistant", "content": resposta}
        )

        st.session_state["history"].append(("Cliente", user_input))
        st.session_state["history"].append(("Bot", resposta))

        if fotos and str(fotos).strip().lower() != "nan":
            raw = str(fotos)
            nomes = [n.strip() for n in raw.split(",") if n.strip()]

            caminhos = []
            for nome in nomes:
                if os.path.exists(nome):
                    caminhos.append(nome)
                else:
                    caminho_rel = os.path.join("imagens", nome)
                    caminhos.append(caminho_rel)

           
            if len(caminhos) == 1:
                st.image(caminhos[0], caption="Sugestão do menu")
            else:
                st.image(caminhos) 

    except Exception as e:
        st.error("Ocorreu um erro ao gerar a resposta:")
        st.exception(e)
        st.stop()

for speaker, text in st.session_state["history"]:
    if speaker == "Cliente":
        st.markdown(f"**Cliente:** {text}")
    else:
        st.markdown(f"**Bot:** {text}")
