import os
import streamlit as st
from src.chatbot_llm import (
    create_openrouter_client,
    load_embeddings_and_metadata,
    responder,
    calcular_mesas_necessarias,
    verificar_disponibilidade,
    registar_reserva,
    LIMITE_MESAS,
)

try:
    from src.calendar_service import create_reservation_event
except ImportError:
    from calendar_service import create_reservation_event


st.set_page_config(page_title="Chatbot Restaurante", page_icon="🍽")

st.title("Chatbot do Restaurante")
st.write("Fala com o nosso assistente virtual do restaurante.")


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



def process_reservation_if_any(bot_reply: str):

    sistema_msgs = []

    if "[RESERVA|" not in bot_reply:
        return bot_reply, sistema_msgs

    texto_visivel = bot_reply.split("[RESERVA|")[0].strip()

    try:
        dados_brutos = bot_reply.split("[RESERVA|")[1].split("]")[0]
        nome, data, hora, pessoas_str = [x.strip() for x in dados_brutos.split("|")]
        num_pessoas = int(pessoas_str)

        mesas_precisas = calcular_mesas_necessarias(num_pessoas)
        ocupacao_atual = verificar_disponibilidade(data, hora)

        if ocupacao_atual + mesas_precisas <= LIMITE_MESAS:
            registar_reserva(nome, data, hora, num_pessoas)
            sistema_msgs.append(
                f"Reserva confirmada em nome de {nome} para {num_pessoas} pessoas em {data} às {hora}."
            )

            try:
                event_id = create_reservation_event(
                    name=nome,
                    people=num_pessoas,
                    date_str=data,
                    time_str=hora,
                    phone=None,
                    notes=None,
                )
                sistema_msgs.append(f"Evento criado no Google Calendar (ID: {event_id}).")
            except Exception as e:
                sistema_msgs.append(
                    f"Reserva registada, mas falhou a criação no Google Calendar: {e}"
                )

        else:
            sistema_msgs.append(
                "Lamento, mas já não existem mesas suficientes para esse horário. "
                f"Mesas ocupadas: {ocupacao_atual}, mesas necessárias: {mesas_precisas}."
            )

    except Exception as e:
        sistema_msgs.append(f"Erro ao processar dados da reserva: {e}")
        texto_visivel = bot_reply

    return texto_visivel, sistema_msgs


def show_images_if_any(fotos):
    if not fotos or str(fotos).strip().lower() == "nan":
        return

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
    elif len(caminhos) > 1:
        st.image(caminhos)



user_input = st.text_input("Cliente:")

if st.button("Enviar") and user_input.strip():
    try:
        bot_reply, fotos = responder(
            client,
            user_input,
            st.session_state["llm_history"],
            embeddings,
            metadata,
        )

        st.session_state["llm_history"].append({"role": "user", "content": user_input})
        st.session_state["llm_history"].append({"role": "assistant", "content": bot_reply})

        texto_bot, sistema_msgs = process_reservation_if_any(bot_reply)

        st.session_state["history"].append({
            "speaker": "Cliente",
            "text": user_input,
            "images": None,
        })

        st.session_state["history"].append({
            "speaker": "Bot",
            "text": texto_bot,
            "images": fotos,
        })

        for msg in sistema_msgs:
            lower_msg = msg.lower()

            if "reserva confirmada" in lower_msg or "evento criado" in lower_msg:
                st.success(msg)
            elif ("lamento" in lower_msg 
                  or "falhou" in lower_msg 
                  or "erro" in lower_msg):
                st.warning(msg)
            else:
                st.info(msg)

    except Exception as e:
        st.error("Ocorreu um erro ao gerar a resposta:")
        st.exception(e)


for msg in st.session_state["history"]:
    speaker = msg["speaker"]
    text = msg["text"]
    fotos = msg.get("images")

    if speaker == "Cliente":
        st.markdown(f"**Cliente:** {text}")
    else:
        st.markdown(f"**Bot:** {text}")
        if fotos:
            show_images_if_any(fotos)
