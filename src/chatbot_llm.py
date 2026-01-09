# src/chatbot_llm.py

import os
import json
import csv
import math
from typing import List, Dict
from datetime import datetime
import numpy as np
from openai import OpenAI

PASTA_RESERVAS = os.path.join("data", "reservas.csv")
LIMITE_MESAS = 15

def create_openrouter_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
        )
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "http://localhost:5000",
            "X-Title": "Chatbot Restaurante LLM",
        },
    )
    return client


def load_embeddings_and_metadata():
    emb_path = os.path.join("models", "faq_embeddings.npy")
    meta_path = os.path.join("models", "faq_metadata.json")

    if not os.path.exists(emb_path):
        raise FileNotFoundError(
            f"Ficheiro de embeddings não encontrado em {emb_path}. "
        )
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Ficheiro de metadata não encontrado em {meta_path}. "
        )

    embeddings = np.load(emb_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return embeddings, metadata


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def embed_text(client: OpenAI, text: str) -> np.ndarray:
    response = client.embeddings.create(
        model="openai/text-embedding-3-large",
        input=text,
    )
    vec = np.array(response.data[0].embedding, dtype=np.float32)
    return vec


def retrieve_top_k(
    client: OpenAI,
    user_query: str,
    embeddings: np.ndarray,
    metadata: List[Dict],
    k: int = 3,
) -> List[Dict]:
    q_emb = embed_text(client, user_query)
    sims = []

    for idx, emb in enumerate(embeddings):
        sim = cosine_sim(q_emb, emb)
        sims.append((idx, sim))

    sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True)
    top_idx = [idx for idx, _ in sims_sorted[:k]]

    top_docs = [metadata[i] for i in top_idx]
    return top_docs


def build_context_from_docs(top_docs: List[Dict]) -> str:
    parts = []
    for d in top_docs:
        p = d.get("pergunta", "")
        r = d.get("resposta", "")
        cat = d.get("categoria", "")

        linha = (
            f"PERGUNTA EXEMPLO ({cat}): {p}\n"
            f"RESPOSTA OFICIAL: {r}"
        )
        parts.append(linha)

    context = "\n\n".join(parts)
    return context

def calcular_mesas_necessarias(num_pessoas):
    return math.ceil(num_pessoas / 4)

def verificar_disponibilidade(data_pretendida, hora_pretendida):
    if not os.path.exists(PASTA_RESERVAS):
        return 0
    mesas_ocupadas = 0
    with open(PASTA_RESERVAS, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Data'] == data_pretendida and row['Hora'] == hora_pretendida:
                mesas_ocupadas += int(row['Mesas'])
    return mesas_ocupadas

def registar_reserva(nome, data, hora, pessoas):
    mesas = calcular_mesas_necessarias(pessoas)
    file_exists = os.path.isfile(PASTA_RESERVAS)
    os.makedirs("data", exist_ok=True)
    with open(PASTA_RESERVAS, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Data', 'Hora', 'Nome', 'Pessoas', 'Mesas', 'Data_Registo'])
        writer.writerow([data, hora, nome, pessoas, mesas, datetime.now().strftime("%Y-%m-%d %H:%M")])

def responder(
    client: OpenAI,
    user_query: str,
    historico,
    embeddings: np.ndarray,
    metadata: List[Dict],
) -> tuple:
    top_docs = retrieve_top_k(client, user_query, embeddings, metadata, k=3)
    context = build_context_from_docs(top_docs)
    hoje = datetime.now().strftime("%Y-%m-%d")


    messages = [
        {
            "role": "system",
            "content": (
                f"És um assistente virtual de um restaurante. Hoje é {hoje}.\n"
            "REGRAS OBRIGATÓRIAS:\n"
            "1. Responde com base no contexto (FAQs). Se não souberes, pede para ligarem.\n"
            "2. Para reservar, precisas de: Nome, Data (AAAA-MM-DD), Hora e Pessoas. "
            "Não te esqueças do que o utilizador já disse nas mensagens anteriores.\n"
            "3. PROCESSO DE CONFIRMAÇÃO:\n"
            "   - Quando tiveres os 4 dados, faz um resumo e pergunta: 'Posso confirmar?'\n"
            "   - APENAS quando o utilizador disser 'sim', 'pode', 'confirma', deves gerar o código no fim da resposta: "
            "[RESERVA|Nome|Data|Hora|Pessoas]"
            ),
        },
        {
            "role": "system",
            "content": f"Informação do restaurante (FAQs):\n\n{context}",
        },
    ]

    messages.extend(historico)
    messages.append({"role": "user", "content": user_query})

    completion = client.chat.completions.create(
        model="openai/gpt-4o-mini",  
        messages=messages,
        temperature=0.2,
    )

    resposta = completion.choices[0].message.content
    imagens_encontradas = top_docs[0].get("imagem","")
    return resposta, imagens_encontradas


def main():
    client = create_openrouter_client()
    embeddings, metadata = load_embeddings_and_metadata()
    historico_conversa = []

    print("Chatbot do restaurante com LLM + RAG via OpenRouter")
    print("Escreve 'sair' para terminar.\n")

    while True:
        try:
            user_query = input("Tu: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Até à próxima!")
            break

        if user_query.lower() in ("sair", "exit", "quit"):
            print("Bot: Até à próxima!")
            break

        try:
            resposta, fotos = responder(client, user_query,historico_conversa, embeddings, metadata)
            
            historico_conversa.append({"role": "user", "content": user_query})
            historico_conversa.append({"role": "assistant", "content": resposta})
            
            if len(historico_conversa) > 10:
                historico_conversa = historico_conversa[-10:]

            if "[RESERVA|" in resposta:
                try:
                    dados_brutos = resposta.split("[RESERVA|")[1].split("]")[0]
                    nome, data, hora, pessoas_str = dados_brutos.split("|")
                    num_pessoas = int(pessoas_str)
                    
                    mesas_precisas = calcular_mesas_necessarias(num_pessoas)
                    ocupacao_atual = verificar_disponibilidade(data, hora)

                    if ocupacao_atual + mesas_precisas <= LIMITE_MESAS:
                        registar_reserva(nome, data, hora, num_pessoas)
                        print("Bot:", resposta.split("[RESERVA|")[0].strip())
                        print(f"✅ [SISTEMA]: Reserva confirmada em nome de {nome}.")
                    else:
                        print("Bot: Lamento, mas já não temos mesas para essa hora. Temos ocupadas", ocupacao_atual, "mesas.")
                except Exception as e:
                    print("Bot:", resposta)
            else:
                print("Bot:", resposta)

            if fotos and str(fotos).lower() != "nan":
                print(f"[IMAGENS ASSOCIADAS]: {fotos}")
            
        except Exception as e:
            print(f"[ERRO]: {e}")
        print()


if __name__ == "__main__":
    main()
