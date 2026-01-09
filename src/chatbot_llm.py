# src/chatbot_llm.py

import os
import json
from typing import List, Dict

import numpy as np
from openai import OpenAI



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

def responder(
    client: OpenAI,
    user_query: str,
    embeddings: np.ndarray,
    metadata: List[Dict],
) -> tuple:
    top_docs = retrieve_top_k(client, user_query, embeddings, metadata, k=3)
    context = build_context_from_docs(top_docs)


    messages = [
        {
            "role": "system",
            "content": (
                "És um assistente virtual de um restaurante. "
                "Responde sempre com base EXCLUSIVAMENTE nas informações abaixo. "
                "Se a pergunta não estiver coberta, diz claramente que não tens "
                "essa informação e sugere ao utilizador que contacte o restaurante "
                "por telefone."
            ),
        },
        {
            "role": "system",
            "content": f"Informação do restaurante (FAQs):\n\n{context}",
        },
        {
            "role": "user",
            "content": user_query,
        },
    ]

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
            resposta, fotos = responder(client, user_query, embeddings, metadata)
        except Exception as e:
            print(f"[ERRO] Ocorreu um problema ao gerar a resposta: {e}")
            continue

        print("Bot:", resposta)
        if fotos and str(fotos).lower() != "nan":
            print(f"[IMAGENS ASSOCIADAS]: {fotos}")
        print()


if __name__ == "__main__":
    main()
