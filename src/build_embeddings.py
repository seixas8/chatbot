# src/build_embeddings.py

import os
import json
import numpy as np
import pandas as pd
from openai import OpenAI

def main():
    client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

    csv_path = os.path.join("data", "dataset.csv")
    faq_df = pd.read_csv(csv_path)

    documents = []
    metadata = [] 
    for _, row in faq_df.iterrows():
        pergunta = str(row["pergunta"])
        resposta = str(row["resposta"])
        categoria = str(row.get("categoria", ""))
        lista_imagens = str(row.get("imagem", ""))

        doc_text = f"PERGUNTA: {pergunta}\nRESPOSTA: {resposta}"
        documents.append(doc_text)

        metadata.append({
            "pergunta": pergunta,
            "resposta": resposta,
            "categoria": categoria,
            "imagem": lista_imagens
        })

    print(f"Total de documentos: {len(documents)}")

    
    print("A gerar embeddings, isto pode demorar alguns segundos...")

   
    response = client.embeddings.create(
        model="text-embedding-3-large", 
        input=documents,
    )

    embeddings = []
    for item in response.data:
        embeddings.append(item.embedding)

    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"Formato da matriz de embeddings: {embeddings.shape}")

    os.makedirs("models", exist_ok=True)

    emb_path = os.path.join("models", "faq_embeddings.npy")
    meta_path = os.path.join("models", "faq_metadata.json")

    np.save(emb_path, embeddings)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Embeddings guardados em: {emb_path}")
    print(f"Metadata guardada em: {meta_path}")
    print("Concluído com sucesso.")

if __name__ == "__main__":
    main()
