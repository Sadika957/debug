import os
import numpy as np
from chromadb import PersistentClient

def check_dim(path):
    print("\n===============================")
    print("Checking DB at:", path)

    try:
        client = PersistentClient(path=path)

        # Try default collection first
        try:
            col = client.get_default_collection()
        except:
            cols = client.list_collections()
            if len(cols) == 0:
                print("❌ No collections found")
                return
            col = client.get_collection(cols[0].name)

        data = col.get(include=["embeddings"], limit=1)

        embeddings = data.get("embeddings")
        if not embeddings:
            print("❌ No embeddings found")
            return

        emb = embeddings[0]
        emb = np.array(emb)
        print("✅ Embedding dimension:", emb.shape[0])

    except Exception as e:
        print("❌ ERROR:", e)


# -------- RUN ON STREAMLIT CLOUD ----------
base = os.path.dirname(os.path.abspath(__file__))

check_dim(os.path.join(base, "chroma_db_nomic"))
check_dim(os.path.join(base, "chroma_db_jsonl"))

print("\n=== DONE CHECKING DIMENSIONS ===")
