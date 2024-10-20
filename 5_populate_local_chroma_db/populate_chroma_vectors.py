import os

## Chroma DB との接続を初期化
import chromadb
from pathlib import Path

chroma_client = chromadb.PersistentClient(path="/home/cdsw/chroma-data")

from chromadb.utils import embedding_functions
EMBEDDING_MODEL_REPO = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
EMBEDDING_FUNCTION = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)

COLLECTION_NAME = 'cml-default'

print("Chroma DB との接続を初期化しています...")

print(f"'{COLLECTION_NAME}' をオブジェクトとして作成 or 取得します...")
try:
    chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION)
    print("Success")
    collection = chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION)
except:
    print("新しいコレクションを作成しています...")
    collection = chroma_client.create_collection(name=COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION)
    print("Success")

# インデックスの最新の統計情報を表示
current_collection_stats = collection.count()
print('Chroma DB のインデックスにあるエンべディングの件数: ' + str(current_collection_stats))

# Chrom にドキュメントを追加するためのヘルパー関数
def upsert_document(collection, document, metadata=None, classification="public", file_path=None):
    
    # ドキュメントを Chroma DB にプッシュする
    if file_path is not None:
        response = collection.add(
            documents=[document],
            metadatas=[{"classification": classification}],
            ids=[file_path]
        )
    else:
        # ファイルパスが取得できない場合は、ドキュメントの最初の50文字を使用する
        response = collection.add(
            documents=[document],
            metadatas=[{"classification": classification}],
            ids=document[:50]
        )
    return response

# ナレッジベース ID (相対ファイルパス) に基づいて ナレッジベース（ドキュメント）を返す
def load_context_chunk_from_data(id_path):
    with open(id_path, "r") as f: # 読み取りモードでファイルを開く
        return f.read()

# ./dataディレクトリにあるナレッジベース（ドキュメント）を読み込み、各ドキュメントのエンベディングをVector DBに挿入
doc_dir = '/home/cdsw/data'
for file in Path(doc_dir).glob(f'**/*.txt'):
    print(file)
    with open(file, "r") as f: # 読み取りモードでファイルを開く
        print("Generating embeddings for: %s" % file.name)
        text = f.read()
        upsert_document(collection=collection, document=text, file_path=os.path.abspath(file))
print('Chroma DB へのナレッジベース挿入が完了しました')

