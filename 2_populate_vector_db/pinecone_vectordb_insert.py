import os
import subprocess
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# 環境変数の取得
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
PINECONE_INDEX = os.getenv('PINECONE_INDEX')

dimension = 768

# エンべディングモデルの設定
EMBEDDING_MODEL_REPO = "sentence-transformers/all-mpnet-base-v2"

# models/embedding-model に入ったモデルをロード
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_REPO)
model = AutoModel.from_pretrained(EMBEDDING_MODEL_REPO)

# Pinecone の特定のインデックスの中にコレクションを作成する関数
def create_pinecone_collection(pinecone, PINECONE_INDEX):
    try:
        print(f"Creating 768-dimensional index called '{PINECONE_INDEX}'...")
        # Create the Pinecone index with the specified dimension.
        pinecone.create_index(
            name=PINECONE_INDEX,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws', 
                region=PINECONE_ENVIRONMENT
            ) 
        )
        print("Success")
    except:
        # インデックスが既に存在する場合はスルー
        pass

    print("Checking Pinecone for active indexes...")
    active_indexes = pinecone.list_indexes()
    print("Active indexes:")
    print(active_indexes)
    print(f"Getting description for '{PINECONE_INDEX}'...")
    index_description = pinecone.describe_index(PINECONE_INDEX)
    print("Description:")
    print(index_description)

    print(f"Getting '{PINECONE_INDEX}' as object...")
    pinecone_index = pinecone.Index(PINECONE_INDEX)
    print("Success")

    # Pinecone の index オブジェクトを返す
    return pinecone_index
    
# アテンションマスクを考慮に入れ、平均値プーリングを行う
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # model_output の最初の要素が、エンべディングされたすべてのトークンを保持している
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# エンべディングを実施
def get_embeddings(sentence):
    # エンべディングしたい文
    sentences = [sentence]

    # 文をトークナイズする。
    # デフォルトでは、モデルはドキュメントの最初の256トークンのみを切り詰めて保持します。
    # セマンティックサーチは、この最初の256トークンに対してのみ有効です。
    # コンテクストローディングは、ドキュメント全体に対して有効です。
    encoded_input = tokenizer(sentences, padding='max_length', truncation=True, return_tensors='pt')

    # トークンのエンべディング
    with torch.no_grad():
        model_output = model(**encoded_input)

    # 平均値プーリングの実行
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # エンべディングの正規化
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    return (sentence_embeddings.tolist()[0])

    
# 特定のテキスト・ドキュメントをエンべディングし、Pinecone のベクターDBに挿入する
def insert_embedding(pinecone_index, id_path, text):
    print("Upserting vectors...")
    vectors = list(zip([text[:512]], [get_embeddings(text)], [{"file_path": id_path}]))
    upsert_response = pinecone_index.upsert(
        vectors=vectors
        )
    print("Success")
    
    
def main():
    try:
        print("Pinecone のコネクションを初期化中...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        print("Pinecone のコネクション 初期化完了")
        
        # Pinecone とのコネクションを作成
        collection = create_pinecone_collection(pc, PINECONE_INDEX)
        
        # Same files are ignored (e.g. running process repetitively won't overwrite, just pick up new files)
        print("Pinecone とのコネクションが作成されました")

        # Read KB documents in ./data directory and insert embeddings into Vector DB for each doc
        doc_dir = '/home/cdsw/data'
        for file in Path(doc_dir).glob(f'**/*.txt'):
            with open(file, "r") as f: # Open file in read mode
                print("Generating embeddings for: %s" % file.name)
                text = f.read()
                # エンべディングされたベクトルを Pinecone のベクターDBに挿入
                insert_embedding(collection, os.path.abspath(file), text)
        print('Finished loading Knowledge Base embeddings into Pinecone')

    except Exception as e:
        raise (e)


if __name__ == "__main__":
    main()
