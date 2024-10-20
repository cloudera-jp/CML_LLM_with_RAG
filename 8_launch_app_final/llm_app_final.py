import os
import gradio as gr
import cmlapi
from pinecone import Pinecone
from typing import Any, Union, Optional
from pydantic import BaseModel
import tensorflow as tf
from sentence_transformers import SentenceTransformer
import requests
import json
import time
from typing import Optional
import boto3
from botocore.config import Config
import chromadb
from chromadb.utils import embedding_functions

from huggingface_hub import hf_hub_download

# 各種のベクターDBを使いたくない場合は、以下を False にします
USE_PINECONE = True 
USE_CHROMA = True 

EMBEDDING_MODEL_REPO = "sentence-transformers/all-mpnet-base-v2"


if USE_PINECONE:
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
    PINECONE_INDEX = os.getenv('PINECONE_INDEX')

    print("initialising Pinecone connection...")
    pinecone = Pinecone(api_key=PINECONE_API_KEY)
    print("Pinecone initialised")

    print(f"Getting '{PINECONE_INDEX}' as object...")
    index = pinecone.Index(PINECONE_INDEX)
    print("Success")

    # index の最新の統計情報を表示
    current_collection_stats = index.describe_index_stats()
    print('Pinecone インデックスのベクトルの数 : {}.'.format(current_collection_stats.get('total_vector_count')))

    
if USE_CHROMA:
    # ローカルの Chroma のデータに接続
    chroma_client = chromadb.PersistentClient(path="/home/cdsw/chroma-data")
    
    EMBEDDING_MODEL_REPO = "sentence-transformers/all-mpnet-base-v2"
    EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
    EMBEDDING_FUNCTION = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)

    COLLECTION_NAME = 'cml-default'

    print("initialising Chroma DB connection...")

    print(f"Getting '{COLLECTION_NAME}' as object...")
    try:
        chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION)
        print("Success")
        collection = chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION)
    except:
        print("Creating new collection...")
        collection = chroma_client.create_collection(name=COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION)
        print("Success")

    # インデックスの最新の統計情報を表示
    current_collection_stats = collection.count()
    print('Total number of embeddings in Chroma DB index is ' + str(current_collection_stats))
    
    
## TO DO GET MODEL DEPLOYMENT
## Need to get the below prgramatically in the future iterations
client = cmlapi.default_client(url=os.getenv("CDSW_API_URL").replace("/api/v1", ""), cml_api_key=os.getenv("CDSW_APIV2_KEY"))
projects = client.list_projects(include_public_projects=True, search_filter=json.dumps({"name": "Shared LLM Model for Hands on Lab"}))
project = projects.projects[0]

## このコードでは、プロジェクトの中にモデルがひとつだけデプロイされていることを想定しています。
## 複数のモデルがある場合は、 model.models[x] のインデックスを調整します。
model = client.list_models(project_id=project.id)
selected_model = model.models[0]

## モデルのアクセスキーを環境変数として保存します。
MODEL_ACCESS_KEY = selected_model.access_key

MODEL_ENDPOINT = os.getenv("CDSW_API_URL").replace("https://", "https://modelservice.").replace("/api/v1", "/model?accessKey=")
MODEL_ENDPOINT = MODEL_ENDPOINT + MODEL_ACCESS_KEY

if os.environ.get("AWS_DEFAULT_REGION") == "":
    os.environ["AWS_DEFAULT_REGION"] = "us-west-2"

    
## Bedrock クライアントの設定
def get_bedrock_client(
    endpoint_url: Optional[str] = None,
    region: Optional[str] = None,
):
    """Create a boto3 client for Amazon Bedrock, with optional configuration overrides

    Parameters
    ----------
    endpoint_url :
        Optional override for the Bedrock service API Endpoint. If setting this, it should usually
        include the protocol i.e. "https://..."
    region :
        Optional name of the AWS Region in which the service should be called (e.g. "us-east-1").
        If not specified, AWS_REGION or AWS_DEFAULT_REGION environment variable will be used.
    """
    target_region = region

    print(f"Create new client\n  Using region: {target_region}")
    session_kwargs = {"region_name": target_region}
    client_kwargs = {**session_kwargs}

    profile_name = os.environ.get("AWS_PROFILE")
    if profile_name:
        print(f"  Using profile: {profile_name}")
        session_kwargs["profile_name"] = profile_name

    retry_config = Config(
        region_name=target_region,
        retries={
            "max_attempts": 10,
            "mode": "standard",
        },
    )
    session = boto3.Session(**session_kwargs)


    if endpoint_url:
        client_kwargs["endpoint_url"] = endpoint_url

    bedrock_client = session.client(
        service_name="bedrock-runtime",
        config=retry_config,
        **client_kwargs
    )

    print("boto3 Bedrock client successfully created!")
    print(bedrock_client._endpoint)
    return bedrock_client


boto3_bedrock = get_bedrock_client(
      region=os.environ.get("AWS_DEFAULT_REGION", None))


def main():
    # gradio app の設定
    print("Configuring gradio app")

    DESC = "This AI-powered assistant showcases the flexibility of Cloudera Machine Learning to work with 3rd party solutions for LLMs and Vector Databases, as well as internally hosted models and vector DBs. The prototype does not yet implement chat history and session context - every prompt is treated as a brand new one."
    
    # Gradio のUIを作成
    demo = gr.ChatInterface(
        fn=get_responses, 
        #examples=[["What is Cloudera?", "AWS Bedrock Claude v2.1", 0.5, "100"], ["What is Apache Spark?", 0.5, "100"], ["What is CML HoL?", 0.5, "100"]], 
        title="Enterprise Custom Knowledge Base Chatbot",
        description = DESC,
        additional_inputs=[gr.Radio(['Local Mistral 7B', 'AWS Bedrock Claude v2.1'], label="Select Foundational Model", value="AWS Bedrock Claude v2.1"), 
                           gr.Slider(minimum=0.01, maximum=1.0, step=0.01, value=0.5, label="Select Temperature (Randomness of Response)"),
                           gr.Radio(["50", "100", "250", "500", "1000"], label="Select Number of Tokens (Length of Response)", value="250"),
                           gr.Radio(['None', 'Pinecone', 'Chroma'], label="Vector Database Choices", value="None")],
        retry_btn = None,
        undo_btn = None,
        clear_btn = None,
        autofocus = True
        )

    # gradio アプリの起動
    print("Launching gradio app")
    demo.launch(share=True,   
                enable_queue=True,
                show_error=True,
                server_name='127.0.0.1',
                server_port=int(os.getenv('CDSW_READONLY_PORT')))
    print("Gradio app ready")

# QAアプリに回答するためのヘルパー関数
def get_responses(message, history, model, temperature, token_count, vector_db):
    
    # ローカルのモデル
    if model == "Local Mistral 7B":
        
        if vector_db == "None":
            context_chunk = ""
            response = get_llama2_response_with_context(message, context_chunk, temperature, token_count)
        
            # Stream output to UI
            for i in range(len(response)):
                time.sleep(0.02)
                yield response[:i+1]
                
        elif vector_db == "Pinecone":
            # TODO: sub this with call to Pinecone to get context chunks
            #response = "ERROR: Pinecone is not implemented for LLama yet"
            
            # ベクター検索
            context_chunk, source, score = get_nearest_chunk_from_pinecone_vectordb(index, message)
            
            # CML でホストしているモデルの Call
            response = get_llama2_response_with_context(message, context_chunk, temperature, token_count)
            
            # 関連文書の表示
            response = f"{response}\n\n 追加情報は、こちらを参照してください: {url_from_source(source)}"
            
            # 結果をUIにストリーム表示（タイピングしているような表示）するためのループ
            # 単純に結果を出すだけであれば、 yield response だけでOK 
            for i in range(len(response)):
                time.sleep(0.02)
                yield response[:i+1]
                
        elif vector_db == "Chroma":
            # Chroma のベクトル検索
            context_chunk, source = get_nearest_chunk_from_chroma_vectordb(collection, message)
            
            # CML でホストしているモデルの Call
            response = get_llama2_response_with_context(message, context_chunk, temperature, token_count)
            
            # 関連文書の表示
            response = f"{response}\n\n 追加情報は、こちらを参照してください: {url_from_source(source)}"
            
            # 結果をUIにストリーム表示（タイピングしているような表示）するためのループ
            # 単純に結果を出すだけであれば、 yield response だけでOK 
            for i in range(len(response)):
                time.sleep(0.02)
                yield response[:i+1]
    
    elif model == "AWS Bedrock Claude v2.1":
        if vector_db == "None":
            # 文脈なしの場合
            response = get_bedrock_response_with_context(message, "", temperature, token_count)
        
            # 結果をUIにストリーム表示（タイピングしているような表示）するためのループ
            # 単純に結果を出すだけであれば、 yield response だけでOK 
            for i in range(len(response)):
                time.sleep(0.02)
                yield response[:i+1]
                
        elif vector_db == "Pinecone":

            # ベクトル検索（Pinecone）
            context_chunk, source, score = get_nearest_chunk_from_pinecone_vectordb(index, message)
            
            # モデル呼び出し（Bedrock）
            response = get_bedrock_response_with_context(message, context_chunk, temperature, token_count)
            
            # 関連文書の表示
            response = f"{response}\n\n 追加情報は、こちらを参照してください: {url_from_source(source)}"
            
            # 結果をUIにストリーム表示（タイピングしているような表示）するためのループ
            # 単純に結果を出すだけであれば、 yield response だけでOK 
            for i in range(len(response)):
                time.sleep(0.01)
                yield response[:i+1]
                
        elif vector_db == "Chroma":

            # ベクトル検索（Chroma）
            context_chunk, source = get_nearest_chunk_from_chroma_vectordb(collection, message)
            
            # モデルの呼び出し（CMLのローカルホスト）
            response = get_bedrock_response_with_context(message, context_chunk, temperature, token_count)
            
            # 関連文書の表示
            response = f"{response}\n\n 追加情報は、こちらを参照してください: {url_from_source(source)}"
            
            # 結果をUIにストリーム表示（タイピングしているような表示）するためのループ
            # 単純に結果を出すだけであれば、 yield response だけでOK 
            for i in range(len(response)):
                time.sleep(0.02)
                yield response[:i+1]

def url_from_source(source):
    url = source.replace('/home/cdsw/data/https:/', 'https://').replace('.txt', '.html')
    return f"[Reference 1]({url})"
    

# Pinecone - ユーザーの質問をエンべディングし、最も関連性の高いドキュメントを抽出
def get_nearest_chunk_from_pinecone_vectordb(index, question):

    # ユーザーの質問のエンべディング
    retriever = SentenceTransformer(EMBEDDING_MODEL_REPO)
    xq = retriever.encode([question]).tolist()
    xc = index.query(vector=xq, top_k=5,include_metadata=True)
    
    matching_files = []
    scores = []
    for match in xc['matches']:
        # メタデータの中にある file_path を抽出
        file_path = match['metadata']['file_path']
        # ベクターのスコアを抽出
        score = match['score']
        scores.append(score)
        matching_files.append(file_path)

    ## このコードでは、プロジェクトの中にモデルがひとつだけデプロイされていることを想定しています。
    ## 複数のモデルがある場合は、 model.models[x] のインデックスを調整します。
    response = load_context_chunk_from_data(matching_files[0])
    sources = matching_files[0]
    score = scores[0]
    
    print(f"Response of context chunk {response}")
    return response, sources, score

# ナレッジベースのID(ファイルパス)に基づいて、ナレッジベースの内容を返す
def load_context_chunk_from_data(id_path):
    with open(id_path, "r") as f: # 読み取りモードでファイルを開く
        return f.read()

# Chroma - ユーザーの質問をエンべディングし、最も関連性の高いドキュメントを抽出
def get_nearest_chunk_from_chroma_vectordb(collection, question):
    ## Chroma DBのクエリ
    ## 最も近い1件を抽出
    response = collection.query(
                    query_texts=[question],
                    n_results=1
                    # where={"metadata_field": "is_equal_to_this"}, # メタデータによるフィルタリング例
                    # where_document={"$contains":"search_string"}  # メタデータによるフィルタリング例
    )
    
    return response['documents'][0][0], response['ids'][0][0]

def get_bedrock_response_with_context(question, context, temperature, token_count):
    
    # 文脈の有無によるプロンプトの出し分け
    if context == "":
        instruction_text = """Human: あなたは有能で、正直で、礼儀正しいアシスタントです。もし答えを知らなければ、単純に「申し訳ありません。存じ上げません」などのように、知らない旨を答えてください。 <question></question> のタグで囲まれた質問に、正直かつ正確に答えてください。回答の中で、質問文を繰り返さないでください。
    
    <question>{{QUESTION}}</question>
                    Assistant:"""
    else:
        instruction_text = """Human: あなたは有能で、正直で、礼儀正しいアシスタントです。もし答えを知らなければ、単純に「申し訳ありません。存じ上げません」などのように、知らない旨を答えてください。<text></text> のタグで囲まれた知識を読んで、その知識に基づき、<question></question> のタグで囲まれた質問に、正直かつ正確に答えてください。回答の中で、質問文を繰り返さないでください。
    <text>{{CONTEXT}}</text>
    
    <question>{{QUESTION}}</question>
                    Assistant:"""
    
    # {{QUESTION}} と　{{CONTEXT}} のプレースホルダを置き換え、プロンプトを完成させる
    full_prompt = instruction_text.replace("{{QUESTION}}", question).replace("{{CONTEXT}}", context)
    
    # モデルが要求するスキーマで JSON オブジェクトを作成する
    body = json.dumps({"prompt": full_prompt,
             "max_tokens_to_sample":int(token_count),
             "temperature":float(temperature),
             "top_k":250,
             "top_p":1.0,
             "stop_sequences":[]
              })
    
    # モデルのIDを指定し、JSONペイロードを渡して Call
    modelId = 'anthropic.claude-v2:1'
    response = boto3_bedrock.invoke_model(body=body, modelId=modelId, accept='application/json', contentType='application/json')
    response_body = json.loads(response.get('body').read())
    print("Model results successfully retreived")
    
    result = response_body.get('completion')
    
    return result

    
# ローカルにホストされたモデル（Mistral7B）の呼び出し
def get_llama2_response_with_context(question, context, temperature, token_count):
    
    question_inst = f"あなたは有能で、正直で、礼儀正しい日本語話者のアシスタントです。以下の質問に答えてください。{str(question)} "
    context_inst = "もし答えを知らなければ、単純に「申し訳ありません。存じ上げません」などのように、知らない旨を答えてください。また、回答の中で質問文を繰り返さないでください。"

    if context != "":
        
        context_inst = context_inst +  f"以下の知識に基づいて、知識の範囲内で、正確に答えてください。{str(context)}"
    
    question_and_context = question_inst + context_inst
        
    try:
        # CML でホストするモデル用のJSONペイロードを作成
        data={ "request": {"prompt":question_and_context,"temperature":temperature,"max_new_tokens":token_count,"repetition_penalty":1.0} }
        
        r = requests.post(MODEL_ENDPOINT, data=json.dumps(data), headers={'Content-Type': 'application/json'})
        
        # ログ用
        print(f"Request: {data}")
        print(f"Response: {r.json()}")
        
        no_inst_response = str(r.json()['response']['prediction']['response'])[len(question_and_context)-6:]
            
        return no_inst_response
        
    except Exception as e:
        print(e)
        return e


if __name__ == "__main__":
    main()
