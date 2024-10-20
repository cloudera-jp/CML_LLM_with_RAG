import os
import cmlapi
import sys
import gradio as gr
from pinecone import Pinecone
from typing import Any, Union, Optional
from pydantic import BaseModel
import tensorflow as tf
from sentence_transformers import SentenceTransformer
import requests
import json
import time
import boto3
from botocore.config import Config
from huggingface_hub import hf_hub_download


USE_PINECONE = True # Pinecone を呼び出したくない場合は、ここを false にする

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

    # Pinecone のインデックスの最新の統計情報を取得
    current_collection_stats = index.describe_index_stats()
    print('Total number of embeddings in Pinecone index is {}.'.format(current_collection_stats.get('total_vector_count')))

## TO DO GET MODEL DEPLOYMENT
## Need to get the below prgramatically in the future iterations
client = cmlapi.default_client(url=os.getenv("CDSW_API_URL").replace("/api/v1", ""), cml_api_key=os.getenv("CDSW_APIV2_KEY"))
projects = client.list_projects(include_public_projects=True, search_filter=json.dumps({"name": "LLM_local_model"}))
project = projects.projects[0]

## このコードでは、プロジェクトの中にモデルがひとつだけデプロイされていることを想定しています。
## 複数のモデルがある場合は、 model.models[x] のインデックスを調整します。
model = client.list_models(project_id=project.id)
selected_model = model.models[0]

## モデルのアクセスキーを、このプロジェクトの環境変数として保存します。
MODEL_ACCESS_KEY = selected_model.access_key

MODEL_ENDPOINT = os.getenv("CDSW_API_URL").replace("https://", "https://modelservice.").replace("/api/v1", "/model?accessKey=")
MODEL_ENDPOINT = MODEL_ENDPOINT + MODEL_ACCESS_KEY

#　デフォルトのリージョンを確認
if os.environ.get("AWS_DEFAULT_REGION") == "":
    os.environ["AWS_DEFAULT_REGION"] = "us-west-2"

    
## Bedrock クライアントの設定:
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
    # gradio QA の設定
    # gradio : LLM に特化した Web アプリを Python コードのみでデプロイできるツール （詳細：https://www.gradio.app/）
    print("Configuring gradio app")

    DESC = "このアプリは、サードパーティーのLLMモデルやベクターDBを使って、企業固有の知識を反映した回答を返すチャットボットのデモンストレーションです。このプロトタイプには、チャットの履歴機能は実装されておらず、すべてのプロンプトは新しいものとして扱われます。" 

    # Gradio の Web UI を作成する
    demo = gr.ChatInterface(
        fn=get_responses, 
        title="企業固有の知識を持つチャットボット",
        description = DESC,
        additional_inputs=[gr.Radio(['Local Mistral 7B', 'AWS Bedrock Claude v2.1'], label="Select Foundational Model", value="Local Mistral 7B"), 
                           gr.Slider(minimum=0.01, maximum=1.0, step=0.01, value=0.5, label="Temperature (回答のランダムさ)を選択してください"),
                           gr.Radio(["50", "100", "250", "500", "1000"], label="トークン（≒回答の単語）の数を選択してください", value="250"),
                           gr.Radio(['なし', 'Pinecone'], label="ベクターDBを選択してください", value="なし")],
        retry_btn = None,
        undo_btn = None,
        clear_btn = None,
        autofocus = True
        )

    # gradio のアプリを起動
    print("gradio アプリを起動")
    demo.launch(share=True,   
                enable_queue=True,
                show_error=True,
                server_name='127.0.0.1',
                server_port=int(os.getenv('CDSW_READONLY_PORT')))
    print("gradio アプリの準備ができました")

# QAアプリにレスポンスを返すためのヘルパー関数
def get_responses(message, history, model, temperature, token_count, vector_db):
    
    
    if model == "Local Mistral 7B":

        if vector_db == "なし":

            context_chunk = ""
            
            response = get_llama2_response_with_context(message, context_chunk, temperature, token_count)
            
            yield response
            
        elif vector_db == "Pinecone":
            
            context_chunk, source, score = get_nearest_chunk_from_pinecone_vectordb(index, message)
            
            response = get_llama2_response_with_context(message, context_chunk, temperature, token_count)
            
            response = f"{response}\n\n 追加情報はこちらをご覧ください: {url_from_source(source)}"

            yield response
            
                    
    elif model == "AWS Bedrock Claude v2.1":
        if vector_db == "なし":
            # No context call Bedrock
            context_chunk = ""
            response = get_bedrock_response_with_context(message, context_chunk, temperature, token_count)
        
            # Stream output to UI
            for i in range(len(response)):
                time.sleep(0.01)
                yield response[:i+1]
                
        elif vector_db == "Pinecone":
            # Vector search the index
            context_chunk, source, score = get_nearest_chunk_from_pinecone_vectordb(index, message)
            
            # Call Bedrock model
            response = get_bedrock_response_with_context(message, context_chunk, temperature, token_count)
            
            response = f"{response}\n\n 追加情報はこちらをご覧ください: {url_from_source(source)}"
            
            # Stream output to UI
            for i in range(len(response)):
                time.sleep(0.01)
                yield response[:i+1]

def url_from_source(source):
    url = source.replace('/home/cdsw/data/https:/', 'https://').replace('.txt', '.html')
    return f"[Reference 1]({url})"
    

# ベクターDB（Pinecone）から、質問と最も近いナレッジベースを取得する
def get_nearest_chunk_from_pinecone_vectordb(index, question):

    # ユーザーの質問をエンべディングする
    retriever = SentenceTransformer(EMBEDDING_MODEL_REPO)
    xq = retriever.encode([question]).tolist()
    xc = index.query(vector=xq, top_k=5,include_metadata=True)
    
    matching_files = []
    scores = []
    for match in xc['matches']:
        # ベクターDBのメタデータの中から 'file_path' を取得
        file_path = match['metadata']['file_path']
        # 各ベクトルのスコアを抽出
        score = match['score']
        scores.append(score)
        matching_files.append(file_path)

    # 最も近いナレッジベースのテキストを返す
    # このコードでは、最も近いナレッジベースをひとつだけ使っていることに注意
    # ユースケースによっては、単純に上位1件の抽出ではなく、スコアなどを加味して複数のナレッジベースを取得することが適切な場合もあります。
    response = load_context_chunk_from_data(matching_files[0])
    sources = matching_files[0]
    score = scores[0]
    
    print(f"Response of context chunk {response}")
    return response, sources, score

# ナレッジベースのID(ファイルパス)に基づいて、ナレッジベースの内容を返す
def load_context_chunk_from_data(id_path):
    with open(id_path, "r") as f: # 読み取りモードでファイルを開く
        return f.read()


# Bedrock のモデル（Claude）の呼び出し
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
        
        # Following LLama's spec for prompt engineering
        #llama_sys = f"<<SYS>>\n You are a helpful, respectful and honest assistant. If you are unsurae about an answer, truthfully say \"I don't know\".\n<</SYS>>\n\n"
        #llama_inst = f"[INST]Use your own knowledge and additionally use the following information to answer the user's question: {context} [/INST]"
        #question_and_context = f"{llama_sys} {llama_inst} [INST] User: {question} [/INST]"
        
        data={ "request": {"prompt":question_and_context,"temperature":temperature,"max_new_tokens":token_count,"repetition_penalty":1.0} }
        
        r = requests.post(MODEL_ENDPOINT, data=json.dumps(data), headers={'Content-Type': 'application/json'})
        
        # ログ用
        print(f"Request: {data}")
        print(f"Response: {r.json()}")
        
        no_inst_response = r.json()['response']['prediction']['response']
            
        return no_inst_response
        
    except Exception as e:
        print(e)
        return e


if __name__ == "__main__":
    main()
