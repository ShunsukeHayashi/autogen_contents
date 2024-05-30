import os
import streamlit as st
import asyncio
from serpapi import GoogleSearch
from dotenv import load_dotenv
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# 環境変数を読み込む
load_dotenv()

# ページのタイトルと説明を設定
st.set_page_config(page_title="SEO最適化された記事作成ワークフロー", page_icon="✍️", layout="wide")

st.markdown("このアプリケーションは、SEO最適化された記事を作成するためのワークフローを自動化します。")
st.markdown("始めるには、OpenAIのAPIキーとSerpAPIのAPIキーを取得してください。")

# LLMの設定
llm_config = {
    "config_list": [{
        "model": "gpt-4",
        "api_key": os.environ.get("OPENAI_API_KEY")
    }]
}

# SerpAPIの設定
serpapi_api_key = os.environ.get("SERPAPI_API_KEY")

# SerpAPIを使用してキーワードを取得する関数
def get_keywords_from_serpapi(query):
    search = GoogleSearch({
        "q": query,
        "api_key": serpapi_api_key,
        "location": "Tokyo, Japan",
        "hl": "ja",
        "gl": "jp"
    })
    results = search.get_dict()
    keywords = []
    for result in results.get("related_searches", []):
        keywords.append(result.get("query"))
    return keywords

# 各エージェントの作成
persona_analysis_agent = AssistantAgent(
    name="PersonaAnalysisAgent",
    system_message="あなたはペルソナ分析の専門家です。提供された記事構成リストを分析し、ターゲットオーディエンスを特定してください。",
    llm_config=llm_config
)

title_creation_agent = AssistantAgent(
    name="TitleCreationAgent",
    system_message="あなたは、魅力的な記事タイトルを作成する専門家です。提供されたペルソナに基づいて、記事タイトルを複数提案してください。",
    llm_config=llm_config
)

heading_structure_agent = AssistantAgent(
    name="HeadingStructureAgent",
    system_message="あなたは、記事の見出し構成の専門家です。提供された記事タイトルとコンテンツ分析に基づいて、論理的で魅力的な見出し構造を決定してください。",
    llm_config=llm_config
)

instruction_creation_agent = AssistantAgent(
    name="InstructionCreationAgent",
    system_message="あなたは、記事の各見出しに対する詳細な執筆指示を作成する専門家です。提供された見出し構造に基づいて、各見出しに合わせた具体的な指示を作成してください。",
    llm_config=llm_config
)

text_design_agent = AssistantAgent(
    name="TextDesignAgent",
    system_message="あなたは、記事の文章構成を設計する専門家です。提供された執筆指示に基づいて、記事の文章構成を設計してください。",
    llm_config=llm_config
)

text_generation_agent = AssistantAgent(
    name="TextGenerationAgent",
    system_message="あなたは、記事の文章を生成する専門家です。提供された文章構成に基づいて、各段落の文章を生成してください。",
    llm_config=llm_config
)

text_evaluation_agent = AssistantAgent(
    name="TextEvaluationAgent",
    system_message="あなたは、生成された記事の文章を評価する専門家です。提供された記事の文章を、正確性、関連性、明瞭さ、文章の質、ターゲットオーディエンスへの適合性の観点から評価してください。",
    llm_config=llm_config
)

final_confirmation_agent = AssistantAgent(
    name="FinalConfirmationAgent",
    system_message="あなたは、生成された記事の最終確認と承認を行う責任者です。提供された記事と評価結果を参考に、記事の最終的な確認と修正を行い、公開に適しているかどうかを判断してください。",
    llm_config=llm_config
)

# エージェントのリスト
agents = [
    persona_analysis_agent,
    title_creation_agent,
    heading_structure_agent,
    instruction_creation_agent,
    text_design_agent,
    text_generation_agent,
    text_evaluation_agent,
    final_confirmation_agent
]

# 発言順序のロジックを実装する関数
def custom_speaker_selection_func(last_speaker, groupchat):
    if last_speaker is persona_analysis_agent:
        return title_creation_agent
    elif last_speaker is title_creation_agent:
        return heading_structure_agent
    elif last_speaker is heading_structure_agent:
        return instruction_creation_agent
    elif last_speaker is instruction_creation_agent:
        return text_design_agent
    elif last_speaker is text_design_agent:
        return text_generation_agent
    elif last_speaker is text_generation_agent:
        return text_evaluation_agent
    elif last_speaker is text_evaluation_agent:
        return final_confirmation_agent
    elif last_speaker is final_confirmation_agent:
        return None  # terminate chat

# グループチャットの作成
groupchat = GroupChat(
    agents=agents,
    messages=[],
    max_round=20,
    speaker_selection_method=custom_speaker_selection_func
)

# UserProxyAgent インスタンスの作成
user_proxy = UserProxyAgent(
    name="UserProxy",
    llm_config=llm_config
)

# ワークフローの各ステップを定義
workflow_steps = [
    {"name": "キーワード入力", "agent": None, "description": "ブログ記事の作成に使用するキーワードを入力してください。", "input": "", "output": "", "requires_human_input": True},
    {"name": "ペルソナ分析", "agent": persona_analysis_agent, "description": "提供された記事構成リストを分析し、ターゲットオーディエンスを特定する", "input": "記事構成リスト", "output": "ペルソナ設定", "requires_human_input": False},
    {"name": "タイトル作成", "agent": title_creation_agent, "description": "定義されたペルソナに基づいて記事タイトルを作成する", "input": "ペルソナ設定", "output": "記事タイトル", "requires_human_input": False},
    {"name": "見出し構成", "agent": heading_structure_agent, "description": "記事タイトルとコンテンツ分析に基づいて見出し構成を決定する", "input": "記事タイトル", "output": "見出し構成", "requires_human_input": False},
    {"name": "インストラクション作成", "agent": instruction_creation_agent, "description": "提供された見出し構成に基づいて、各見出しの詳細なインストラクションを作成する", "input": "見出し構成", "output": "インストラクションリスト", "requires_human_input": False},
    {"name": "文章設計", "agent": text_design_agent, "description": "提供されたインストラクションに基づいて、記事の文章構成を設計する", "input": "インストラクションリスト", "output": "文章設計", "requires_human_input": False},
    {"name": "文章生成", "agent": text_generation_agent, "description": "提供された文章設計に基づいて、記事の文章を生成する", "input": "文章設計", "output": "記事", "requires_human_input": False},
    {"name": "文章評価", "agent": text_evaluation_agent, "description": "生成された記事の文章を評価する", "input": "記事", "output": "評価結果", "requires_human_input": False},
    {"name": "最終確認", "agent": final_confirmation_agent, "description": "生成された文章の最終確認と修正を行う", "input": "評価結果", "output": "最終記事", "requires_human_input": False}
]

# ワークフローを実行する関数
def run_workflow_step(step_index, user_input=None):
    step = workflow_steps[step_index]
    agent = step["agent"]
    
    if step["name"] == "キーワード入力":
        # SerpAPIを使用してキーワードを取得
        keywords = get_keywords_from_serpapi(user_input)
        response = f"関連するキーワード: {keywords}"
        # 次のステップに進む
        workflow_steps[step_index + 1]["input"] = keywords
        return {"next_step": step_index + 1, "response": response}
    
    if user_input:
        agent.initiate_chat(user_proxy, message=user_input)
    
    agent.initiate_chat(user_proxy, message=f"プロジェクトを開始します。次のタスクは{step['name']}です。{step['description']}")
    
    if step_index + 1 < len(workflow_steps):
        return {"next_step": step_index + 1, "response": f"{step['name']}が完了しました。次のステップに進みます。"}
    else:
        return {"next_step": None, "response": "ワークフロー完了"}

# サイドバー: APIキーの入力
with st.sidebar:
    st.header("APIキーの設定")
    st.markdown("OpenAIとSerpAPIのAPIキーを入力してください。")
    openai_key = st.text_input("OpenAI APIキー", type="password")
    serpapi_key = st.text_input("SerpAPI APIキー", type="password")

# メインエリア: ユーザー入力とチャットメッセージ
with st.container():
    user_input = st.text_input("キーワードを入力")
    
    # ユーザー入力が空でなく、APIキーが設定されている場合のみ実行
    if user_input:
        if not openai_key or not serpapi_key:
            st.warning("OpenAIとSerpAPIの有効なAPIキーを提供する必要があります。", icon="⚠️")
            st.stop()
        
        # APIキーを環境変数に設定
        os.environ["OPENAI_API_KEY"] = openai_key
        os.environ["SERPAPI_API_KEY"] = serpapi_key

        # イベントループを作成: 非同期関数を実行するために必要
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # 非同期関数を定義: awaitを使用するために必要
        if "chat_initiated" not in st.session_state:
            st.session_state.chat_initiated = False  # セッション状態を初期化

        if not st.session_state.chat_initiated:
            async def initiate_chat():
                for step in workflow_steps:
                    st.sidebar.header(step["name"])
                    st.sidebar.write(step["description"])
                    
                    agent = step["agent"]
                    description = step["description"]
                    
                    # エージェントにメッセージを送信
                    st.write(f"Executing step: {step['name']} - {description}")
                    
                    # 人間の入力が必要なステップの場合
                    if step["requires_human_input"]:
                        result = run_workflow_step(workflow_steps.index(step), user_input)
                        st.write(result["response"])
                    else:
                        result = run_workflow_step(workflow_steps.index(step))
                        st.write(result["response"])
                        
                    # 次のステップに進む
                    if result["next_step"] is None:
                        break
                
                st.session_state.chat_initiated = True  # チャットが実行された後、状態をTrueに設定

            # イベントループ内で非同期関数を実行
            loop.run_until_complete(initiate_chat())

            # イベントループを閉じる
            loop.close()

# 終了コマンド後にアプリを停止
st.stop()