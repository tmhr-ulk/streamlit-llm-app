import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv


def get_llm_answer(input_text: str, expert_type: str) -> str:
    """
    引数:
      - input_text: ユーザー入力テキスト
      - expert_type: ラジオボタンの選択値（例: "A" / "B"）
    戻り値:
      - LLMの回答テキスト
    """
    # 専門家の種類に応じてシステムメッセージを切り替え
    if expert_type == "A":
        system_prompt = (
            "あなたはプロのソフトウェアアーキテクトです。"
            "要件を整理し、設計方針、トレードオフ、実装上の注意点を明確にしながら回答してください。"
            "不明点があれば、最小限の追加質問も提示してください。"
        )
    else:  # "B"
        system_prompt = (
            "あなたはプロの健康・生活習慣コーチです。"
            "安全第一で、実行しやすい小さなステップに分解して助言してください。"
            "医療行為が必要そうな場合は受診を促し、断定は避けてください。"
        )

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=input_text),
    ]
    # 最新のLLM呼び出し方法に合わせて invoke() を使用
    result = llm.invoke(messages)
    return result.content


def main():
    st.set_page_config(page_title="LangChain Expert Chat", page_icon="🧠", layout="centered")

    st.title("🧠 LangChain Expert Chat")
    st.write(
        """
このWebアプリは、入力したテキストを LangChain 経由で LLM に渡して回答を表示します。  
ラジオボタンで **専門家の種類（A / B）** を選ぶと、LLMの振る舞い（システムメッセージ）が切り替わります。

**使い方**
1. 「専門家の種類」を選択  
2. 下の入力フォームに質問を入力  
3. 「送信」を押すと回答が表示されます
"""
    )

    # ラジオボタン（専門家の種類）
    expert_label = st.radio(
        "専門家の種類を選択してください",
        options=["A（ソフトウェアアーキテクト）", "B（健康・生活習慣コーチ）"],
        index=0,
        horizontal=True,
    )

    # 選択値を A/B に変換
    expert_type = "A" if expert_label.startswith("A") else "B"

    # 入力フォーム（1つ）
    with st.form(key="input_form"):
        user_input = st.text_area(
            "入力フォーム（質問/相談内容）",
            placeholder="ここにテキストを入力して送信してください。",
            height=140,
        )
        submitted = st.form_submit_button("送信")

    if submitted:
        if not user_input.strip():
            st.warning("テキストを入力してください。")
            return

        with st.spinner("LLMに問い合わせ中..."):
            try:
                answer = get_llm_answer(user_input, expert_type)
            except Exception as e:
                st.error("LLM呼び出しでエラーが発生しました。設定（OPENAI_API_KEY）等を確認してください。")
                st.exception(e)
                return

        st.subheader("回答")
        st.write(answer)

    st.divider()
    st.caption(
        ""
    )


if __name__ == "__main__":
    load_dotenv()
    main()