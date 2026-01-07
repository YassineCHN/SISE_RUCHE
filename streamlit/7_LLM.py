from dotenv import find_dotenv, load_dotenv
import numpy
import streamlit as st
import os
from mistralai import Mistral


load_dotenv(find_dotenv())

st.set_page_config(
    page_title="ChatBot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


class MistralAPI:

    def __init__(self, model: str) -> None:
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError(
                "No MISTRAL_API_KEY as environment variable, please set it!"
            )
        self.client = Mistral(api_key=api_key)
        self.model = model

    def query(
        self,
        query: str,
        temperature: float = 0.5,
        max_tokens: int = 50,
        top_p: float = 0.1,
    ) -> str:

        chat_response = self.client.chat.complete(
            model=self.model,
            temperature=temperature,
            top_p=top_p,  # proba cumulative
            max_tokens=max_tokens,  # calibre la taille de la réponse généré
            messages=[
                {
                    "role": "user",
                    "content": query,
                },
            ],
        )
        print(chat_response.choices[0].message.content)
        return chat_response.choices[0].message.content

    def build_prompt(self, history: str, query: str) -> list[dict[str, str]]:
        history_prompt = f"""
        # Historique de conversation:
        {history}
        """
        query_prompt = f"""
        # Question:
        {query}

        # Réponse:
        """
        return [
            {"role": "system", "content": history_prompt},
            {"role": "user", "content": query_prompt},
        ]


col1, col2 = st.columns([1, 2])

with col1:
    generation_model = st.selectbox(
        label="Choose your LLM",
        options=[
            "mistral-small-latest",
            "ministral-14b-2512",
            "ministral-3b-2512",
        ],
    )
    llm = MistralAPI(model=generation_model)

with col2:
    role_prompt = st.text_area(
        label="Le rôle du chatbot",
        value="""Tu es un agent conversationnel. Tu vas recevoir la description d'une offre et tu devras analyser l'offre pour la reformater dans un format spécifique.""",
    )

col_max_tokens, col_temperature, _ = st.columns([0.25, 0.25, 0.5])
with col_max_tokens:
    max_tokens = st.select_slider(
        label="Output max tokens", options=list(range(200, 2000, 50))
    )

with col_temperature:
    range_temperature = [round(x, 2) for x in list(numpy.linspace(0, 5, num=50))]
    temperature = st.select_slider(label="Temperature", options=range_temperature)


if "messages" not in st.session_state:
    st.session_state.messages = []

# On affiche les messages de l'utilisateur et de l'IA entre chaque message
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Si présence d'un input par l'utilisateur,
if query := st.chat_input(""):
    if query.strip():
        # On affiche le message de l'utilisateur
        with st.chat_message("user"):
            st.markdown(query)
        # On ajoute le message de l'utilisateur dans l'historique de la conversation
        st.session_state.messages.append({"role": "user", "content": query})
        # On récupère la réponse du chatbot à la question de l'utilisateur
        response = llm(
            query=query,
            history=st.session_state.messages,
        )
        # On affiche la réponse du chatbot
        with st.chat_message("assistant"):
            st.markdown(response)
        # On ajoute le message du chatbot dans l'historique de la conversation
        st.session_state.messages.append({"role": "assistant", "content": response})
    # On ajoute un bouton pour réinitialiser le chat

if st.button("Réinitialiser le Chat", type="primary"):
    st.session_state.messages = []
    st.rerun()


llm.query(
    query="Peux tu me dire en une phrase, quel est le métier le plus compliqué dans le domaine de la Data?",
    temperature=1,
    max_tokens=15,
    top_p=0.5,
)
