from pyexpat.errors import messages
from dotenv import find_dotenv, load_dotenv
import numpy as np
import streamlit as st
import os
import json
from datetime import date 
from mistralai import Mistral




load_dotenv(find_dotenv())

st.set_page_config(
    page_title="ChatBot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


class MistralAPI:

    DEFAULT_TEMPERATURE = 0.2
    DEFAULT_MAX_TOKENS = 1200
    DEFAULT_TOP_P = 0.9

    def __init__(self, model: str) -> None:
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError(
                "No MISTRAL_API_KEY as environment variable, please set it!"
            )
        self.client = Mistral(api_key=api_key)
        self.model = model

    def chat_json(
        self,
        system_prompt: str, 
        user_query: str,
        history_messages: list | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
    ) -> dict:
        """ Send back a dict Json of the offer asked by the user (or raise an exception if not possible)"""
        
        if history_messages is None:
            history_messages = []


        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if top_p is None:
            top_p = self.DEFAULT_TOP_P


        
        messages = [{"role": "system", "content": system_prompt}]

        for m in history_messages:
            role = m.get("role")
            content = m.get("content")
            if role in ("user", "assistant") and isinstance(content, str):
                messages.append({"role": role, "content": content})

        
        messages.append({"role": "user", "content": user_query}) # Dernière question 

        chat_response = self.client.chat.complete(
            model=self.model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,  # proba cumulative
            max_tokens=max_tokens,  # calibre la taille de la réponse généré
            response_format={"type": "json_object"},
        )

        content = chat_response.choices[0].message.content
        print("Mistral Raw Response:", content)

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            # On remonte le contenu brut pour debug
            raise ValueError(f"JSON invalide: {e}\n\nRAW:\n{content}")

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
        value="""Tu es un agent conversationnel. Tu vas recevoir la description d'une offre et tu devras analyser l'offre pour la reformater dans un JSON structuré.""",
        height=120,
    )


if "messages" not in st.session_state:
    st.session_state.messages = []

# On affiche les messages de l'utilisateur et de l'IA entre chaque message
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

SCHEMA_TEXT = """{
  "scraped_at": "YYYY-MM-DD",

  "title": "Intitulé du poste",
  "description": "Texte complet de l'offre",

  "company_name": "Nom de l'entreprise",
  "company_description": "Optionnel | none",

  "nb_annees_experience": 0,
  "experience_required": "Optionnel | none",

  "ville": "Ville | UNKNOWN",
  "code_postal": "00 | none",

  "type_contrat": "CDI | CDD | Stage | Alternance | Freelance | ... | UNKNOWN",
  "duree_contrat_mois": 0,
  "start_date": "YYYY-MM-DD | none",
  "is_teletravail": "true|false",

  "publication_date": "YYYY-MM-DD | none",

  "salaire": "texte libre | none",
  "hard_skills": "Python;SQL;Docker | none",
  "soft_skills": "Communication;Autonomie | none",
  "langages": "Français;Anglais | none",

  "education_level": "Bac+5 | ... | UNKNOWN",
  "job_function": "Data / IA | ... | UNKNOWN",
  "job_grade": "Junior | Confirmé | Senior | UNKNOWN"
}"""

today = date.today().isoformat()

SYSTEM_PROMPT = f"""
{role_prompt}
Tu DOIS répondre UNIQUEMENT avec un JSON valide (pas de markdown, pas de texte autour).
Respecte EXACTEMENT les clés du schéma ci-dessous (ne rajoute pas de clés).

Règles:
- Si une info est absente: mets "none" (string) ou 0 pour les champs numériques, ou "UNKNOWN" pour ville/contrat/niveau quand pertinent.
- Les listes doivent être une string séparée par ';' (ex: "Python;SQL;Docker").
- is_teletravail doit être un booléen strict (true/false).
- scraped_at doit être "{today}" (date du scraping aujourd'hui).

Schéma à respecter:
{SCHEMA_TEXT}
""".strip()

# Si présence d'un input par l'utilisateur,
if query := st.chat_input(""):
    if query.strip():
        # On affiche le message de l'utilisateur
        with st.chat_message("user"):
            st.markdown(query)
        # On ajoute le message de l'utilisateur dans l'historique de la conversation
        st.session_state.messages.append({"role": "user", "content": query})
        # On récupère la réponse du chatbot à la question de l'utilisateur
        try : 
            data = llm.chat_json(
                system_prompt=SYSTEM_PROMPT,
                history_messages=st.session_state.messages[:-1],
                user_query=query,
                temperature=llm.DEFAULT_TEMPERATURE,
                max_tokens=llm.DEFAULT_MAX_TOKENS,
                top_p=llm.DEFAULT_TOP_P,
            )

            pretty = json.dumps(data, ensure_ascii=False, indent=2)
            with st.chat_message("assistant"):
               st.code(pretty, language="json")
               st.download_button(
                    label="Télécharger le JSON",
                    data=pretty,
                    file_name="offre_structuree.json",
                    mime="application/json",
            )

            st.session_state.messages.append({"role": "assistant", "content": pretty})

        except Exception as e:
            with st.chat_message("assistant"):
                st.error(f"Erreur lors de la génération de la réponse: {e}")

if st.button("Clear Conversation", type="primary"):
    st.session_state.messages = []
    st.rerun()

