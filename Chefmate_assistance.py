"""
ChefMate AI - A specialized culinary assistant built with Streamlit + LangChain.

Run with:
    python -m streamlit run app.py
"""

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# =========================================================
# CONFIGURATION
# =========================================================

MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.7

PAGE_TITLE = "🍳 ChefMate AI"
PAGE_SUBTITLE = "Turn your ingredients into something delicious!"

EXAMPLE_PROMPTS = [
    "I have chicken, potatoes, onion and yogurt. What can I make?",
    "I have Indomie chicken noodles, an egg and spring onions.",
    "I have grape leaves, rice and minced beef.",
    "I have Buldak noodles, cheese and egg. I don't eat spicy food.",
    "What can I substitute for buttermilk?",
    "What is dolma?",
]

# =========================================================
# SYSTEM PROMPT
# This is the heart of ChefMate's behavior: food-only scope,
# contextual entity understanding (no giant hard-coded brand
# lists), whole-inventory reasoning, preference memory, and
# food safety awareness.
# =========================================================

SYSTEM_PROMPT = """You are ChefMate AI, a knowledgeable, friendly international home chef
and culinary assistant. You help users with recipes, meal ideas, ingredients, food
products, cooking techniques, food safety, and general culinary knowledge.

=== SCOPE: FOOD ONLY ===
You ONLY answer questions related to food, cooking, ingredients, recipes, food products,
food brands, nutrition basics, and culinary culture. If a user asks about something
unrelated to food (programming, politics, general trivia, etc.), politely refuse with a
short message like:
"Sorry! 🍳 I'm ChefMate AI, so I can only help with food, cooking, ingredients, recipes,
and other culinary questions. Please ask me something food-related!"
Do not answer the unrelated part of the question, even partially. Use judgment: a message
that merely contains a food word but is really about something else (e.g. "write a Python
program to calculate calories") is NOT a food question and should be refused. A genuine
food/culinary question phrased in an unusual way (e.g. "what's the history of Italian
food?") should always be answered.

=== UNDERSTANDING FOOD ENTITIES CONTEXTUALLY ===
You must recognize and reason about ALL kinds of food-related terms using your own
culinary knowledge and the surrounding context - not a fixed list. This includes:
- Ingredients (chicken, onion, tomato paste)
- Food brands and packaged products (Indomie, Buldak, Maggi, Nutella, Heinz, etc.)
- Prepared dishes (samosa, biryani, ramen, dolma)
- Wrappers/components of dishes (samosa sheets, dough, phyllo)
- Cuisines and regional context (Pakistani, Korean, Lebanese, Mexican, etc.)
- Cooking methods (frying, braising, steaming)
- Condiments, sauces, spices, snacks, beverages, desserts, frozen foods, and leftovers

Never dismiss or ignore a term simply because it isn't a plain generic ingredient. A
capitalized or unfamiliar word in a food context (e.g. "Indomie", "Buldak", "Maggi") is
very likely a food brand or product - reason about what it probably is from context, the
same way a well-traveled home cook would. Do not assume a brand always maps to one fixed
product; interpret it from what the user says (e.g. "Indomie chicken noodles" = an instant
noodle product; "Maggi" alone could be noodles, seasoning cubes, or sauce depending on
context).

Handle reasonable misspellings, transliterations, and spelling variants of food terms when
context strongly suggests a specific product or dish (e.g. "Budak noodles" -> likely
"Buldak noodles"). If you are genuinely uncertain what a term refers to, do NOT confidently
invent an interpretation - ask a short, friendly clarifying question instead, e.g. "Do you
mean Buldak noodles? If so, I can help you build a recipe with them."

Never ignore unfamiliar food-related proper nouns outright - always try to reason about
what category they likely belong to (brand, dish, ingredient, cuisine, etc.) before
responding.

=== USE THE ENTIRE INGREDIENT/PRODUCT LIST ===
When a user lists multiple ingredients or products, first mentally review the ENTIRE list
before deciding on a recipe. Do not just grab the first recognizable item (e.g. don't
jump straight to "samosas" just because "samosa sheets" appears, while ignoring "Indomie
noodles" elsewhere in the same list). Weigh all the items together and choose the most
practical, well-fitting recipe(s). If more than one recipe genuinely fits well, offer 2-3
options and briefly explain why each is a good fit, rather than arbitrarily picking one.

=== RECIPE REALISM ===
Do not invent ingredients the user never mentioned, except for common pantry staples
(salt, black pepper, water, cooking oil) which you may reasonably assume are available
unless the user says otherwise. If something important is missing, either suggest a
substitution or ask the user - never pretend they have something they said they don't.

=== USER PREFERENCES ===
Track and respect preferences mentioned earlier in the conversation: cuisine, spice level,
dietary restrictions (vegetarian, vegan, halal, gluten-free, dairy-free, etc.), allergies,
available equipment, time constraints, servings, difficulty, and budget. If a preference
conflicts with a request (e.g. user dislikes spicy food but has an inherently spicy
product like Buldak), acknowledge the conflict, explain that the product itself is
typically associated with a certain flavor profile, and suggest a reasonable adaptation
(smaller amount of sauce, mixing with dairy/egg to mellow heat, etc.) where appropriate.

=== FOOD SAFETY ===
Take extra care with raw meat, poultry, seafood, eggs, food storage, cross-contamination,
allergens, and spoiled food. Never give false certainty about safety. For serious medical
dietary conditions or severe allergies, recommend that the user consult a doctor or
qualified professional in addition to any general guidance you provide.

=== ANSWERING NON-RECIPE FOOD QUESTIONS ===
Not every food question needs a full recipe. Answer conceptual/informational culinary
questions directly and concisely (e.g. "what is tahini?", "what can I substitute for
buttermilk?", "how do I store cooked rice?", "what is dolma?").

=== RECIPE OUTPUT FORMAT ===
When you do provide a recipe, format it clearly using relevant sections from the
following, only including sections that are actually useful (don't force every section
every time):

🍽️ Recipe Name
📝 Short description
🧂 Ingredients
👨‍🍳 Step-by-step instructions
⏱️ Preparation time
🔥 Cooking time
👥 Servings
💡 Cooking tips
🔄 Substitutions
🍴 Serving suggestions

Keep answers as long as they need to be to be useful, but do not pad simple questions with
unnecessary length.

=== TONE ===
Be warm, encouraging, and knowledgeable - like a well-traveled home chef friend, not a
generic corporate assistant.
"""

# =========================================================
# LIGHTWEIGHT, NON-AGGRESSIVE FOOD-CONTEXT PRE-CHECK
# This is deliberately minimal and only used to avoid making
# an API call for completely empty input. All real food-vs-
# not-food classification is left to the LLM via the system
# prompt, per the design principle: no giant keyword lists.
# =========================================================


def is_message_empty(message: str) -> bool:
    """Check whether the user submitted an empty/whitespace-only message."""
    return not message or not message.strip()


# =========================================================
# STREAMLIT PAGE SETUP
# =========================================================

st.set_page_config(page_title="ChefMate AI", page_icon="🍳", layout="centered")

# ---------------------------------------------------------
# SESSION STATE INITIALIZATION
# ---------------------------------------------------------
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "connected" not in st.session_state:
    st.session_state.connected = False
if "messages" not in st.session_state:
    # Stores the visible chat history (list of dicts: role, content)
    st.session_state.messages = []
if "llm" not in st.session_state:
    st.session_state.llm = None


def connect_to_openai(api_key: str) -> bool:
    """
    Attempt to initialize the LangChain ChatOpenAI client with the
    provided API key. Returns True on success, False otherwise.
    """
    if is_message_empty(api_key):
        st.error("⚠️ Please enter a valid OpenAI API key.")
        return False

    try:
        llm = ChatOpenAI(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            api_key=api_key,
        )
        # Make a tiny test call to validate the key works.
        llm.invoke([HumanMessage(content="Hello")])

        st.session_state.llm = llm
        st.session_state.api_key = api_key
        st.session_state.connected = True
        return True

    except Exception:
        # Never expose raw exception details to the user - they may
        # contain sensitive information (e.g. partial key fragments).
        st.error(
            "❌ Could not connect with that API key. Please check that it is "
            "correct, active, and has available quota, then try again."
        )
        st.session_state.connected = False
        st.session_state.llm = None
        return False


def disconnect():
    """Clear the API key and connection state (does not clear chat history)."""
    st.session_state.api_key = ""
    st.session_state.connected = False
    st.session_state.llm = None


def clear_conversation():
    """Clear the chat history only."""
    st.session_state.messages = []


def build_langchain_messages():
    """
    Convert the Streamlit chat history into a list of LangChain message
    objects, prefixed with the ChefMate system prompt.
    """
    lc_messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        else:
            lc_messages.append(AIMessage(content=msg["content"]))
    return lc_messages


def get_chefmate_response(user_input: str) -> str:
    """
    Send the full conversation (including the new user message) to the
    LLM and return the assistant's reply text. Handles errors gracefully.
    """
    try:
        lc_messages = build_langchain_messages()
        lc_messages.append(HumanMessage(content=user_input))
        response = st.session_state.llm.invoke(lc_messages)
        return response.content

    except Exception as exc:
        error_text = str(exc).lower()
        if "rate limit" in error_text or "quota" in error_text:
            return (
                "⚠️ It looks like your OpenAI account has hit a rate limit or "
                "quota issue. Please check your OpenAI usage/billing and try "
                "again shortly."
            )
        if "authentic" in error_text or "api key" in error_text or "401" in error_text:
            return (
                "⚠️ There seems to be an issue with your API key. Please "
                "disconnect and reconnect with a valid key."
            )
        if "connection" in error_text or "timeout" in error_text or "network" in error_text:
            return (
                "⚠️ I couldn't reach the OpenAI service due to a network issue. "
                "Please check your connection and try again."
            )
        return (
            "⚠️ Something went wrong while generating a response. Please try "
            "again in a moment."
        )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.header("🔑 API Connection")

    if not st.session_state.connected:
        api_key_input = st.text_input(
            "Enter your OpenAI API key",
            type="password",
            placeholder="sk-...",
            help="Your key is only kept for this session. It is never written to disk.",
        )
        if st.button("🔌 Connect", use_container_width=True):
            with st.spinner("Connecting..."):
                success = connect_to_openai(api_key_input)
            if success:
                st.success("✅ Connected!")
                st.rerun()
    else:
        st.success("✅ Connected to OpenAI")
        if st.button("🔐 Disconnect", use_container_width=True):
            disconnect()
            st.rerun()

    st.divider()

    st.header("🍴 Example Prompts")
    st.caption("Tap one to try it out (you must be connected first).")
    for prompt in EXAMPLE_PROMPTS:
        if st.button(prompt, use_container_width=True, key=f"example_{prompt}"):
            if not st.session_state.connected:
                st.warning("⚠️ Please connect your API key first.")
            else:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.spinner("👨‍🍳 ChefMate is thinking..."):
                    reply = get_chefmate_response(prompt)
                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.rerun()

    st.divider()

    st.header("🗑️ Conversation")
    if st.button("Clear conversation", use_container_width=True):
        clear_conversation()
        st.rerun()


# =========================================================
# MAIN AREA
# =========================================================

st.title(PAGE_TITLE)
st.caption(PAGE_SUBTITLE)

if not st.session_state.connected:
    st.info(
        "👋 Welcome to ChefMate AI! Enter your OpenAI API key in the sidebar to "
        "get started. I can help you turn your ingredients into recipes, "
        "identify food products and brands, explain cooking techniques, and "
        "answer culinary questions from around the world."
    )
else:
    if not st.session_state.messages:
        st.info(
            "👋 Hi, I'm ChefMate AI! Tell me what ingredients or food products "
            "you have, or ask me any food/cooking question, and I'll help you "
            "out."
        )

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("What's cooking? Ask me anything food-related...")

    if user_input:
        if is_message_empty(user_input):
            st.warning("⚠️ Please type a message.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("👨‍🍳 ChefMate is thinking..."):
                    reply = get_chefmate_response(user_input)
                st.markdown(reply)

            st.session_state.messages.append({"role": "assistant", "content": reply})
