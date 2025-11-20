import os
import streamlit as st
from dotenv import load_dotenv
from crew_setup import run_appointment_crew

load_dotenv()


def ensure_api_key():
    """Make sure OPENAI_API_KEY is set."""
    if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
        st.sidebar.warning("No OPENAI_API_KEY found in environment.")
        key = st.sidebar.text_input(
            "Enter your OpenAI API key",
            type="password",
            help="This is only used locally by the app.",
        )
        if key:
            os.environ["OPENAI_API_KEY"] = key

    if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
        st.stop()


def main():
    st.set_page_config(page_title="Doctor Appointment Assistant", page_icon="🩺")
    st.title("🩺 Doctor Appointment Assistant (CrewAI + Streamlit)")

    st.write(
        "Chat with an AI clinic assistant to **check or book doctor appointments**.\n\n"
        "Note: This is a demo with **mock appointment slots**. "
        "For medical emergencies, always call your local emergency number."
    )

    ensure_api_key()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role, content in st.session_state.chat_history:
        if role == "user":
            st.chat_message("user").markdown(content)
        else:
            st.chat_message("assistant").markdown(content)

    user_input = st.chat_input("How can I help with your appointment today?")
    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        st.chat_message("user").markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = run_appointment_crew(user_input)
                except Exception as e:
                    answer = f"Error while calling the appointment agent: {e}"

                st.markdown(answer)
                st.session_state.chat_history.append(("assistant", answer))


if __name__ == "__main__":
    main()
