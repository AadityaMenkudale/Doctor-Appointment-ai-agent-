import os
from crewai import Agent, Task, Crew, Process, LLM
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()  # Load environment variables from .env if present


def get_llm():
    """Configure the underlying LLM for CrewAI using Gemini 2.5 Flash via CrewAI's LLM wrapper."""
    api_key = os.getenv("GOOGLE_API_KEY", "")

    return LLM(
        model="gemini/gemini-2.5-flash",
        temperature=0.3,
        api_key=api_key,
    )



def build_doctor_agent():
    """Agent that behaves like a medical receptionist / appointment assistant."""
    llm = get_llm()

    doctor_agent = Agent(
        role="Clinic Appointment Assistant",
        goal=(
            "Help patients schedule, reschedule, or cancel doctor appointments, "
            "explain available time slots and basic visit information. "
            "NEVER provide medical diagnosis or emergency advice. "
            "For emergencies, always tell the user to call emergency services."
        ),
        backstory=(
            "You work at a multi-specialty clinic. You know how to: "
            "1) Ask for patient's name, preferred date/time, and reason (brief). "
            "2) Suggest available slots. "
            "3) Confirm appointments in a friendly, concise way."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    return doctor_agent


MOCK_SLOTS = {
    "Dr. Sharma (General Physician)": [
        "Today 4:00 PM",
        "Today 6:00 PM",
        "Tomorrow 10:00 AM",
    ],
    "Dr. Mehta (Dermatologist)": [
        "Tomorrow 3:00 PM",
        "Tomorrow 5:00 PM",
    ],
    "Dr. Rao (Pediatrician)": [
        "Today 5:30 PM",
        "Day after tomorrow 11:00 AM",
    ],
}


def build_appointment_task(user_message: str, context: Dict[str, Any] | None = None):
    """Task that uses the agent to respond to the user about doctor appointments."""
    context = context or {}

    slots_text = "\n".join(
        f"- {doctor}: {', '.join(times)}" for doctor, times in MOCK_SLOTS.items()
    )

    # Multiline prompt using a single f-string with proper triple quotes
    task_description = f"""You are helping a user with doctor appointments.

User message:
{user_message}

Available slots in the clinic (mock data):
{slots_text}

Instructions:
- Greet the user politely.
- Ask for missing booking details (name, doctor/speciality, preferred date/time) if needed.
- Propose suitable available slots from the list above.
- Confirm the chosen slot clearly.
- DO NOT invent extra slots outside the list above.
- DO NOT provide any medical diagnosis or emergency instructions.
- If user mentions emergency or severe symptoms, say:
  "Please contact your nearest emergency service or hospital immediately."
- Reply in short, clear paragraphs. Avoid long walls of text.
"""

    task = Task(
        description=task_description,
        agent=build_doctor_agent(),
        expected_output=(
            "A concise conversation-style response helping the user book or manage an appointment."
        ),
    )

    return task


def run_appointment_crew(user_message: str) -> str:
    """Orchestrate crew run for a single user message."""
    agent = build_doctor_agent()
    task = build_appointment_task(user_message)

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()
    return str(result)
