import os
import certifi
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, Runner, trace, function_tool, OpenAIChatCompletionsModel, input_guardrail, GuardrailFunctionOutput
from typing import Dict
from pydantic import BaseModel
import sendgrid
from sendgrid.helpers.mail import Mail, Email, To, Content
import asyncio

# SSL cert fix
os.environ['SSL_CERT_FILE'] = certifi.where()

# Load environment variables
load_dotenv(override=True)

# --------------------------
# Agent: Email writer
# --------------------------
prompt_agent_writer = """
You are a support course agent working for Thara_courses, 
a company that provides private tutoring sessions for students. 
Your role is to send professional and polite emails to clients who have requested tutoring services. 
In your emails, you offer the available time slots for the requested subject, 
clearly specifying the date and time for each slot so that the client can choose the one that suits them best.
Always keep your emails courteous, helpful, and clear.
"""

agent_writer = Agent(name="agent_writer", instructions=prompt_agent_writer, model="gpt-4o-mini")
tool1 = agent_writer.as_tool(tool_name="agent_writer", tool_description="write cold tutoring services Emails")

# --------------------------
# Function to send email
# --------------------------
@function_tool
def send_html_email(subject: str, html_body: str) -> Dict[str, str]:
    sg = sendgrid.SendGridAPIClient(api_key=os.environ.get('SENDGRID_API_KEY'))
    from_email = Email("aazighoul@gmail.com")  # Change to your verified sender
    to_email = To("aazighoul@gmail.com")  # Change to your recipient
    content = Content("text/html", html_body)
    mail = Mail(from_email, to_email, subject, content).get()
    sg.client.mail.send.post(request_body=mail)
    return {"status": "success"}

# --------------------------
# Agents for subject and HTML
# --------------------------
subject_instructions = """
You can write a subject for an email offering tutoring sessions. 
You are given a message with available dates and times, 
and you need to write a subject that is clear, polite, and likely to get the client to open the email and choose a slot.
"""
agent_subject = Agent(name="agent_subject", instructions=subject_instructions, model="gpt-4o-mini")
subject_tool = agent_subject.as_tool(tool_name="subject_writer", tool_description="write subjects for cold tutoring service emails")

html_instructions = """
You can convert a text email body offering tutoring sessions to an HTML email body. 
You are given a text email body which may have some basic formatting, 
and you need to convert it into an HTML email with a simple, clear, and professional layout 
that highlights the available dates and times for the client to choose from.
"""
agent_html = Agent(name="agent_html", instructions=html_instructions, model="gpt-4o-mini")
html_tool = agent_html.as_tool(tool_name="html_converter", tool_description="Convert a text email body to an HTML email body")

email_tools = [subject_tool, html_tool, send_html_email]

# --------------------------
# Email sender agent
# --------------------------
instructions = """
You are an email formatter and sender. You receive the body of an email to be sent. 
You first use the subject_writer tool to write a subject for the email, 
then use the html_converter tool to convert the body to HTML. 
Finally, you use the send_html_email tool to send the email with the subject and HTML body.
"""
agent_sender = Agent(
    name="agent_sender",
    instructions=instructions,
    tools=email_tools,
    model="gpt-4o-mini", 
    handoff_description="convert emails to html and send them"
)

# --------------------------
# Guardrail: Name check
# --------------------------
class NameCheckOutput(BaseModel):
    is_name_in_message: bool
    name: str

guardrail_agent = Agent( 
    name="Name check",
    instructions="Check if the user is including someone's personal name in what they want you to do.",
    output_type=NameCheckOutput,
    model="gpt-4o-mini"
)

@input_guardrail
async def guardrail_against_name(ctx, agent, message):
    result = await Runner.run(guardrail_agent, message, context=ctx.context)
    is_name_in_message = result.final_output.is_name_in_message
    return GuardrailFunctionOutput(output_info={"found_name": result.final_output}, tripwire_triggered=is_name_in_message)

# --------------------------
# Tutoring manager agent
# --------------------------
tutoring_manager_instructions = """
You are a Tutoring Manager at Thara_courses. Your goal is to find the best email offering available tutoring sessions 
using the writing_tool.

Follow these steps carefully:
1. Generate Draft: Use the writing_tool to generate an email draft offering time slots. 
   Do not proceed until the draft is ready.

2. Evaluate and Select: Review the draft and ensure it is clear, polite, and most likely to get the client to select a time slot. 
   You can regenerate the draft if you are not satisfied with the result.

3. Handoffs for Sending: Pass ONLY the final email draft to the 'agent_sender' agent. 
   The agent_sender will take care of formatting and sending.

Crucial Rules:
- You must use the writing_tool to generate the draft — do not write it yourself.
- You must hand off exactly ONE email to the agent_sender — never more than one.
"""

Email_Manager = Agent(
   name="Email_Manager",
   instructions=tutoring_manager_instructions,
   tools=[tool1],
   model="gpt-4o-mini",
   handoffs=[agent_sender],
   input_guardrails=[guardrail_against_name]
)

# --------------------------
# Run example
# --------------------------
async def main():
    message = "write a cold email"
    with trace("Protected_Thara_Courses :"):
        result = await Runner.run(Email_Manager, message)
        print(result)

if __name__ == "__main__":
    asyncio.run(main())
