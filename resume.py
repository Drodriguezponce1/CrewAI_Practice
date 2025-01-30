import warnings
warnings.filterwarnings('ignore')

from crewai import Agent, Task, Crew

import os
from utils import get_openai_api_key, get_serper_api_key 

openai_api_key = get_openai_api_key()
os.environ["OPENAI_MODEL_NAME"] = 'gpt-4-turbo'
os.environ["SERPERDEV_API_KEY"] = get_serper_api_key()

from crewai_tools import PDFSearchTool, FileReadTool

resume = PDFSearchTool(pdf="C:\\Users\\Daniel\\Downloads\\Daniel_J._Rodriguez-Ponce.pdf")  # Replace with actual initialization

resume_formatter = Agent(

    role="Resume Formatter",
    goal="Format a resume pdf fie into a markdown file",
    backstory="You are a resume formatter who is tasked with formatting a resume into a markdown file from a pdf file. "
              "The format of this resume includes the header section (name and contact links), Education, Work Experience, Project Experience, and Skills. "
              "Your goal is to format this resume into a markdown file using the various sections, subsections and all their bullet points."
              "Also please provide a brief summary of the resume at the end of the markdown file"
              ", including the candidate's skills, experience, and qualifications. You should also highlight any key "
              "achievements or accomplishments mentioned in the resume. For the format use the sections and subsections found by the Resume Reader.",
    allow_delegation= True,
    memory = True,
    tools=[resume],
    verbose=True
)

resume_writer = Task(
    description="Generate a Markdown file with a structured format that includes sections and subsections from the resume",
    expected_output="Format the resume into a markdown file with the different sections. subsections and bullet points found in the resume "
      "and provide a brief summary of the resume at the end of the markdown file",
    output_file = "res2.md",
    human_input=True,
    agent=resume_formatter

)


crew = Crew(
    agents = [resume_formatter],
    tasks = [resume_writer],
    verbose = 2
)

kick = crew.kickoff()