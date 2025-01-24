import warnings
warnings.filterwarnings('ignore')

from crewai import Agent, Task, Crew

import os
from utils import get_openai_api_key 

openai_api_key = get_openai_api_key()
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'

from crewai_tools import SerperDevTool, \
                         ScrapeWebsiteTool, \
                         WebsiteSearchTool, \
                         FileReadTool, \
                         PDFSearchTool

planner = Agent(
    role="Star Planner",
    goal="Plan a factual and precise 'How-to' using the topic {topic}",
    backstory="You're working on planning a 'How-to' guide "
              "about the topic: {topic}."
              "You collect information that helps the"
              "audience learn the most optimal way to talk about their {major}"
              "for {level} level into their career as a {role}. "
              "You also consider the best practices using {skills}."
              "Your work is the basis for "
              "the Content Writer to write a 'How-to' talk for {topic} for the industry of {role}.",
    allow_delegation=False,
	verbose=True
)

response = PDFSearchTool(pdf="C:\\Users\\Daniel\\Desktop\\CrewAI_Practice\\Daniel_J_Rodriguez_Ponce_Resume.pdf", query=["Work Experience", "Project Experience", "Education", "Skills"])  # Replace with actual initialization

planner2 = Agent(
    role="Resume Planner",
    goal="Extract and summarize the key details from the resume into sections like Professional Experience, Education, Skills, Certifications, and Awards.",
    backstory="You are working on reading a given resume for a {level} level {role} role."
              "You collect information that helps the"
              "user learn the most optimal way to talk about their experience"
              "for {level} level into their career as a {role}. "
              "You also consider the best practices using the {topic}."
              "Your work is the basis for "
              "the Content Writer to write a guide on how to use the collected information from the resume for {topic}.",
    allow_delegation=False,
    tools = [response],
	verbose=True
)


writer = Agent(
    role = "star Writer",
    goal= "Using the candidate's resume, draft a professional document using {topic}. Highlight their relevant skills, experience, and achievements tailored to {level} level {role} role .",
    backstory="""You are tasked with writing a comprehensive 'how-to' guide on effectively discussing results found from the Resume Planner through the lens of the {topic} method.
      Your writing is guided by the principles and insights provided by the Star Planner, who offers an outline, key objectives, and relevant context about the topic.
      You are also given insights of a candidates resume from the Resume Planner""",
    allow_delegation= False,
	verbose=True
)

editor = Agent(
    role="Editor",
    goal="Edit a given 'how-to' guide to align with "
         "what is the best practice using {topic} for {level} level jobs for {role}. ",
    backstory="You are an editor who receives a 'how-to' guide "
              "from the Star Writer. "
              "Your goal is to review the guide "
              "to ensure that it follows best practices for {topic},"
              "provides balanced goals "
              "when providing the best ways to present yourself. "
              "This also avoids unnecessary goals like talking about growing up.",
    allow_delegation=False,
    verbose=True
)

plan = Task(
    description=(
        "1. Prioritize the best practices for {topic} based on {level} level {role} roles.\n"
        "2. Identify the the best interests for each principle of {topic}, considering "
            "their best practices and what those practices mean.\n"
        "3. Develop a detailed outline including "
            "an principles of {topic}, {level} level jobs for {role} roles coming straight from college.\n"
        "4. Include {topic} and {level} level {role} role keywords using relevant data and skills: {skills}."
    ),
    expected_output="A comprehensive document with an outline, keywords, and resources.",
    agent=planner,
)

plan2 = Task(
    description=(
        "1. Prioritize the highlights of the given resume from the user.\n"
        "2. Identify the the best interests for each principle of {topic}, considering "
            "their best practices and what those practices mean using those experiences from the resume.\n"
        "3. Develop a detailed summary of work experience, project experience, and skills that can relate "
            "an principles of {topic}, {level} level jobs for {role} roles coming straight from college.\n"
        "4. Include {topic} and {level} level {role} role keywords using relevant data and skills used in the resume."
    ),
    expected_output="A comprehensive document with an skills, keywords, experiences, and resources.",
    agent=planner2,
)

write = Task(
    description=(
        "1. Follow the main objectives and structure outlined by the Star Planner..\n"
        "2. Incorporate objective and impartial insights, backing them with information provided by the Star Planner and the Resume Planner.\n"
		"3. Organize your guide systematically, creating a clear and descriptive approach for each aspect of the {topic}."
        "4. Lastly, List some of the main topics found using the Resume Planner to help the read identify what the resume contained"
    ),
    expected_output="""Your ultimate goal is to teach individuals how to articulate their skills {skills} and educational background {major} in a way that is compelling, 
    organized, and relevant. Focus on creating practical, step-by-step instructions and actionable advice for each section of the guide.""",
    agent=writer,
)

edit = Task(
    description=("Proofread the given guide for "
                 "grammatical errors and "
                 "alignment with the the users {skills}, {major}, and experience."),
    expected_output="A well-written guide in a numbered format, ready for publication, each section should have a description on how to talk about each section of {topic}.",
    agent=editor
)

crew = Crew(
    agents=[planner, planner2, writer,editor],
    tasks=[plan, plan2, write, edit],
    verbose = 2
)

result = crew.kickoff(inputs={"topic": "STAR method", "major":"Computer Science", "level":"entry", "role":"Junior Developer", "skills":["Java", "Python", "SQL"]})

from IPython.display import Markdown
Markdown(result)