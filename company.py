import warnings
warnings.filterwarnings('ignore')

from crewai import Agent, Task, Crew

import os
from utils import get_openai_api_key 

openai_api_key = get_openai_api_key()
os.environ["OPENAI_MODEL_NAME"] = 'gpt-4-turbo'

from crewai_tools import SerperDevTool, \
                         ScrapeWebsiteTool, \
                         WebsiteSearchTool, \
                         FileReadTool, \
                         PDFSearchTool

search_tool = SerperDevTool() # this scapes google
scrape_tool = ScrapeWebsiteTool(website_url="https://www.northropgrumman.com/", query="Northrop Grumman")  # if no paramters, this scrapes any website found by serper

''' use sepecific words to search like deep understanding, derives, refines, evaluates, 
scrutinize, strategize, effeciency, best execution, optimal, detailed, comprehensive,'''

# Agent 1: Looks up the company information
company_lookup = Agent(

    role="Company Agent",
    goal="""Find information about the company {company_name} to help the user understand the company's mission, values, and culture, especially identifying
          their most recent projects relating to {role} roles.""",
    backstory="You are working on finding information about the company {company_name} to help the user understand the company's mission, values, and culture.",
    allow_delegation=False,
    tools=[search_tool, scrape_tool],
    verbose=True
)

# Agent 2: Looks at the job description, espcially the where it is located and the job title itself
scrape_too = ScrapeWebsiteTool(website_url="https://www.northropgrumman.com/jobs/engineering/software/united-states-of-america/oklahoma/oklahoma-city/r10183494/engineer-software-simulation-okc-oklahoma")  # Replace with actual initialization

from pydantic import BaseModel
from typing import List

# Define a Pydantic model for venue details 
# (demonstrating Output as Pydantic)

# try making a json file where it has company name, location, job title, date, spots available, desired skills listed in the description, and pay
# try out human_input = True
# try out async_execution, can make it work with async tasks
class JobDescription(BaseModel):
    job_title: str
    location: str
    date: str
    desired_skills: List[str]

job_description = Agent(
    role="Job Agent",
    goal="Analyze the job description for the role of {job_title} at {company_name} located at {location} to understand the desired skills and preferred skills.",
    backstory="You are working on analyzing the job description for the role of {job_title} at {company_name} to understand the desired skills and preferred skills.",
    allow_delegation=False,
    tools=[scrape_too],
    verbose=True
)

# Agent 3: Analyzes the job description and company information
overall_description = Agent(
    role="Overall Description",
    goal="Analyze the job description for the role of {job_title} at {company_name} using information given bt the Company Agent and Job Agent.",
    backstory="You are working on analyzing the job description for the role of {job_title} at {company_name} using information from the Company Agent and Job Agent.",
    allow_delegation=False,
    verbose=True
)

company_task = Task(
    description="Look up the company information for {company_name} for {role} roles",
    expected_output="All the details of a company including their mission, values, culture, and recent projects related to {role} roles.",
    human_input = True,
    agent=company_lookup
)

job_task = Task(
    description="Gather data from the job description for the role of {job_title} at {company_name} located at {location}",
    expected_output="All the details of a specifically gathering data on the job title, date posted, spots available, and the desired skills.",
    human_input=True,

    output_file="job details_details.json",  
      # Outputs the venue details as a JSON file
    agent=job_description
)

overall_task = Task(
    description="Analyze the job description for the role of {job_title} at {company_name} using information given bt the Company Agent and Job Agent.",
    expected_output="A detailed analysis of the job description for the role of {job_title} at {company_name} using information from the Company Agent and Job Agent.",
    human_input=True,
    agent=overall_description
)

input_data = {
    "company_name": "Northrop Grumman",
    "role": "Software Engineer",
    "job_title": "Engineer Software Simulation",
    "location": "Oklahoma City"
}

crew = Crew(
    agents=[company_lookup, job_description, overall_description],
    tasks=[company_task, job_task, overall_task], 
    input_data=input_data
    )

result = crew.kickoff(inputs=input_data)