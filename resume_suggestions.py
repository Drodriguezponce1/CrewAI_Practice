import warnings
warnings.filterwarnings('ignore')

from crewai import Agent, Task, Crew

import os
from utils import get_openai_api_key, get_serper_api_key, get_gh_api_key

openai_api_key = get_openai_api_key()
gh_token = get_gh_api_key()
os.environ["OPENAI_MODEL_NAME"] = 'gpt-4-turbo'
os.environ["SERPERDEV_API_KEY"] = get_serper_api_key()

from crewai_tools import SerperDevTool, \
                         ScrapeWebsiteTool, \
                         GithubSearchTool, \
                         MDXSearchTool, \
                         PDFSearchTool
                         

# readng the internet tools
search_tool = SerperDevTool() # this scapes google
scrape_tool = ScrapeWebsiteTool(website_url="https://www.northropgrumman.com/", query="Northrop Grumman")  # if no paramters, this scrapes any website found by serper

# resume tools
resume = PDFSearchTool(pdf="C:\\Users\\Daniel\\Desktop\\CrewAI_Practice\\Daniel_J_Rodriguez_Ponce_Resume.pdf")  # Replace with actual initialization
semantic_search_resume = MDXSearchTool(mdx='./res.md')

# maybe look through my repos
repos = tool = GithubSearchTool(
	gh_token=gh_token,
	content_types=['code','repo', 'issue'] # Options: code, repo, pr, issue
)

resume_strategist = Agent(
    role="Resume Strategist for Engineers",
    goal="Find all the best ways to make a "
         "resume stand out in the job market.",
    tools = [scrape_tool, search_tool,
             resume, semantic_search_resume, repos],
    verbose=True,
    backstory=(
        "With a strategic mind and an eye for detail, you "
        "excel at refining resumes to highlight the most "
        "relevant skills and experiences, ensuring they "
        "resonate perfectly with the job's requirements."
    )
)

# Agent 4: Interview Preparer
interview_preparer = Agent(
    role="Engineering Interview Preparer",
    goal="Create technical interview questions and talking points "
         "based on the resume and job requirements",
    tools = [scrape_tool, search_tool,
             resume, semantic_search_resume],
    verbose=True,
    backstory=(
        "Your role is crucial in anticipating the dynamics of "
        "interviews. With your ability to formulate key questions "
        "and talking points, you prepare candidates for success, "
        "ensuring they can confidently address all aspects of the "
        "job they are applying for."
    )
)

star_preparer = Agent(
    role="Star Method Preparer",
    goal="Prepare STAR method responses for common "
         "interview questions based on the resume and job requirements",
    backstory=(
        "Your expertise lies in crafting compelling STAR method "
        "responses that effectively showcase a candidate's skills "
        "and experiences. By aligning these responses with the job "
        "requirements, you help candidates excel in interviews, "
        "demonstrating their value to potential employers."
    ),
    tools = [scrape_tool, search_tool,
             resume, semantic_search_resume],
    verbose=True