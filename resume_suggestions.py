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
                         PDFSearchTool, \
                         JSONSearchTool, \
                         FileReadTool
                         

# readng the internet tools
search_tool = SerperDevTool(n_results=5) # this scapes google
scrape_tool = ScrapeWebsiteTool()  # if no paramters, this scrapes any website found by serper

# resume tools
resume = PDFSearchTool(pdf="C:\\Users\\Daniel\\Desktop\\CrewAI_Practice\\Daniel_J_Rodriguez_Ponce_Resume.pdf")  # Replace with actual initialization
resume_mdx = FileReadTool(file_path="C:\\Users\\Daniel\\Desktop\\CrewAI_Practice\\res.md")

# company tools
company = FileReadTool(file_path="C:\\Users\\Daniel\\Desktop\\CrewAI_Practice\\company_details.md")

# Job Description tools
job_description = JSONSearchTool(json="C:\\Users\\Daniel\\Desktop\\CrewAI_Practice\\job_details.json")


# maybe look through my repos
repos = tool = GithubSearchTool(
	gh_token=gh_token,
	content_types=['code','repo', 'issue'] # Options: code, repo, pr, issue
)

resume_strategist = Agent(
    role="Resume Strategist for Engineers",
    goal="Find all the best ways to make a resume stand out in the {level} level {position} positions.",
    tools = [scrape_tool, search_tool],
    verbose=True,
    backstory=(
        "With a strategic mind and an eye for detail, you "
        "excel at refining resumes to highlight the most "
        "relevant skills and experiences, ensuring they "
        "resonate perfectly with the job's requirements."
    )
)

star_preparer = Agent(
    role="Star Method Preparer",
    goal="Prepare STAR method responses for common interview questions based on the resume and job requirements",
    backstory=(
        "Your expertise lies in crafting compelling STAR method "
        "responses that effectively showcase a candidate's skills "
        "and experiences. By aligning these responses with the job "
        "requirements, you help candidates excel in interviews, "
        "demonstrating their value to potential employers."
    ),
    tools = [scrape_tool, search_tool,
             resume, resume_mdx, company, job_description],
    verbose=True
)

resume_rater = Agent(
    role="Resume Rater",
    goal="Rate and analyze the resume based on the job description, company details, and the resume itself to help them stand out",
    tools = [resume, resume_mdx, company, job_description],
    verbose=True,
    backstory=(
        "As a resume rater, you have a keen eye for detail and a "
        "deep understanding of what employers look for in candidates. "
        "You are able to craft comprehensive personal and professional profiles, laying the "
        "groundwork for personalized resume enhancements."
    )
)

interview_preparer = Agent(
    role="Engineering Interview Preparer",
    goal="Create technical interview questions and talking points based on the resume, job requirements, company details, job details, and STAR method responses",
    tools = [scrape_tool, search_tool,
             resume, resume_mdx, company, job_description],
    verbose=True,
    backstory=(
        "Your role is crucial in anticipating the dynamics of "
        "interviews. With your ability to formulate key questions "
        "and talking points, you prepare candidates for success, "
        "ensuring they can confidently address all aspects of the "
        "job they are applying for."
    )
)


#TASKS Resume strategists, star prep, resume rater, interview prep
resume_task = Task(
    description="Compile a detailed personal and professional profile "
        "using the Resume, Github ({github}) URL, and the job description ({job}). Utilize tools to extract and "
        "synthesize information from these sources. Consider adding more projects that seem like a good fit from the GitHub repositories at {github}.",
    expected_output="A detailed analysis of the resume with suggestions for improvement in a markdown format.",
    agent=resume_strategist,
    asynchronous=True,
)

star_task = Task(
    description="Prepare STAR method responses for common interview questions "
        "based on the resume and job requirements.",
    expected_output="A list of STAR method responses tailored to the job requirements.",
    agent=star_preparer,
    context= [resume_task],
    output_file="star_method.md",
    asynchronous=True,
)

resume_rate_task = Task(
    description="Using the starting resume, rate the resume out of 100 based how the resume is compared to the desired skills and qualifications from the company job description."
                "Next, Using the profile and job requirements obtained from "
                "previous tasks, tailor the resume to highlight the most "
                "relevant qualifications. Employ tools to adjust and enhance the "
                "resume content. Make sure this is the best resume even but "
                "don't make up any information. Update every section, "
                "inlcuding the work experience, project experience, and skills."
                "All to better reflect the candidates "
                "abilities and how it matches the job posting. Also consider making better bullet points for each section.",
    expected_output="A detailed rating of the resume with suggestions for improvement.",
    agent=resume_rater,
    context= [resume_task],
    output_file="resume_improvements.md",
    asynchronous=True,
)

interview_preparer_task = Task(
    description=(
        "Create a set of potential interview questions and talking "
        "points based on the tailored resume and job requirements. "
        "Utilize tools to generate relevant questions and discussion "
        "points. Make sure to use these question and talking points to "
        "help the candidate highlight the main points of the resume "
        "and how it matches the job posting."
    ),
    expected_output=(
        "A document containing key questions and talking points "
        "that the candidate should prepare for the initial interview."
    ),
    agent=interview_preparer,
    context= [resume_task, star_task, resume_rate_task],
    output_file="interview_questions.md",
    asynchronous=True,
)

input_data = {
    "level": "entry",
    "position": "Software Engineer",
    "github": "https://github.com/Drodriguezponce1?tab=repositories",
    "job": "https://www.northropgrumman.com/jobs/engineering/software/united-states-of-america/oklahoma/oklahoma-city/r10183494/engineer-software-simulation-okc-oklahoma"
}

crew = Crew(
    agents=[resume_strategist, star_preparer, resume_rater, interview_preparer],
    tasks=[resume_task, star_task, resume_rate_task, interview_preparer_task],
    input_data=input_data
)

result = crew.kickoff(inputs=input_data)