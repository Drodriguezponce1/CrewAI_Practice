import warnings
warnings.filterwarnings('ignore')

from crewai import Agent, Task, Crew

import os
from utils import get_openai_api_key 

openai_api_key = get_openai_api_key()
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'

from crewai_tools import SerperDevTool, \
                         ScrapeWebsiteTool, \
                         WebsiteSearchTool

planner = Agent(
    role="Leveling Planner",
    goal="Plan a factual and precise ironman guide for the video game {topic}",
    backstory="You're working on planning a precise leveling guide "
              "about the topic: {topic}."
              "You collect information that helps the"
              "audience learn the most optimal route for their hardcore ironman"
              "and make informed decisions. "
              "You also consider the best weapon and armor to go for once the leveling goals are met."
              "Your work is the basis for "
              "the Content Writer to write a leveling guide on this topic.",
    allow_delegation=False,
	verbose=True
)

writer = Agent(
    role = "Leveling Writer",
    goal= "Write an insiteful and precise leveling guide for a level 1 ironman in the video game {topic}",
    backstory="You're working on a writing "
              "a new ironaman leveling guide for the video game: {topic}. "
              "You base your writing on the work of "
              "the Leveling Planner, who provides an outline "
              "and relevant context about the topic. "
              "You follow the main objectives and "
              "direction of the outline, "
              "as provide by the Leveling Planner. "
              "You also provide objective and impartial insights "
              "and back them up with information "
              "provide by the Leveling Planner. "
              "You order this list and create descriptive "
              "goals on why the current step is the best.",
    allow_delegation=False,
	verbose=True
)

editor = Agent(
    role="Editor",
    goal="Edit a given leveling guide to align with "
         "the what players usually go with for their leveling guides. ",
    backstory="You are an editor who receives a leveling guide "
              "from the Leveling Writer. "
              "Your goal is to review the leveling guide "
              "to ensure that it follows best practices for video game guides,"
              "provides balanced goals "
              "when providing the best quests, weapons, and armor. "
              "and also avoids unnecessary goals like going for a max skill in the early stages of the game.",
    allow_delegation=False,
    verbose=True
)

plan = Task(
    description=(
        "1. Prioritize the , best quests early on, key items, "
            "and noteworthy ironman upgrades on {topic}.\n"
        "2. Identify the the best interests for each skill, considering "
            "their best levels and what those levels offer.\n"
        "3. Develop a detailed leveling outline including "
            "an early quests to do, key upgrades, and a call to action.\n"
        "4. Include OSRS keywords and relevant data or sources."
    ),
    expected_output="A comprehensive leveling plan document "
        "with an outline, OSRS keywords, and resources.",
    agent=planner,
)

write = Task(
    description=(
        "1. Use the leveling plan to craft a compelling "
            "guide post on {topic}.\n"
        "2. Incorporate OSRS keywords naturally.\n"
		"3. Sections/Subtitles are properly named "
            "in an engaging manner.\n"
        "4. Ensure the post is structured with an "
            "engaging details, insightful body, "
            "and a summarizing why this is the best technique.\n"
        "5. Proofread for grammatical errors and "
            "alignment.\n"
    ),
    expected_output="A well-written leveling guide "
        "in a numbered format, ready for publication, "
        "each section should have a description on why this the best step.",
    agent=writer,
)

edit = Task(
    description=("Proofread the given leveling guide for "
                 "grammatical errors and "
                 "alignment with the brand's voice."),
    expected_output="A well-written leveling guide "
                    "in a numbered format, ready for publication, "
                    "each section should have a description on why this the best step.",
    agent=editor
)

crew = Crew(
    agents=[planner, writer,editor],
    tasks=[plan, write, edit],
    verbose = 2
)

result = crew.kickoff(inputs={"topic": "Old School Runescape"})

from IPython.display import Markdown
Markdown(result)