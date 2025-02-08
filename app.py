# %%writefile app.py

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from crewai import Agent, Task, Crew, LLM
from IPython.display import Markdown

def main():

    # st.sidebar.title("About:")
    # st.sidebar.text('''The Pre-Read Generator is a
    # cutting-edge AI Powered application designed to streamline the process of generating comprehensive
    # technical reports on any specified topic. By leveraging the power of advanced AI models and APIs, this app empowers users
    # to produce well-structured reports effortlessly.''')

    st.markdown(
    """
    <style>
        .sidebar-content {
            background-color: black;
            color: white;
            padding: 10px;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True)

# Sidebar content
    st.sidebar.markdown(
    '<div class="sidebar-content">'
    '<h2>About:</h2>'
    '<p>The Pre-Read Generator is a cutting-edge AI Powered application designed to streamline the process of generating comprehensive technical reports on any specified topic. By leveraging the power of advanced AI models and APIs, this app empowers users to produce well-structured reports effortlessly.</p>'
    '</div>',
    unsafe_allow_html=True
    )
    
    st.sidebar.title("Configuration")
    api_key = st.sidebar.text_input("Enter Groq Cloud API Key:", type="password")
    serper_api_key = st.sidebar.text_input("Enter serper_API key:", type="password")
    model = st.sidebar.text_input("Give model:", type="password")

    st.title("Pre-Read Generator")

    # api_key = st.text_input("Enter API key:", type="password")
    topic = st.text_input("Enter the topic:")

    if st.button("Generate Pre-Read Document"):
        if not api_key or not topic:
            st.error("Please enter both the API key and topic.")
            return

        # Initialize Large Language Model (LLM) of your choice
        llm = LLM(model="groq/gemma2-9b-it", api_key=api_key)
        #gemma2-9b-it - Use this model when they ask for model, so mug it won;t be tedious

        # CrewAI agents.......@@@
        Research = Agent(
            role='Research Specialist',
            goal='Gather comprehensive information about the given topic',
            backstory='''Expert researcher with deep knowledge in computer science and technical topics who
                    gathers information for a beginner-friendly pre-read document for any given technical topic.
                    A pre-read which you work on is A concise and beginner-friendly resource designed to introduce students to a specific topic before a lecture,
                    workshop. It starts with Introduction of topic and provides an overview of the topic's key concepts,
                    real-world applications, and foundational knowledge, using clear and accessible language. The goal of a pre-read document is
                    to prepare students with the necessary context, spark curiosity, and enhance their understanding of the subject during formal
                    instruction. It often includes examples, analogies, and links for examples for deeper exploration.''',
            llm=llm,
            verbose=True
        )

        writer = Agent(
            role='Technical Writer',
            goal='Create well-structured, educational pre-read documents',
            backstory='''Expert researcher with deep knowledge in computer science and technical topics who
                    writes content for a beginner-friendly pre-read document for any given technical topic.
                    A pre-read which you work on is A concise and beginner-friendly resource designed to introduce students to a specific topic before a lecture,
                    workshop, or reference links. It starts with Introduction of topic and provides an overview of the topic's key concepts,
                    real-world applications, and foundational knowledge, using clear and accessible language. The goal of a pre-read document is
                    to prepare students with the necessary context, spark curiosity, and enhance their understanding of the subject during formal
                    instruction. It often includes examples, analogies, and links and hyperlinks for examples wherever necessary in the script for deeper exploration.''',
            llm=llm,
            verbose=True
        )

        reviewer = Agent(
            role='Expert Technical Reviewer',
            goal='Ensure accuracy, completeness, and educational value of the content',
            backstory='''Expert researcher with deep knowledge in computer science and technical topics who
                    reviews contents for a beginner-friendly pre-read document for any given technical topic.
                    A pre-read which you review on is A concise and beginner-friendly resource designed to introduce students to a specific topic before a lecture,
                    workshop, or reference links. It starts with Introduction of topic and provides an overview of the topic's key concepts,
                    real-world applications, and foundational knowledge, using clear and accessible language. The goal of a pre-read document is
                    to prepare students with the necessary context, spark curiosity, and enhance their understanding of the subject during formal
                    instruction. It often includes examples, analogies, and links for examples for deeper exploration.''',
            llm=llm,
            verbose=True
        )

        # tasks
        research_task = Task(
                description=f"""Research the topic: {topic}
                Introduction: Provide a clear, engaging introduction to the topic. Explain what it is, why it’s important, and its relevance in today’s world.
                Applications: Highlight real-world applications with specific examples that showcase the practical value of the topic.
                References and Developments: Include recent advancements, and practical use cases to support your findings with some reference links.""",
                expected_output="""A detailed research report:
                Each section (Introduction, Applications,references etc.) should be 2-3 paragraphs long, with clear headings.
                Include bullet points, examples, and references where applicable.
                Ensure the content is accurate, engaging, and easy for a beginner to follow.
                """,
                agent=Research
            )

        writing_task = Task(
                description=f"""Create a pre-read document for {topic} using the research provided.
                Follow this structure:
                1. Title and Introduction: Write an engaging title and provide an introductory section that captures the topic’s essence.
                3. Real-World Applications: Include a section highlighting the topic’s practical applications, supported by relatable examples.
                4. Key Concepts: Explain foundational concepts clearly, using simple language, analogies, or examples to help learners understand.
                5. Suggested Readings and Resources: Include website links which give examples for people who want to dive deeper as well as
                    add hyperlinks wherever necessary.
                6. a) Use markdown with clear headings and subheadings for each section.
                  b) Include bullet points, lists, and visuals where applicable.
                  c) Write in an engaging and friendly tone, as if explaining to someone new to the topic.""",
                expected_output="""A structured markdown document containing:
                1. Title and introduction
                2. Detailed sections with clear headings
                3. All sections in 4-5 paragraphs
                4. Suggested readings and resources.
                Format: Following the provided template structure.""",
                agent=writer
            )
        review_task =  Task(
        description="""Review the document for technical accuracy, completeness, educational value, and clarity.""",
        expected_output="""A review report containing:
                  1. Technical accuracy assessment: Validate that all concepts and examples
                      provided are factually correct and recent. Flag any inconsistencies or outdated information.
                  2. Structure and Flow: Check that the document adheres to this structure:
                        1.Introduction
                        2.Applications
                        3.Key Concepts
                        4.Suggested Readings
                      Provide feedback on whether each section is clear and comprehensive.
                  3. Educational value analysis
                  4. Specific improvement recommendations
                  Length: 1-2 paragraphs with actionable feedback.
                  5. After the script is ready, give it a final go through and changes.
                  6. Proper formatting for headings, sub-headings and content""",
                  agent=reviewer
        )

        # Crew Pipeline
        crew = Crew(
            agents=[Research, writer, reviewer],
            tasks=[research_task, writing_task, review_task],
            verbose=True
        )

        with st.spinner("Generating the document..."):
            result = crew.kickoff()
            output = crew.tasks[1].output.raw

        st.subheader("Generated Pre-Read Document")
        st.markdown(output)

if __name__ == "__main__":
    main()
