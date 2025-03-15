import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Initialize LLM
llm = ChatGroq(
    temperature=0, 
    groq_api_key='gsk_***', #Enter your api key here
    model_name="llama-3.3-70b-specdec"
)

# Streamlit UI
st.title("Cold Email Generator for Job Descriptions 📩")
st.markdown("### Enter the job posting URL, and I'll generate a professional cold email for you!")

# User Input
url = st.text_input("🔗 Enter Job URL", "")

if st.button("Generate Email") and url:
    try:
        # Scrape job description
        loader = WebBaseLoader(url)
        page_data = loader.load().pop().page_content

        # Extract Job Details Prompt
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}

            ### INSTRUCTION:
            Extract job postings and return them as a **valid JSON object**.

            The JSON should strictly follow this format:
            {{
                "role": "<Job Title>",
                "experience": "<Required Experience>",
                "skills": ["<Skill 1>", "<Skill 2>", "<Skill 3>", ...],
                "description": "<Job Description>"
            }}

            - Ensure the JSON is valid (not a list unless there are multiple jobs).
            - No extra formatting, markdown, or explanations.
            - Strings in lists must be properly comma-separated.

            ### JSON OUTPUT:
            """
        )

        # Extract job details
        chain_extract = prompt_extract | llm 
        res = chain_extract.invoke({'page_data': page_data})

        # Parse JSON output
        json_parser = JsonOutputParser()
        job = json_parser.parse(res.content)

        # Email Prompt
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:  
            You are **Karthik Mohan**, a **Data Analyst at Mercedes-Benz** with expertise in **big data analytics, AI, and predictive modeling**.  

            You are applying for the above job.  
            Craft a compelling, **personalized cold email** that highlights how your skills and experience align with the job requirements.  

            ### YOUR BACKGROUND:  
            - **Industry Experience:** Data Analyst at Mercedes-Benz, focusing on data-driven decision-making for vehicle performance, failure prediction, and process automation.  
            - **Key Contributions:**
            - Designed a **Damage KPI** for axle joint failures, reducing unnecessary testing and saving costs.  
            - Built a **cloud-based UI** for test data uploads, improving efficiency and collaboration.  
            - Automated **design data extraction** using Excel VBA, saving engineers 1 hour per day.  
            - Lead Developer of an **AI-driven car health assessment system** to predict part failures.  
            - **AI & Data Science Projects:**
            - **AI-Powered Cold Email Generator** using **Llama 3.1, LangChain, and ChromaDB** to automate personalized outreach.  
            - **Real-world Azure Databricks Project** on **Formula1 racing analytics** (DP203 certified).  
            - **Data-driven ODI World Cup Best XI Selection** using SQL and Spark.  
            - **Technical Skills:** Python, PySpark, SQL, Azure Databricks, Apache Spark, Machine Learning (Scikit-learn, XGBoost), and Data Visualization (Matplotlib, Seaborn, Streamlit).  
            - **Certifications:** Oracle Cloud Infrastructure (Gen AI), Azure Databricks & Spark, Machine Learning for Data Science.  

            ### EMAIL REQUIREMENTS:
            - **Professional, yet engaging.**  
            - **Showcase how your skills align with the job.**  
            - **Personalized, with specific achievements where relevant.**  
            - **No unnecessary preamble—straight to the point.**  

            ### EMAIL (NO PREAMBLE):  
            """

        )

        # Generate the email
        chain_email = prompt_email | llm
        res = chain_email.invoke({"job_description": str(job)})

        # Display the email
        st.subheader("Generated Cold Email 📧")
        st.text_area("Generated Cold Email 📧", value=res.content, height=300)

    except Exception as e:
        st.error(f"An error occurred: {e}")

