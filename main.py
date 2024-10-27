import os
import json
import io
import requests
import openai
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
import re 
import requests

# Load environment variables (OpenAI API key)
load_dotenv('.env')

# Pass the API Key to the OpenAI Client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def get_embedding(input, model='text-embedding-3-small'):
    response = client.embeddings.create(
        input=input,
        model=model
    )
    return [x.embedding for x in response.data]

# Streamlit UI Setup
st.set_page_config(page_title="Business Start Up in Singapore", layout="wide")
st.title("Business Start Up in Singapore")
st.subheader(
    "Explore business start-ups and current industry trends and talent needs in Singapore"
)

# Step 1: Scrape General Data
def scrape_general_data():
    url = [
        "https://www.enterprisesg.gov.sg/grow-your-business/boost-capabilities/growth-and-transformation,
        https://www.enterprisesg.gov.sg/grow-your-business/boost-capabilities/productivity-and-digitalisation,
        https://www.enterprisesg.gov.sg/grow-your-business/boost-capabilities/talent-attraction-and-development,
        https://www.enterprisesg.gov.sg/grow-your-business/boost-capabilities/quality-and-standards,
        https://www.enterprisesg.gov.sg/grow-your-business/boost-capabilities/sustainability,
        https://www.enterprisesg.gov.sg/financial-support/productivity-solutions-grant,
        https://www.enterprisesg.gov.sg/financial-support/energy-efficiency-grant,
        https://www.enterprisesg.gov.sg/financial-support/market-readiness-assistance-grant,
        https://www.enterprisesg.gov.sg/financial-support/skillsfuture-enterprise-credit,
        https://www.enterprisesg.gov.sg/financial-support/double-tax-deduction-for-internationalisation,
        https://www.enterprisesg.gov.sg/financial-support/enterprise-financing-scheme,
        https://www.enterprisesg.gov.sg/financial-support/enterprise-financing-scheme-foreign-based-financial-institutions-multilateral-development-banks,
        https://www.enterprisesg.gov.sg/financial-support/enterprise-financing-scheme---green,
        https://www.enterprisesg.gov.sg/financial-support/enterprise-financing-scheme---mergers-and-acquisitions,
        https://www.enterprisesg.gov.sg/financial-support/enterprise-financing-scheme---project-loan,
        https://www.enterprisesg.gov.sg/financial-support/enterprise-financing-scheme---sme-fixed-assets,
        https://www.enterprisesg.gov.sg/financial-support/enterprise-financing-scheme---sme-working-capital,
        https://www.enterprisesg.gov.sg/financial-support/enterprise-financing-scheme---trade-loan,
        https://www.enterprisesg.gov.sg/financial-support/enterprise-financing-scheme---venture-debt,
        https://www.enterprisesg.gov.sg/financial-support/enterprise-development-grant,
        https://www.enterprisesg.gov.sg/financial-support/co-innovation-programmes,
        https://www.enterprisesg.gov.sg/financial-support/edbi,
        https://www.enterprisesg.gov.sg/financial-support/fund-management-incentive,
        https://www.enterprisesg.gov.sg/financial-support/global-trader-programme,
        https://www.enterprisesg.gov.sg/financial-support/lead-trade-fairs,
        https://www.enterprisesg.gov.sg/financial-support/local-enterprise-and-association-development-programme,
        https://www.enterprisesg.gov.sg/financial-support/seeds-capital,
        https://www.enterprisesg.gov.sg/financial-support/startup-sg-accelerator,
        https://www.enterprisesg.gov.sg/financial-support/startup-sg-equity,
        https://www.enterprisesg.gov.sg/financial-support/startup-sg-founder,
        https://www.enterprisesg.gov.sg/financial-support/startup-sg-tech,
        https://www.enterprisesg.gov.sg/financial-support/sustainability-reporting-grant,
        https://www.enterprisesg.gov.sg/financial-support/venture-capital-fund-incentive,
        https://www.enterprisesg.gov.sg/financial-support/sustainability-reporting-grant/learn-more-about-our-programmes,
        https://www.enterprisesg.gov.sg/financial-support/sustainability-reporting-grant/expand-into-southeast-asia
        "
    ]
    scraped_info = []

    for url in urls:
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")

            # Extract relevant information: paragraphs, headings, lists, etc.
            paragraphs = [p.get_text().strip() for p in soup.find_all('p')]
            headings = [h.get_text().strip() for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])]
            lists = [li.get_text().strip() for li in soup.find_all('li')]

            # Combine all extracted information
            page_data = {
                'url': url,
                'headings': headings,
                'paragraphs': paragraphs,
                'lists': lists
            }
            scraped_info.append(page_data)

        except requests.exceptions.RequestException as e:
            st.error(f"Failed to fetch data from {url}: {e}")

    return scraped_info

# Step 2: Identify Relevant Information Based on User Query
def identify_relevant_information(user_message, scraped_data):
    delimiter = "####"

    # Consolidate all scraped information into a single string for easier LLM processing.
    content = ""
    for page in scraped_data:
        content += f"Headings: {page['headings']}\n"
        content += f"Paragraphs: {page['paragraphs']}\n"
        content += f"Lists: {page['lists']}\n"

    system_message = f"""
    You will be provided with a user query related to upskilling or career guidance. \
    The user query will be enclosed in the pair of {delimiter}. \
    
    You have the following information available that was scraped from official sources:
    {content}

    Decide which part of the information is most relevant to answer the user's query. \
    The output should include relevant headings, paragraphs, or list items that are relevant to the user's query. \
    
    If no relevant information is found, output an empty JSON array: [].

    Ensure your response is **only** a valid JSON array containing the most relevant points, without any additional text or comments.
    """

    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': f"{delimiter}{user_message}{delimiter}"},
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=2048,
        temperature=0.7
    )

    relevant_information_response_str = response.choices[0].message.content

    # Display LMM Raw Data (Optional)
    if st.checkbox("LLM Raw Data"):
        st.write("LLM Raw Response:", relevant_information_response_str)
        if not relevant_information_response_str.strip():
            return []  # Return an empty list if the response is empty
        
    # Attempt to extract valid JSON using regex
    json_match = re.search(r'(\[.*\])', relevant_information_response_str, re.DOTALL)
    if json_match:
        relevant_information_response_str = json_match.group(1)
    else:
        st.error("Failed to find JSON in the response from the LLM. Please try again.")
        return []

    # Try parsing the JSON
    try:
        relevant_information = json.loads(relevant_information_response_str)
    except json.JSONDecodeError:
        st.error("Failed to parse the response from the LLM. Please try again.")
        relevant_information = []

    return relevant_information

# Step 3: Generate a Detailed Response
def generate_response_based_on_scraped_info(user_message, relevant_info):
    delimiter = "####"

    system_message = f"""
    Follow these steps to answer user queries related to upskilling or career guidance.
    The user query will be delimited with a pair of {delimiter}.

    Step 1:{delimiter} If the user is asking about upskilling or career guidance, \
    understand the relevant information from the list below.
    Available details are shown in the JSON data below:
    {relevant_info}

    Step 2:{delimiter} Use the information to generate an answer to the user query.
    Your response should be detailed, comprehensive, and help the user understand the options available to them.

    Step 3:{delimiter} Answer the user in a friendly and informative tone.
    Make sure the statements are factually accurate. The response should be complete with helpful information \
    that assists the user in making decisions.

    Use the following format:
    Step 1:{delimiter} <step 1 reasoning>
    Step 2:{delimiter} <step 2 reasoning>
    Step 3:{delimiter} <step 3 response to user>

    Make sure to include {delimiter} to separate every step.
    """

    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': f"{delimiter}{user_message}{delimiter}"},
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=2048,
        temperature=0.7
    )

    response_to_user = response.choices[0].message.content
    response_to_user = response_to_user.split(delimiter)[-1]
    return response_to_user

# Step 4: Main Query Handling
scraped_data = scrape_general_data()  # Move scraped_data outside the if block to make it accessible globally

user_query = st.text_input("Enter your question:", placeholder="E.g., 'How do I use my SkillsFutire credits?'")
if user_query:
    st.write("Searching for relevant information...")
    relevant_info = identify_relevant_information(user_query, scraped_data)

    if relevant_info:
        # Generate a short response to use as the subheader
        subheader_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {'role': 'user', 'content': f"Provide a brief response to the following query: '{user_query}'"}
            ],
            max_tokens=2048,
            temperature=0.7
        )
        subheader_text = subheader_response.choices[0].message.content.strip()
        
        reply = generate_response_based_on_scraped_info(user_query, relevant_info)
        st.subheader(subheader_text)
        st.write(reply)
    else:
        st.write(f"No relevant information found for your query.")

# Display Raw Data (Optional)
if st.checkbox("Show Raw Data"):
    st.write("Scraped Information:")
    st.json(scraped_data)
    response = requests.get(url, headers=headers)

