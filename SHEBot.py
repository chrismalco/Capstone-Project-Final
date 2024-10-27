import os
import requests
import openai
import streamlit as st
from dotenv import load_dotenv
import json
import pdfplumber
import io
import random
from helper_functions.utility import check_password 

# Load environment variables (OpenAI API key)
load_dotenv('.env')
openai.api_key = os.getenv('OPENAI_API_KEY')

# Streamlit UI Setup
st.set_page_config(page_title="LTA SHEBot", layout="wide")



st.markdown(
    """
    <style>
        body {
            background-color: #f5f5f5;
            color: #333333;
        }
        .stTextInput, .stButton, .stSelectbox {
            border-radius: 8px;
        }
        .response-box {
            border: 2px solid #4CAF50;
            border-radius: 10px;
            background-color: #ffffff;
            padding: 10px;
            margin-top: 15px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# App title and subtitle
st.title("Hi! I am SHEBot.")
st.subheader("Ask me about Safety, Health, and Environmental for construction of LTA projects")

# Check if the password is correct.  
if not check_password():  
    st.stop()

# Step 1: Scrape General Data
def scrape_general_data():
    urls = [
        "https://file.go.gov.sg/gsappa042024.pdf"
    ]
    scraped_info = []
    
    for url in urls:
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            content_type = response.headers.get('Content-Type', '')
            print(f"Fetching data from {url} - Content-Type: {content_type}")

            # Use pdfplumber to extract text from PDF
            with pdfplumber.open(io.BytesIO(response.content)) as pdf:
                paragraphs = []
                headings = [] # Initialize headings as an empty list
                lists = []  # Initialize lists as an empty list

                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        paragraphs.extend(text.split('\n'))  # Split text into paragraphs

                page_data = {
                    'url': url,
                    'headings': headings,
                    'paragraphs': paragraphs,
                    'lists': lists
                }
                scraped_info.append(page_data)

        except requests.exceptions.RequestException as e:
            st.error(f"Failed to fetch data from {url}: {e}")
            print(f"Failed to fetch data from {url}: {e}")

    return scraped_info

# Step 2: Identify Relevant Information Based on User Query
def identify_relevant_information(user_message, scraped_data):
    delimiter = "####"
    content = ""
    for page in scraped_data:
        content += f"Headings: {page['headings']}\n"
        content += f"Paragraphs: {page['paragraphs']}\n"
        content += f"Lists: {page['lists']}\n"

    system_message = f"""
    You will be provided with a user query related to {topic.upper()} in construction of Land Transport Authority projects . \
    The user query will be enclosed in the pair of {delimiter}. \
    
    You have the following information available that was scraped from official sources:
    {content}

    Decide which part of the information is most relevant to answer the user's query. \
    The output should include relevant headings, paragraphs, or list items that are relevant to the user's query. \
    Include all the information related to the keywords of query. \
    Do not be so verbatim.
    
    If no relevant information is found, please respond: "No information found, please clarify your query".

    Ensure your response is only plain text in paragraph format without JSON code.
    """

    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': f"{delimiter}{user_message}{delimiter}"},
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Use a valid model name
            messages=messages
        )
        response_content = response['choices'][0]['message']['content']
        
        # Validate and load as JSON
    
        # Ensure response is plain text without JSON code
        return response_content
    
    except Exception as e:
        st.error(f"Error with OpenAI API: {e}")
        return "Error fetching information. Please try again."

# Streamlit App Execution

# Dropdown for user to choose a topic
topic = st.selectbox("Choose a topic:", ["Safety", "Health", "Environment"])

# Text input for user query
user_query = st.text_input("Enter your query related to the selected topic:")

if st.button("Submit"):
    if user_query:
        scraped_data = scrape_general_data()
        relevant_info = identify_relevant_information(user_query, scraped_data)

        # Display the response in a styled response box
        st.markdown(f"<div class='response-box'>{relevant_info}</div>", unsafe_allow_html=True)
    else:
        st.warning("Please enter a query to get information.")
