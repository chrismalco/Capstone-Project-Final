import os
import json
import io
import requests
import openai
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import re

# Load environment variables (OpenAI API key)
load_dotenv('.env')

# Pass the API Key to the OpenAI Client
openai.api_key = os.getenv('OPENAI_API_KEY')

def get_embedding(input, model='gpt-4o-mini'):
    response = openai.Embedding.create(
        input=input,
        model=model
    )
    return [x['embedding'] for x in response['data']]

# Streamlit UI Setup
st.set_page_config(page_title="Business Start Up Bot", layout="wide")
st.title("Business Start Up Bot")
st.subheader("Explore business growth and financial support in Singapore")

# Step 1: Scrape General Data
def scrape_general_data():
    urls = [
    "https://www.enterprisesg.gov.sg/financial-support/market-readiness-assistance-grant"
    ]
    scraped_info = []
    
    # Loop through each URL
    for url in urls:
        try:
            response = requests.get(url)
            response.raise_for_status()

            # Debugging: Check the Content-Type of the response
            content_type = response.headers.get('Content-Type', '')
            print(f"Fetching data from {url} - Content-Type: {content_type}")

            # Check if the response is HTML
            if 'text/html' in content_type:
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
            else:
                print(f"Skipping non-HTML content from {url}")

        except requests.exceptions.RequestException as e:
            st.error(f"Failed to fetch data from {url}: {e}")
            print(f"Failed to fetch data from {url}: {e}")  # Print error for debugging

    # Return the scraped data  
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
    You will be provided with a user query related to business growth support and financial assistance in Singapore. \
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

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Add the correct model name
            messages=messages
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        st.error(f"Error with OpenAI API: {e}")

# Usage example (to be called in the Streamlit app or elsewhere)
scraped_data = scrape_general_data()
user_query = st.text_input("Enter your query related to business growth or financial support:")
relevant_info = identify_relevant_information(user_query, scraped_data)
st.json(relevant_info)

if st.button("Submit"):
    if user_query:  # Check if the user has entered a query
        scraped_data = scrape_general_data()  # Fetch data from the URLs
        relevant_info = identify_relevant_information(user_query, scraped_data)
        st.json(relevant_info)  # Display the response
    else:
        st.warning("Please enter a query to get information.")