import streamlit as st
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from datetime import datetime
import pandas as pd

# Set up LangChain components
output_parser = JsonOutputParser()

prompt_template = ChatPromptTemplate(
    messages=[
        ('system', """"You are an intelligent AI assistant designed to estimate travel costs from a given source to a destination. 
        Based on user inputs, you provide various travel options, including cabs, trains, buses, and flights, along with their estimated costs. 
        Additionally, you consider the selected date to enhance accuracy in your estimations."
                      Output Format Instructions: {output_format_instructions}"""),
        ('human', 'Show me the travel options from {source} to {destination} on {date}')
    ],
    partial_variables={'output_format_instructions': output_parser.get_format_instructions()}
)

# Load API key from environment variables
API_KEY = st.secrets["api_key"]

# Initialize the Gemini model
chat_model = ChatGoogleGenerativeAI(api_key=API_KEY, model='gemini-2.0-flash-exp')

# Define the LangChain pipeline
chain = prompt_template | chat_model | output_parser

# Streamlit UI
st.set_page_config(page_title="AI Travel Planner", layout="centered")
st.title("🚆 TravelGenie – Your AI-powered travel assistant")

# User inputs
source = st.text_input("Enter source location", placeholder="E.g., Delhi")
destination = st.text_input("Enter destination location", placeholder="E.g., Hyderabad")
date = st.date_input("Select travel date", min_value=datetime.today())

if st.button("Find Travel Options"):
    if source and destination and date:
        formatted_date = date.strftime("%Y-%m-%d")  # Convert date to string format
        raw_input = {'source': source, 'destination': destination, 'date': formatted_date}
        response = chain.invoke(raw_input)  # Get response from Gemini

        if "travel_options" in response:

            # Convert response to DataFrame
            travel_options = response["travel_options"]
            df = pd.DataFrame(travel_options)

            # **Remove unnecessary columns (like 'notes', if present)**
            if "details" in df.columns:
                df = df.drop(columns=["details"])

            # **Format estimated_cost column for better readability**
            def format_cost(cost):
                if isinstance(cost, dict):  # Only process if cost is a dictionary
                    if "currency" in cost and "min" in cost and "max" in cost:
                        return f"{cost['currency']} {cost['min']} - {cost['max']}"
                    elif "1AC" in cost and "2AC" in cost:
                        return f"1AC: {cost['1AC']['currency']} {cost['1AC']['min']}-{cost['1AC']['max']}, " \
                               f"2AC: {cost['2AC']['currency']} {cost['2AC']['max']}"
                return str(cost)  # Convert any other type (like int) to string


            df["estimated_cost"] = df["estimated_cost"].apply(format_cost)

            # **Clean up the travel_time column** (Ensure consistent format)
            def clean_duration(duration):
                if isinstance(duration, (int, float)):  # If it's a number, convert it to string
                    return f"{duration} hours"
                duration = str(duration).replace(" - ", "-").strip()  # Remove extra spaces
                if "hours" not in duration:
                    return f"{duration} hours"
                return duration


            df["duration"] = df["duration"].apply(clean_duration)

            # **Display clean DataFrame**
            st.subheader("📊 Travel Cost Breakdown")
            st.dataframe(df.style.set_properties(**{'text-align': 'left'}))

        else:
            st.error("No travel options found. Please try again.")

    else:
        st.warning("Please enter source, destination, and select a date.")