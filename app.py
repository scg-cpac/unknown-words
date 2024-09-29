import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os

# Default prompt template
DEFAULT_PROMPT_TEMPLATE = """Act as a risk management expert familiar with Thailand’s legal, environmental, and regulatory frameworks, as well as global terminologies. Analyze the following text for any specialized, technical, or less commonly known terms. Extract these words and organize them into a table with two columns:

'Words' – The term or phrase that might be unfamiliar, either in Thai or other languages.
'Reasons' – Why the word may not be widely understood, with explanations that account for Thai context, legal or regulatory specifics, industry jargon, or language barriers.

Format as markdown table without any additional information."""

# Load the uploaded file and return the dataframe
def load_uploaded_file(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        return pd.read_excel(uploaded_file)
    return None

# Perform analysis using LangChain with OpenAI
def analyze_text_with_ai(api_key, prompt_template, selected_data, rows_to_analyze):
    llm_model = ChatOpenAI(api_key=api_key, model_name="gpt-4o-mini")
    chain = ChatPromptTemplate(
        [
            ("system", prompt_template + " Let's think step by step"),
            ("user", "{text}")
        ]
    ) | llm_model

    results = []
    for index, row in selected_data.head(rows_to_analyze).iterrows():
        row_text = ' '.join(row.astype(str))
        with st.spinner(f"Analyzing row {index + 1}/{rows_to_analyze}..."):
            response = chain.invoke({'text': row_text})
            results.append({
                "Row": index + 1,
                "Analysis": response.content
            })

    return results

# Display analysis results in Streamlit
def display_analysis_results(results):
    st.write("Analysis Results:")
    for result in results:
        st.markdown(f"**Row {result['Row']} Analysis**:")
        st.write(result['Analysis'])

# Streamlit App Function
def main():
    # Title of the App
    st.title("Extract Words with AI")

    # File uploader: Allow users to upload either a CSV or Excel file
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file:
        # Load the data from the uploaded file
        data_frame = load_uploaded_file(uploaded_file)
        if data_frame is not None:
            # Display the uploaded data for user reference
            st.write("Data Preview:", data_frame.head())

            # Create a multiselect box for users to choose columns for analysis
            selected_columns = st.multiselect("Select Columns to Use for Analysis:", options=data_frame.columns.tolist())

            if selected_columns:
                selected_data = data_frame[selected_columns]
                st.write("You selected these columns for analysis:", selected_data.head())

                # User input to limit the number of rows to analyze
                max_rows = len(data_frame)
                rows_to_analyze = st.number_input(
                    f"Enter the number of rows to analyze (max: {max_rows})",
                    min_value=1,
                    max_value=max_rows,
                    value=min(10, max_rows)
                )

                # User-editable system prompt text area
                system_prompt = st.text_area("Edit System Prompt", value=DEFAULT_PROMPT_TEMPLATE, height=200)

                # Create a button to start the analysis
                if st.button("Analyze Selected Text with AI"):
                    # Initialize OpenAI API through LangChain
                    openai_api_key = os.environ.get('OPENAI_API_KEY')  # Ensure you add your API key to Streamlit secrets
                    if not openai_api_key:
                        st.error("Please set your OpenAI API key.")
                        return

                    # Analyze selected text with AI
                    analysis_results = analyze_text_with_ai(
                        api_key=openai_api_key,
                        prompt_template=system_prompt,
                        selected_data=selected_data,
                        rows_to_analyze=rows_to_analyze
                    )

                    # Display the results
                    display_analysis_results(analysis_results)

            else:
                st.warning("Please select at least one column for analysis.")
        else:
            st.error("Could not read the uploaded file. Please upload a valid CSV or Excel file.")
    else:
        st.info("Upload a CSV or Excel file to proceed.")

# Run the main function
if __name__ == "__main__":
    main()
