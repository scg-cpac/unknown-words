import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
import json

# Default prompt template
DEFAULT_PROMPT_TEMPLATE = """เรากำลังจะทำ Business glossary (อธิบายความหมายของคำเฉพาะ) สำหรับไปใช้ในระบบ AI ให้รู้จักธุรกิจเราและช่วยตอบคำถามได้ดีขึ้น
 
จงถอดคำศัพท์จากข้อความด้านล่าง โดยถอดคำศัพท์ที่เราจะต้องเขียนคำอธิบายให้ AI รู้จัก จำนวนไม่เกิน 10 คำศัพท์ เน้นที่คำศัพท์ ตัวย่อ ชื่อเฉพาะ ที่ AI น่าจะไม่ทราบและจำเป็นต้องอธิบายเพิ่มให้ AI เข้าใจ  ตัดมาทั้งคำ ห้ามตกแต่งแก้ไขคำศัพท์ โดยตอบในรูปแบบตาราง id (ใช้ข้อมูลจาก (id:xxx)) | No. | Word | Name (กรณีเป็นชื่อเฉพาะ ระบุ Y, คำศัพท์ทั่วไป ระบุ N) | Translation (คำแปล)| Suggest description (Column นี้ ถ้าไม่แน่ใจให้ตอบว่า : ไม่มีคำแนะนำ) | Example sentence (ตัวอย่างประโยคที่พบ ตัดคำก่อนหน้าและตามหลัง รวมกันไม่เกิน 10 คำ ไม่ต้องยกมาทั้งประโยค)  

**ไม่ต้องรวม คำศัพท์ทั่วไป ที่ น่าจะทราบอยู่แล้ว

pls format as json
 
ข้อมูลตั้งต้นในการค้นหาคำศัพท์ดังนี้"""

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
        # Assuming 'row' is a Pandas Series with a column named "id"
        # Extract the 'id' value and the rest of the row
        id_value = row['id']

        # Remove the 'id' column from the row to avoid redundancy
        row_without_id = row.drop(labels='id')

        # Join the remaining row and add the 'id' value in front
        row_text = f"(id:{id_value}) " + ' '.join(row_without_id.astype(str))
        with st.spinner(f"Analyzing row {index + 1}/{rows_to_analyze}..."):
            response = chain.invoke({'text': row_text})
            results.append({
                "Row": index + 1,
                "Analysis": response.content
            })

    return results

# Display analysis results in Streamlit
import json

# Display analysis results in Streamlit
def display_analysis_results(results):
    st.write("Analysis Results:")
    structured_results = []

    for result in results:
        # Display the raw response for each row
        with st.expander(f"Row {result['Row']} Analysis"):
            st.write(result['Analysis'])

        result['Analysis'] = result['Analysis'].replace('json','').replace('```','').strip()
        
        # Check if the response content is empty
        if not result['Analysis']:
            st.warning(f"No content found for row {result['Row']}. Skipping.")
            continue

        try:
            # Validate that the analysis is in proper JSON format
            analysis_data = json.loads(result['Analysis'])
            structured_results.extend(analysis_data)
        except json.JSONDecodeError as e:
            st.error(f"Error parsing analysis for row {result['Row']}: {e}")
            continue

    if structured_results:
        # Convert valid structured results to a DataFrame
        try:
            results_df = pd.json_normalize(structured_results)

            # Displaying the DataFrame (optional)
            st.write("Structured Analysis Results:")
            st.write(results_df)

            # Convert DataFrame to CSV and JSON
            csv = results_df.to_csv(index=False).encode('utf-8')
            json_str = results_df.to_json(orient='records', force_ascii=False, indent=2)

            # Download buttons for CSV and JSON
            st.download_button(
                label="Download analysis as CSV",
                data=csv,
                file_name='analysis_results.csv',
                mime='text/csv',
            )

            st.download_button(
                label="Download analysis as JSON",
                data=json_str,
                file_name='analysis_results.json',
                mime='application/json',
            )

        except Exception as e:
            st.error(f"Error in processing structured analysis results: {e}")
    else:
        st.warning("No valid analysis results to display or download.")


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
            selected_columns = st.multiselect("Select Columns to Use for Analysis:", options=data_frame.columns.tolist(), default=['id', 'latestDetail_eventDescription', 'latestDetail_rootCause', 'latestDetail_indicator', 'latestDetail_triggerPoint', 'latestDetail_assessments[0]_keyCurrentControls', 'latestDetail_assessments[0]_impactAssessmentItems[0]_assumption', 'latestDetail_assessments[0]_impactAssessmentItems[0]_financialImpact', 'latestDetail_assessments[0]_impactAssessmentItems[1]_assumption', 'latestDetail_assessments[0]_impactAssessmentItems[1]_financialImpact', 'latestDetail_assessments[0]_impactAssessmentItems[2]_assumption', 'latestDetail_assessments[0]_impactAssessmentItems[2]_financialImpact', 'latestDetail_assessments[0]_impactAssessmentItems[3]_assumption', 'latestDetail_assessments[0]_impactAssessmentItems[3]_financialImpact', 'latestDetail_assessments[0]_impactAssessmentItems[4]_assumption', 'latestDetail_assessments[0]_impactAssessmentItems[4]_financialImpact', 'latestDetail_assessments[0]_impactAssessmentItems[5]_assumption', 'latestDetail_assessments[0]_impactAssessmentItems[5]_financialImpact', 'latestDetail_mitigationPlans[0]_description', 'latestDetail_mitigationPlans[1]_description', 'latestDetail_mitigationPlans[2]_description', 'latestDetail_mitigationPlans[3]_description', 'latestDetail_mitigationPlans[4]_description', 'latestDetail_mitigationPlans[5]_description', 'latestDetail_mitigationPlans[6]_description', 'latestDetail_mitigationPlans[7]_description', 'latestDetail_mitigationPlans[8]_description', 'latestDetail_mitigationPlans[9]_description', 'latestDetail_mitigationPlans[10]_description', 'latestDetail_mitigationPlans[11]_description'])

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
                    openai_api_key = os.environ.get('OPENAI_API_KEY') # Ensure you add your API key to Streamlit secrets
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
