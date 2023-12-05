import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
from io import StringIO

st.title("Survey Results Visualization")
json_column = 'event_json'

@st.cache(allow_output_mutation=True)
def load_and_process_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    if json_column in df.columns:
        df[json_column] = df[json_column].apply(json.loads)
        json_df = df[json_column].apply(pd.Series)
        df = df.drop(json_column, axis=1)
        df = pd.concat([df, json_df], axis=1)
    return df

@st.cache(allow_output_mutation=True)
def load_mappings(uploaded_file):
    return pd.read_csv(uploaded_file)

def df_to_csv(df):
    csv_stream = StringIO()
    df.to_csv(csv_stream, index=False)
    return csv_stream.getvalue()

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
uploaded_mappings = st.file_uploader("Upload Question-Answer Mapping CSV file", type=["csv"])

if uploaded_file:
    df = load_and_process_data(uploaded_file)

    if st.checkbox("Show raw data"):
        st.write(df)

    if uploaded_mappings:
        df_mappings = load_mappings(uploaded_mappings)
        question_map = pd.Series(df_mappings.Description.values, index=df_mappings.Question.astype(str)).to_dict()
        
        # Modify answer mappings
        df.columns = df.columns.map(lambda x: question_map.get(str(x), x))
        for i, col in enumerate(df.columns):
            if col in question_map.values():
                specific_answer_map = {str(k): v for k, v in enumerate(df_mappings.loc[df_mappings['Description'] == col].iloc[:, 2:].values[0]) if pd.notna(v)}
                df[col] = df[col].apply(lambda x: ', '.join([specific_answer_map.get(str(val).strip(), str(val)) for val in str(x).split(',') if val]))

    # Filtering columns
    filter_cols = st.multiselect("Columns to filter by", df.columns.tolist())
    for col in filter_cols:
        if df[col].dtype == 'object':  
            options = list(df[col].unique())
            selected_options = st.multiselect(f"Select values for {col}", options, options)
            df = df[df[col].isin(selected_options)]
        else:  
            user_input = st.number_input(f"Enter a value for {col} to filter by")
            if user_input:
                df = df[df[col] == user_input]
                
    column_to_plot = st.selectbox("Select column to visualize", df.columns)
    
    if st.button('Plot Survey Results'):
        # Splitting the "Other" options
        df['Main Options'] = df[column_to_plot].apply(lambda x: x.split('Other,')[0] if 'Other,' in x else x)
        df['Other Options'] = df[column_to_plot].apply(lambda x: x.split('Other,')[1] if 'Other,' in x else None)
        
        # Cleaning the data to avoid duplicated entries due to case differences or extra spaces
        df['Main Options'] = df['Main Options'].str.lower().str.strip()
        df['Other Options'] = df['Other Options'].str.lower().str.strip()
        
        main_answers = df['Main Options'].str.split(', ', expand=True).stack().str.lower().str.strip().value_counts()
        other_answers = df['Other Options'].fillna('').astype(str).dropna().str.split(', ', expand=True).stack().str.lower().str.strip().value_counts()
        
        # First Histogram (Main Options)
        plt.figure(figsize=(10, 6))
        main_answers.plot(kind='bar')
        plt.title(f'Survey Results of {column_to_plot} - Main Options')
        plt.xlabel('Answer Option')
        plt.ylabel('Count')
        st.pyplot(plt)
        plt.clf()
        
        # Second Histogram (Other Options)
        if not other_answers.empty:
            plt.figure(figsize=(10, 6))
            other_answers.plot(kind='bar')
            plt.title(f'Survey Results of {column_to_plot} - Other Options')
            plt.xlabel('Answer Option')
            plt.ylabel('Count')
            st.pyplot(plt)
            plt.clf()
        else:
            st.warning("No additional details provided in 'Other' options.")


    if st.button('Download Processed Data as CSV'):
        st.download_button(
            label="Download CSV",
            data=df_to_csv(df),
            file_name="processed_data.csv",
            mime="text/csv"
        )
