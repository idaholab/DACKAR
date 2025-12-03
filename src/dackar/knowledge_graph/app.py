# app.py

import os, sys
cwd = os.getcwd()
frameworkDir = os.path.abspath(os.path.join(cwd, os.pardir, os.pardir))
sys.path.insert(0, frameworkDir) 

from dackar.knowledge_graph.KGconstruction import KG


import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import json

# Function to initialize KG instance from a JSON file
def initialize_kg_from_json(json_file):
    if json_file is None:
        st.error("Please upload the initialization parameters JSON file.")
        return None

    data = json.load(json_file)
    config_file_path = data["config_file_path"]
    import_folder_path = data["import_folder_path"]
    uri = data["uri"]
    user = data["user"]
    pwd = data["pwd"]

    return KG(config_file_path, import_folder_path, uri, pwd, user)

# Main Streamlit app function
def main():
    st.title("KG Graph Construction Interface")

    # Use session_state to maintain the KG instance across reruns
    if 'kg_instance' not in st.session_state:
        st.session_state.kg_instance = None

    # Section for initializing KG instance from a JSON file
    st.header("Initialize KG from JSON")
    initializer_file = st.file_uploader("Upload KG Initialization Parameters JSON file", type=["json"])

    if initializer_file:
        if st.button("Initialize KG"):
            st.session_state.kg_instance = initialize_kg_from_json(initializer_file)
            if st.session_state.kg_instance:
                st.success("KG instance initialized successfully!")

    kg_instance = st.session_state.kg_instance

    # Section for importing graph schemas
    st.header("Import Graph Schema")
    schema_file = st.file_uploader("Upload TOML file", type=["toml"])
    schema_name = st.text_input("Enter Schema Name")
    if st.button("Import Schema"):
        if kg_instance and schema_file and schema_name:
            schema_path = f"./{schema_file.name}"
            with open(schema_path, "wb") as f:
                f.write(schema_file.getbuffer())
            kg_instance.importGraphSchema(schema_name, schema_path)
            st.success(f"Schema {schema_name} imported successfully!")

    # Section for loading predefined graph schemas
    st.header("Load Predefined Graph Schemas")
    if kg_instance:
        predefined_schemas = kg_instance.predefinedGraphSchemas
        selected_schema = st.selectbox("Select a predefined schema", list(predefined_schemas.keys()))

        if st.button("Load Predefined Schema"):
            kg_instance.importGraphSchema(selected_schema, predefined_schemas[selected_schema])
            st.success(f"Predefined schema {selected_schema} imported successfully!")
    else:
        st.warning("KG instance is not initialized. Please initialize KG to load predefined schemas.")

    # Section for checking loaded schemas
    st.header("Check Loaded Schemas")
    if kg_instance:
        if st.button("Show Loaded Schemas"):
            kg_instance._createIteractivePlot()
            html_file = "knowledge_graph_schema_interactive.html"
            # Read the HTML content
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            # Display in Streamlit
            st.title("Interactive Knowledge Graph Schema")
            components.html(html_content, height=900, scrolling=True)

    else:
        st.warning("KG instance is not initialized. Please initialize KG to check loaded schemas.")

    # Section for importing data through genericWorkflow
    st.header("Import Data through Generic Workflow")
    data_file = st.file_uploader("Upload Data File", type=["csv", "xlsx"])
    construction_schema_file = st.file_uploader("Upload Construction Schema JSON file", type=["json"])

    if st.button("Import Data"):
        if kg_instance and data_file and construction_schema_file:
            # Load the data file
            if data_file.type == "text/csv":
                data_df = pd.read_csv(data_file)
            elif data_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                data_df = pd.read_excel(data_file)

            # Load the construction schema file
            construction_schema_path = f"./{construction_schema_file.name}"
            with open(construction_schema_path, "wb") as f:
                f.write(construction_schema_file.getbuffer())
            with open(construction_schema_path, "r") as f:
                construction_schema = json.load(f)

            # Call the genericWorkflow method
            kg_instance.genericWorkflow(data_df, construction_schema)
            st.success("Data imported successfully!")

if __name__ == '__main__':
    main()