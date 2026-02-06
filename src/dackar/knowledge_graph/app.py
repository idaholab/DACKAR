
# app.py
# Notes
# 1) from terminal go to: DACKAR/src/dackar/knowledge_graph/
# 2) how to run it: streamlit run app.py, a webpage should open automatically on chrome

import os, sys
cwd = os.getcwd()
frameworkDir = os.path.abspath(os.path.join(cwd, os.pardir, os.pardir))
sys.path.insert(0, frameworkDir)

from dackar.knowledge_graph.KGconstruction import KG

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import json
import logging
from io import StringIO
from datetime import datetime

# -----------------------------
# Helper: initialize KG from JSON
# -----------------------------
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

# -----------------------------
# Main Streamlit app
# -----------------------------
def main():
    st.title("KG Graph Construction Interface")

    # Persist KG instance and loaded schemas across reruns
    if "kg_instance" not in st.session_state:
        st.session_state.kg_instance = None
    if "loaded_schemas" not in st.session_state:
        # Each entry: {"Schema Name": str, "File": str, "Source": "uploaded"|"predefined", "Loaded At": str}
        st.session_state.loaded_schemas = []

    # --- Initialize KG ---
    st.header("Initialize KG from JSON")
    initializer_file = st.file_uploader("Upload KG Initialization Parameters JSON file", type=["json"])
    if initializer_file:
        if st.button("Initialize KG"):
            st.session_state.kg_instance = initialize_kg_from_json(initializer_file)
            if st.session_state.kg_instance:
                st.success("KG instance initialized successfully!")

    kg_instance = st.session_state.kg_instance

    # --- Import user-provided TOML ---
    st.header("Import Graph Schema (Upload)")
    schema_file = st.file_uploader("Upload TOML file", type=["toml"])
    schema_name = st.text_input("Enter Schema Name")
    if st.button("Import Schema"):
        if kg_instance and schema_file and schema_name:
            # Save uploaded file to local disk so KG can read it deterministically
            schema_path = f"./{schema_file.name}"
            with open(schema_path, "wb") as f:
                f.write(schema_file.getbuffer())

            # Import into KG
            kg_instance.importGraphSchema(schema_name, schema_path)

            # Track in session
            st.session_state.loaded_schemas.append({
                "Schema Name": schema_name,
                "File": schema_path,  # store actual path
                "Source": "uploaded",
                "Loaded At": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
            st.success(f"Schema '{schema_name}' imported successfully!")
        else:
            st.warning("Please ensure KG is initialized, a TOML file is uploaded, and a schema name is provided.")

    # --- Load predefined TOML ---
    st.header("Load Predefined Graph Schemas")
    if kg_instance:
        predefined_schemas = kg_instance.predefinedGraphSchemas  # {name: path}
        selected_schema = st.selectbox("Select a predefined schema", list(predefined_schemas.keys()))
        if st.button("Load Predefined Schema"):
            schema_path = predefined_schemas[selected_schema]
            kg_instance.importGraphSchema(selected_schema, schema_path)

            st.session_state.loaded_schemas.append({
                "Schema Name": selected_schema,
                "File": schema_path,
                "Source": "predefined",
                "Loaded At": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
            st.success(f"Predefined schema '{selected_schema}' imported successfully!")
    else:
        st.warning("KG instance is not initialized. Please initialize KG to load predefined schemas.")

    # --- NEW: Loaded Schemas Table with Remove ---
    st.header("Loaded Schemas Table")
    if st.session_state.loaded_schemas:
        st.write("Use **Remove** to delete a schema from the KG and this list.")

        # Optional: show a read-only dataframe for quick scanning
        df = pd.DataFrame(st.session_state.loaded_schemas)
        st.dataframe(df, use_container_width=True)

        # Row-by-row controls
        st.divider()
        st.subheader("Manage Loaded Schemas")
        for idx, entry in enumerate(st.session_state.loaded_schemas):
            cols = st.columns([3, 4, 2, 2])  # Name, File, Source, Remove
            cols[0].write(f"**{entry['Schema Name']}**")
            cols[1].write(entry["File"])
            cols[2].write(entry["Source"])

            # Provide a distinct key to prevent collisions across reruns
            if cols[3].button("Remove", key=f"remove_{idx}"):
                schema_name_to_remove = entry["Schema Name"]

                # 1) Remove from KG instance
                if kg_instance:
                    try:
                        kg_instance.removeGraphSchema(schema_name_to_remove)
                        st.success(f"Schema '{schema_name_to_remove}' removed from KG.")
                    except Exception as e:
                        st.error(f"Failed to remove schema '{schema_name_to_remove}' from KG: {e}")
                        # If removal failed, do not alter table; continue to next row
                        continue

                # 2) Optionally delete uploaded TOML from disk (predefined are left intact)
                try:
                    if entry["Source"] == "uploaded":
                        file_path = entry["File"]
                        if os.path.exists(file_path):
                            os.remove(file_path)
                except Exception as e:
                    # Non-fatal; just notify
                    st.warning(f"Could not delete uploaded file '{entry['File']}': {e}")

                # 3) Remove from session table and rerun to refresh UI
                st.session_state.loaded_schemas.pop(idx)
                st.rerun()
    else:
        st.info("No schemas loaded yet.")

    # --- Interactive graph visualization ---
    st.header("Check Loaded Schemas (Interactive Graph)")
    if kg_instance:
        if st.button("Show Interactive Graph"):
            # Capture warnings from crossSchemasCheck
            log_stream = StringIO()
            handler = logging.StreamHandler(log_stream)
            handler.setLevel(logging.WARNING)
            logging.getLogger().addHandler(handler)

            kg_instance.crossSchemasCheck()

            logging.getLogger().removeHandler(handler)
            handler.flush()
            warnings_output = log_stream.getvalue().strip()

            if warnings_output:
                with st.expander("Cross-Schema Warnings"):
                    st.text(warnings_output)
            else:
                st.success("No cross-schema warnings detected.")

            # Generate and display interactive graph
            try:
                kg_instance._createIteractivePlot()
                html_file = "knowledge_graph_schema_interactive.html"
                with open(html_file, "r", encoding="utf-8") as f:
                    html_content = f.read()
                st.title("Interactive Knowledge Graph Schema")
                components.html(html_content, height=900, scrolling=True)
            except FileNotFoundError:
                st.error(f"Could not find '{html_file}'. Ensure _createIteractivePlot() generates this file.")
    else:
        st.warning("KG instance is not initialized. Please initialize KG to check loaded schemas.")


    # --- NEW: Persist Construction Schemas (JSON) across reruns ---
    if "construction_schemas" not in st.session_state:
        # Each entry: {"Schema Name": str, "File": str, "Source": "uploaded", "Loaded At": str}
        st.session_state.construction_schemas = []


    # --- NEW: Construction Schemas (Upload & Manage) ---
    st.header("Construction Schemas (Upload & Manage)")
    cs_file = st.file_uploader("Upload Construction Schema (JSON)", type=["json"])
    cs_name = st.text_input("Enter Construction Schema Name")

    if st.button("Import Construction Schema"):
        if cs_file and cs_name:
            # Ensure local folder exists
            os.makedirs("construction_schemas", exist_ok=True)
            cs_path = os.path.join("construction_schemas", f"{cs_name}.json")

            # Save uploaded file to deterministic path
            with open(cs_path, "wb") as f:
                f.write(cs_file.getbuffer())

            # Track in session (overwrite if name already exists)
            existing_idx = next((i for i, e in enumerate(st.session_state.construction_schemas)
                                if e["Schema Name"] == cs_name), None)
            meta_entry = {
                "Schema Name": cs_name,
                "File": cs_path,
                "Source": "uploaded",
                "Loaded At": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            if existing_idx is not None:
                st.session_state.construction_schemas[existing_idx] = meta_entry
                st.info(f"Construction schema '{cs_name}' updated.")
            else:
                st.session_state.construction_schemas.append(meta_entry)
                st.success(f"Construction schema '{cs_name}' imported successfully!")
        else:
            st.warning("Please provide both a JSON file and a schema name.")

        # List & Remove construction schemas
        st.subheader("Loaded Construction Schemas")
        if st.session_state.construction_schemas:
            df_cs = pd.DataFrame(st.session_state.construction_schemas)
            st.dataframe(df_cs, use_container_width=True)

            st.divider()
            st.subheader("Manage Construction Schemas")
            for idx, entry in enumerate(st.session_state.construction_schemas):
                cols = st.columns([3, 4, 2, 2])  # Name, File, Source, Remove
                cols[0].write(f"**{entry['Schema Name']}**")
                cols[1].write(entry["File"])
                cols[2].write(entry["Source"])

                if cols[3].button("Remove", key=f"remove_cs_{idx}"):
                    # Delete file from disk (only our uploaded ones)
                    try:
                        file_path = entry["File"]
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        st.warning(f"Could not delete file '{entry['File']}': {e}")

                    # Remove from session and refresh UI
                    st.session_state.construction_schemas.pop(idx)
                    st.rerun()
        else:
            st.info("No construction schemas loaded yet.")



    # --- UPDATED: Import Data through Generic Workflow ---
    st.header("Import Data through Generic Workflow")

    # Upload data file
    data_file = st.file_uploader("Upload Data File", type=["csv", "xlsx"])

    # Select a stored construction schema
    available_cs_names = [e["Schema Name"] for e in st.session_state.construction_schemas]
    selected_cs_name = st.selectbox("Select a stored Construction Schema", options=available_cs_names) if available_cs_names else None

    # Optional: Preview selected construction schema JSON
    if selected_cs_name:
        try:
            selected_cs_path = next(e["File"] for e in st.session_state.construction_schemas if e["Schema Name"] == selected_cs_name)
            with open(selected_cs_path, "r") as f:
                preview_schema = json.load(f)
            with st.expander("Preview Selected Construction Schema"):
                st.json(preview_schema)
        except Exception as e:
            st.warning(f"Could not load selected construction schema: {e}")

    # Import button
    if st.button("Import Data"):
        if kg_instance and data_file and selected_cs_name:
            # Load dataframe robustly based on extension
            ext = os.path.splitext(data_file.name)[1].lower()
            try:
                if ext == ".csv":
                    data_df = pd.read_csv(data_file)
                elif ext in (".xlsx", ".xls"):
                    data_df = pd.read_excel(data_file)
                else:
                    st.error(f"Unsupported file type: {ext}")
                    st.stop()
            except Exception as e:
                st.error(f"Failed to read data file: {e}")
                st.stop()

            # Load the chosen construction schema JSON from disk
            try:
                selected_cs_path = next(e["File"] for e in st.session_state.construction_schemas if e["Schema Name"] == selected_cs_name)
                with open(selected_cs_path, "r") as f:
                    construction_schema = json.load(f)
            except Exception as e:
                st.error(f"Failed to load construction schema '{selected_cs_name}': {e}")
                st.stop()

            # Execute workflow
            try:
                kg_instance.genericWorkflow(data_df, construction_schema)
                st.success("Data imported successfully!")
            except Exception as e:
                st.error(f"Failed to import data via genericWorkflow: {e}")
        else:
            st.warning("Please ensure KG is initialized, a data file is uploaded, and a construction schema is selected.")


if __name__ == "__main__":
    main()
