"""import toml
from pyvis.network import Network
import matplotlib.pyplot as plt
import pandas as pd
import os


# Load TOML schema from file
toml_file = 'schemas/sampleSchema.toml'  # Replace with your actual file path
with open(toml_file, 'r') as f:
    schema = toml.load(f)
"""

import toml
import tomllib
import os
from pyvis.network import Network

"""
# Directory containing TOML schema files
schema_dir = 'schemas'  # Replace with your actual directory path
toml_files = [f for f in os.listdir(schema_dir) if f.endswith('.toml')]

# Initialize combined schema
combined_schema = {'node': {}, 'relation': {}}

# Load and merge all TOML files
for file in toml_files:
    with open(os.path.join(schema_dir, file), 'rb') as f:
        schema = tomllib.load(f)
        combined_schema['node'].update(schema.get('node', {}))
        combined_schema['relation'].update(schema.get('relation', {}))
"""
def createIteractiveFile(schemaList):
    combined_schema = {'node': {}, 'relation': {}}

    for schema in schemaList:
        combined_schema['node'].update(schema.get('node', {}))
        combined_schema['relation'].update(schema.get('relation', {}))
        
    # Create PyVis network
    net = Network(height="900px", width="100%", directed=True)

    # Define colors
    node_color = "#AED6F1"          # Main nodes
    relation_color = "#F5B7B1"      # Relation nodes
    node_prop_color = "#A9DFBF"     # Node properties
    relation_prop_color = "#F9E79F" # Relation properties

    # Add main nodes and their properties as linked squares
    for node_label, node_info in combined_schema.get('node', {}).items():
        description = node_info.get('node_description', '')
        #net.add_node(node_label, label=node_label, title=f"<b>{node_label}</b><br>{description}", color=node_color, shape="ellipse")
        net.add_node(node_label, label=node_label, title=f"{description}", color=node_color, shape="ellipse")
        
        for i, prop in enumerate(node_info.get('node_properties', [])):
            prop_node = f"{node_label}_prop_{i}"
            prop_label = f"{prop['name']} ({prop['type']})"
            #prop_title = f"<b>{prop['name']}</b><br>Type: {prop['type']}<br>{'Optional' if prop.get('optional') else 'Required'}<br>{prop.get('description', '')}"
            prop_title = f"{'Optional' if prop.get('optional') else 'Required'} - {prop.get('description', '')}"
            net.add_node(prop_node, label=prop_label, title=prop_title, color=node_prop_color, shape="box")
            net.add_edge(node_label, prop_node, arrows="")  # Non-directional

    # Add relations and their properties as linked squares
    for rel_label, rel_info in combined_schema.get('relation', {}).items():
        from_node = rel_info['from_entity']
        to_node = rel_info['to_entity']
        #rel_title = f"<b>{rel_label}</b><br>{rel_info.get('relation_description', '')}"
        rel_title = f"{rel_info.get('relation_description', '')}"
        rel_node = f"rel_{rel_label}"
        net.add_node(rel_node, label=rel_label, title=rel_title, color=relation_color, shape="ellipse")
        
        # Connect relation node between main nodes (directional)
        net.add_edge(from_node, rel_node)
        net.add_edge(rel_node, to_node)
        
        # Add relation properties (non-directional)
        for i, prop in enumerate(rel_info.get('relation_properties', [])):
            prop_node = f"{rel_node}_prop_{i}"
            prop_label = f"{prop['name']} ({prop['type']})"
            #prop_title = f"<b>{prop['name']}</b><br>Type: {prop['type']}<br>{'Optional' if prop.get('optional') else 'Required'}<br>{prop.get('description', '')}"
            prop_title = f"{'Optional' if prop.get('optional') else 'Required'} - {prop.get('description', '')}"
            net.add_node(prop_node, label=prop_label, title=prop_title, color=relation_prop_color, shape="box")
            net.add_edge(rel_node, prop_node, arrows="")  # Non-directional

    # Configure interactive options
    net.set_options('''
    var options = {
      "nodes": {"font": {"size": 14}},
      "edges": {
        "font": {"size": 12, "align": "middle"},
        "arrows": {"to": {"enabled": true}},
        "smooth": {"type": "cubicBezier"}
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true
      },
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.3,
          "springLength": 95
        }
      }
    }
    ''')

    # Write HTML file
    output_file = "knowledge_graph_schema_interactive.html"
    net.write_html(output_file, open_browser=False)

    # Inject legend into HTML automatically
    legend_html = """
    <div style='position:fixed;bottom:10px;left:10px;background:#fff;padding:10px;border:1px solid #ccc;font-family:Arial;font-size:14px;z-index:999;'>
    <b>Legend:</b><br>
    <span style='background-color:#AED6F1;padding:4px;'> Main Node </span><br>
    <span style='background-color:#F5B7B1;padding:4px;'> Relation Node </span><br>
    <span style='background-color:#A9DFBF;padding:4px;'> Node Property </span><br>
    <span style='background-color:#F9E79F;padding:4px;'> Relation Property </span>
    </div>
    """

    with open(output_file, "r+", encoding="utf-8") as f:
        content = f.read()
        content = content.replace("</body>", legend_html + "\n</body>")
        f.seek(0)
        f.write(content)
        f.truncate()

    print(f"Interactive visualization saved as '{output_file}' with legend injected automatically.")
