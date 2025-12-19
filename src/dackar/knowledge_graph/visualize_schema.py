
from pathlib import Path
import logging
from typing import Dict, Any, Iterable, Tuple, List, Union
from pyvis.network import Network

# If you need 3.10 compatibility, add fallback:
try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib   # Python 3.10 fallback


def _iter_relations(relations_section: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """
    Yield (relation_name, relation_entry) for each concrete relation entry.
    Supports both:
      - relation_name -> dict
      - relation_name -> list[dict]
    """
    if not isinstance(relations_section, dict):
        return  # or raise ValueError

    for name, value in relations_section.items():
        if isinstance(value, dict):
            yield (name, value)
        elif isinstance(value, list):
            for entry in value:
                if isinstance(entry, dict):
                    yield (name, entry)
                else:
                    logging.warning(f"Relation '{name}' contains non-dict entry: {type(entry)}")
        else:
            logging.warning(f"Relation '{name}' is not dict or list: {type(value)}")


def _collect_nodes(schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    nodes = schema.get("node", {})
    if not isinstance(nodes, dict):
        logging.warning("'node' section is not a dict.")
        return {}
    return nodes


def _collect_relations(schema: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    relations = schema.get("relation", {})
    return list(_iter_relations(relations))


def createInteractiveFile(schema_list: List[Dict[str, Any]],
                          output_file: str = "knowledge_graph_schema_interactive.html",
                          namespace_by_schema: bool = False,
                          raise_on_collision: bool = True) -> None:
    """
    Build an interactive HTML visualization from a list of parsed schema dicts.

    Parameters
    ----------
    schema_list : list[dict]
        Each dict is a parsed TOML schema with 'node' and 'relation'.
    output_file : str
        Where to write the HTML.
    namespace_by_schema : bool
        If True, prefix node and relation labels with schema index to prevent collisions.
    raise_on_collision : bool
        If True, raise on duplicate node or relation names; else, log warning and overwrite.
    """
    # Build combined collections with optional namespacing
    combined_nodes: Dict[str, Dict[str, Any]] = {}
    combined_relations: List[Tuple[str, Dict[str, Any]]] = []

    for idx, schema in enumerate(schema_list):
        ns_prefix = f"s{idx}::" if namespace_by_schema else ""

        nodes = _collect_nodes(schema)
        for node_label, node_info in nodes.items():
            key = ns_prefix + node_label
            if key in combined_nodes:
                msg = f"Duplicate node name after namespacing: '{key}'."
                if raise_on_collision:
                    raise ValueError(msg)
                logging.warning(msg)
            combined_nodes[key] = node_info

        for rel_name, rel_entry in _collect_relations(schema):
            # Normalize endpoints and move any description
            # Standardize expected fields: from_node and to_node
            from_node = rel_entry.get("from_node")
            to_node = rel_entry.get("to_node")

            if from_node is None or to_node is None:
                logging.warning(f"Relation '{rel_name}' missing endpoints: {rel_entry}")
                continue

            # Apply same namespacing to endpoints so they connect to namespaced nodes if enabled
            rel_copy = dict(rel_entry)
            rel_copy["from_node"] = ns_prefix + from_node if namespace_by_schema else from_node
            rel_copy["to_node"] = ns_prefix + to_node if namespace_by_schema else to_node

            combined_relations.append((ns_prefix + rel_name if namespace_by_schema else rel_name, rel_copy))

    # Optional: detect relation duplicates by (name, from_node, to_node)
    seen_rel_keys = set()
    for name, rel in combined_relations:
        key = (name, rel.get("from_node"), rel.get("to_node"))
        if key in seen_rel_keys:
            msg = f"Duplicate relation entry: {key}"
            if raise_on_collision:
                raise ValueError(msg)
            logging.warning(msg)
        seen_rel_keys.add(key)

    # Create PyVis network
    net = Network(height="900px", width="100%", directed=True)

    # Colors
    node_color = "#AED6F1"          # Main nodes
    relation_color = "#F5B7B1"      # Relation nodes
    node_prop_color = "#A9DFBF"     # Node properties
    relation_prop_color = "#F9E79F" # Relation properties

    # Add main nodes & properties
    for node_label, node_info in combined_nodes.items():
        description = node_info.get("node_description", "") or ""
        net.add_node(node_label,
                     label=node_label,
                     title=description,
                     color=node_color,
                     shape="ellipse")

        props = node_info.get("node_properties", []) or []
        for i, prop in enumerate(props):
            prop_node = f"{node_label}::prop::{i}"
            prop_label = f"{prop.get('name', '?')} ({prop.get('type', '?')})"
            prop_title = f"{'Optional' if prop.get('optional') else 'Required'} - {prop.get('description', '') or ''}"
            net.add_node(prop_node, label=prop_label, title=prop_title, color=node_prop_color, shape="box")
            # No arrows for property edges
            net.add_edge(node_label, prop_node, arrows={'to': {'enabled': False}, 'from': {'enabled': False}})

    # Add relations
    for rel_label, rel_info in combined_relations:
        rel_title = rel_info.get("relation_description", "") or ""
        from_node = rel_info.get("from_node")
        to_node = rel_info.get("to_node")

        # Guard: nodes must exist
        if from_node not in combined_nodes:
            logging.warning(f"Relation '{rel_label}' from '{from_node}' not found among nodes. Skipping.")
            continue
        if to_node not in combined_nodes:
            logging.warning(f"Relation '{rel_label}' to '{to_node}' not found among nodes. Skipping.")
            continue

        rel_node = f"rel::{rel_label}::{from_node}->{to_node}"
        net.add_node(rel_node, label=rel_label, title=rel_title, color=relation_color, shape="ellipse")

        # Connect relation node between endpoints (directional chain)
        net.add_edge(from_node, rel_node)    # implicit arrows to rel_node disabled by default
        net.add_edge(rel_node, to_node)      # global arrows show direction to 'to_node'

        # Relation properties (if any)
        rel_props = rel_info.get("relation_properties", []) or []
        for i, prop in enumerate(rel_props):
            prop_node = f"{rel_node}::prop::{i}"
            prop_label = f"{prop.get('name','?')} ({prop.get('type','?')})"
            prop_title = f"{'Optional' if prop.get('optional') else 'Required'} - {prop.get('description', '') or ''}"
            net.add_node(prop_node, label=prop_label, title=prop_title, color=relation_prop_color, shape="box")
            net.add_edge(rel_node, prop_node, arrows={'to': {'enabled': False}, 'from': {'enabled': False}})

    # Configure graph options
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

    # Write HTML
    net.write_html(output_file, open_browser=False)

    # Inject legend safely
    legend_html = """
    <div style='position:fixed;bottom:10px;left:10px;background:#fff;padding:10px;border:1px solid #ccc;font-family:Arial;font-size:14px;z-index:999;'>
      <b>Legend:</b><br>
      <span style='background-color:#AED6F1;padding:4px;'> Main Node </span><br>
      <span style='background-color:#F5B7B1;padding:4px;'> Relation Node </span><br>
      <span style='background-color:#A9DFBF;padding:4px;'> Node Property </span><br>
      <span style='background-color:#F9E79F;padding:4px;'> Relation Property </span>
    </div>
    """

    try:
        with open(output_file, "r+", encoding="utf-8") as f:
            content = f.read()
            if "</body>" in content:
                content = content.replace("</body>", legend_html + "\n</body>")
                f.seek(0)
                f.write(content)
                f.truncate()
            else:
                logging.warning("Could not find '</body>' to inject legend; leaving file unchanged.")
    except Exception as e:
        logging.error(f"Failed to inject legend into '{output_file}': {e}")

