import os
import re

def search_phrase(text, phrase):
    """Search phrase in text

    Args:
        text (str): text string
        phrase (str): phase string

    Returns:
        bool: True if phase in text else False
    """
    # Compile a regular expression pattern for the specific phrase
    pattern = re.compile(re.escape(phrase))
    # Search for the pattern in the text
    match = pattern.search(text)
    # Check if a match was found
    if match:
        return True
    else:
        return False

def set_neo4j_import_folder(config_file_path, import_folder_path):
    """Set neo4j import folder

    Args:
        config_file_path (str): location for Neo4j config file
        import_folder_path (str): location for user provided import folder
    """
    # Ensure the import directory exists
    if not os.path.exists(import_folder_path):
        os.makedirs(import_folder_path)

    with open(config_file_path, 'r') as file:
        lines = file.readlines()

    with open(config_file_path, 'w') as file:
        found_dbms = False
        # found_server = False
        for line in lines:
            if line.startswith('dbms.directories.import='):
                file.write(f'dbms.directories.import={import_folder_path}\n')
                found_dbms = True
            # if line.startswith('server.directories.import='):
            #     file.write(f'server.directories.import={import_folder_path}\n')
            #     found_server = True
            else:
                file.write(line)
        if not found_dbms:
            file.write('\n')
            file.write(f'dbms.directories.import={import_folder_path}\n')
        # if not found_server:
        #     file.write('\n')
        #     file.write(f'server.directories.import={import_folder_path}\n')
