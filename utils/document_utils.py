import os
import glob


def determine_document_type(file_path):
    """Determine document type based on folder structure"""
    if "medical_reference" in file_path:
        return "medical_reference"
    elif "user_stories" in file_path:
        return "user_story"
    elif "community_resources" in file_path:
        return "community_resource"
    elif "clinical_guidelines" in file_path:
        return "clinical_guideline"
    else:
        return "general"


def get_data_directory():
    """Get the correct data directory path for different environments"""
    # Try different possible data directory locations
    possible_paths = [
        "./data",  # Local development
        "/app/data",  # Docker container
        "data",  # Alternative local
        os.path.join(
            os.getcwd(), "data"
        ),  # Absolute path from current working directory
    ]

    for path in possible_paths:
        if os.path.exists(path):
            # Check if it actually contains markdown files
            md_files = glob.glob(f"{path}/**/*.md", recursive=True)
            if md_files:
                return path, md_files

    return None, []
