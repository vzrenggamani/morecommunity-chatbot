import os
import glob


def should_rebuild_vectorstore(vector_store_path, data_path="./data"):
    """Check if vector store needs rebuilding based on file timestamps"""
    if not os.path.exists(vector_store_path):
        return True, "Vector store tidak ditemukan"

    # Get vector store timestamp
    try:
        vectorstore_timestamp = os.path.getmtime(vector_store_path)
    except:
        return True, "Tidak dapat membaca timestamp vector store"

    # Check all markdown files
    md_files = glob.glob(f"{data_path}/**/*.md", recursive=True)
    if not md_files:
        return False, "Tidak ada file markdown ditemukan"

    # Check if any document is newer than vector store
    for file_path in md_files:
        try:
            file_timestamp = os.path.getmtime(file_path)
            if file_timestamp > vectorstore_timestamp:
                return True, f"Dokumen {os.path.basename(file_path)} telah diperbarui"
        except:
            continue

    return False, "Vector store masih up-to-date"
