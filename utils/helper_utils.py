import os
import shutil

def prepare_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        if os.listdir(folder_path):  # Check if folder is not empty
            # Folder is not empty. Deleting contents...
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # Delete file or symlink
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # Delete folder
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
            # Folder is now empty.
        else:
            # Folder is already empty.
            pass