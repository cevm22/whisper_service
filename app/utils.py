import shutil

def move_and_rename_file(source_path, destination_directory, new_name):
    # Construct the destination path with the new name
    destination_path = f"{destination_directory}/{new_name}"
    
    # Move and rename the file
    shutil.move(source_path, destination_path)

