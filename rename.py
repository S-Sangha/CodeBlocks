import os
import shutil


def change_name(name):
    folder_path = f'/home/abdul/IC_Hack/CodeBlocks/dataset/{name}'
    prefix = f'{name}_'  # Set your desired prefix here

    # List all files in the folder
    files = os.listdir(folder_path)

    # Sort files to ensure they are processed in order
    files.sort()

    # Iterate through the files and rename them successively
    for i, file_name in enumerate(files):
        # Construct the new file name with the desired prefix and index
        new_name = f'{prefix}{i+1}.jpg'
        
        # Full paths for the old and new names
        old_path = os.path.join(folder_path, file_name)
        new_path = os.path.join(folder_path, new_name)
        
        # Rename the file
        os.rename(old_path, new_path)

    print('Files renamed successfully.')




def organize_into_folders(folder_path):
    # List all elements in the folder
    elements = os.listdir(folder_path)

    for i in range(len(elements)):
        # Construct the new folder paths
        name = f"{elements[i]}_{i}".replace(".jpg","")
        new_folder_path = os.path.join(folder_path, name)

        # Create a new folder for each element
        os.makedirs(new_folder_path, exist_ok=True)

        # Construct the source and destination paths
        source_path = os.path.join(folder_path, elements[i])
        destination_path = os.path.join(new_folder_path, f"{elements[i]}")

        # Move the element into its own folder
        shutil.move(source_path, destination_path)


def add_label(directory,label):
    for root, dirs, files in os.walk(directory):
        for folder in dirs:
            hello_file_path = os.path.join(root, folder, "label.txt")
            with open(hello_file_path, 'w') as hello_file:
                hello_file.write(label)



if __name__ == "__main__":
    # Set the path to your folder
    name = "2"
    folder_path = f'/home/abdul/IC_Hack/CodeBlocks/dataset/{name}/'

    # Call the function to organize elements into folders
    change_name(name)
    organize_into_folders(folder_path)
    add_label(folder_path,name)

    print('Elements organized into folders successfully.')
