import os
import json

# Replace 'your_folder_path' with the actual path to your folder containing text files
folder_path = './data-preparation/entity_extraction_reports/'

# Initialize an empty list to store JSON objects
json_objects = []

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    # Check if the file is a text file
    if filename.endswith('.txt'):
        print(f"[INFO] Processing file: {filename}")

        # Read the file and load the JSON object
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, 'r') as file:
            try:
                json_object = json.loads(file.read())
                json_objects.extend(json_object)
            except json.JSONDecodeError:
                print(f"Error reading file: {file_path}")

# Combine all JSON objects into one JSON array
combined_json = json.dumps(json_objects, indent=4)

# Save the combined JSON to a file (optional)
with open('./data-preparation/entity-extraction-data.json', 'w') as output_file:
    output_file.write(combined_json)
