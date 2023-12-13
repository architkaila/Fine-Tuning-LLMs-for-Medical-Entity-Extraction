import os
import json
import random

def genrate_train_test_data(folder_path = 'data/entity_extraction_reports/'):
    """
    Generate train and test data from the JSON files in the given folder path.

    Args:
        folder_path (str): Path to the folder containing JSON files.
    
    Returns:
        None
    """

    # Create the folder if it doesn't exist
    if not os.path.exists("data/entity_extraction"):
        os.makedirs("data/entity_extraction")
    
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
                    
                    # Convert the 'output' field from a JSON object to a JSON string
                    for item in json_object:
                        item['output'] = json.dumps(item['output']) 
                    
                    json_objects.extend(json_object)
                except json.JSONDecodeError:
                    print(f"Error reading file: {file_path}")

    # Shuffle the JSON objects
    random.shuffle(json_objects)

    # Split the data into train and test
    train_data = json_objects[:700]  # First 700 objects for training
    test_data = json_objects[700:]   # Last 59 objects for testing

    # Write the train data to a file
    with open('data/entity_extraction/entity-extraction-train-data.json', 'w') as file:
        json.dump(train_data, file, indent=4)

    # Write the test data to a file
    with open('data/entity_extraction/entity-extraction-test-data.json', 'w') as file:
        json.dump(test_data, file, indent=4)

if __name__ == '__main__':
    # Replace 'folder_path' with the actual path to your folder containing text files
    folder_path = 'data/entity_extraction_reports/'

    # Call the function
    genrate_train_test_data(folder_path)