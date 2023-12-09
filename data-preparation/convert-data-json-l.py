import json
from tqdm import tqdm

# Replace 'input_file.json' with the path to your JSON file containing 700 items
input_file = './data-preparation/entity-extraction-data.json'

# Replace 'output_file.jsonl' with the desired path for your JSON-L file
output_file = './data-preparation/entity-extraction-data.jsonl'

# Load the JSON data from the input file
with open(input_file, 'r') as f:
    data = json.load(f)

# Open the output JSON-L file for writing
with open(output_file, 'w') as f:
    # Iterate through the JSON array and write each object as a JSON-L line
     for item in tqdm(data, desc="Converting to JSON-L"):
        json_line = json.dumps(item, ensure_ascii=False)
        f.write(json_line + '\n')

print(f'{len(data)} JSON objects converted to JSON-L format and saved to {output_file}')
