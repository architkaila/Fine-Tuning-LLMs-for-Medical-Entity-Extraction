import json

def parse_entities(record, key):
    """
    Parse entities from a record in the data.
    
    Args:
    - record: A dictionary representing a single record in the data.
    - key: The key to extract data from ('output' or 'prediction').

    Returns:
    - A set containing the extracted entities.
    """
    # Convert the string into a dictionary
    entities = json.loads(record[key])
    
    # Initialize a set to store the entities
    flattened_entities = set()
    
    # Extract the entities
    for value in entities.values():
        # Check if item is a list of adverse events
        if isinstance(value, list):
            flattened_entities.update(value)
        # Parse drug names
        else:
            flattened_entities.add(value)
    
    return flattened_entities

def calculate_precision_recall(data):
    """
    Calculate precision and recall from the data.

    Args:
    - data: A list of dictionaries, each containing 'output' and 'prediction'.

    Returns:
    - precision: The precision of the predictions.
    - recall: The recall of the predictions.
    """
    # Initialize variables
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # parse all the samples in the test dataset
    for record in data:
        # Extract ground truths
        gt_entities = parse_entities(record, 'output')
        # Extract predictions
        pred_entities = parse_entities(record, 'prediction')
    
        # Calculate TP, FP, FN for each sample in test data
        true_positives += len(gt_entities & pred_entities)
        false_positives += len(pred_entities - gt_entities)
        false_negatives += len(gt_entities - pred_entities)
    
    # Calculate Precision
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    # Calculate Recall
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0

    return precision, recall

if __name__ == '__main__':

    prediction_files = [
                'data/predictions-llama2-adapter.json',   # Llama-2 Adapter
                'data/predictions-stablelm-adapter.json', # Stable-LM Adapter
                'data/predictions-llama2-lora.json',      # Llama-2 Lora
                'data/predictions-stablelm-lora.json',    # Stable-LM Lora
                ]
    
    for filename in prediction_files:
        # Get model name and tune type
        file_components = filename.split('-')
        
        # Load the predcitions JSON data
        with open(filename, 'r') as file:
            data = json.load(file)

        precision, recall = calculate_precision_recall(data)
        print(f"[INFO] {file_components[1]}-{file_components[2].split('.')[0]} ----> Precision: {round(precision,3)} Recall: {round(recall,3)}")