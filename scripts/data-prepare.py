from openai import OpenAI
from dotenv import load_dotenv
import openai
import json
import os
import argparse

def generate_adverse_event_report(prompt, model="gpt-4-1106-preview", max_tokens=3500, temperature=1, top_p=1, frequency_penalty=0, presence_penalty=0):
    """
    Generate Adverse Event Reports for the Drug using the OpenAI API.

    Args:
        prompt (str): Prompt for the OpenAI API

    Returns:
        response (str): Response from the OpenAI API
    """
    # OpenAI Client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # OpenAI Completion API
    response = client.chat.completions.create(model=model,
                                            messages=[
                                                {"role": "system", "content": "Act as an expert Analyst with 20+ years of experience in Pharma and Healthcare industry. You have to generate Adverse Event Reports in properly formatted JSON"},
                                                {"role": "user", "content": prompt}],
                                            response_format={ "type": "json_object" },
                                            temperature=temperature,
                                            max_tokens=max_tokens,
                                            top_p=top_p,
                                            frequency_penalty=frequency_penalty,
                                            presence_penalty=presence_penalty
                                        )

    return response.choices[0].message.content.strip()

def create_prompt(drug_name, drug_report):
    """
    Create a prompt for the OpenAI API using the drug name and the drug report.

    Args:
        drug_name (str): Name of the drug
        drug_report (str): Information about the drug
    
    Returns:
        prompt (str): Prompt for the OpenAI API
    """
    return f"""Sample Adverse Event reports:
    [
        {{
            "input": "Nicole Moore
                    moore123nicole@hotmail.com
                    32 McMurray Court, Columbia, SC 41250
                    9840105113, United States 
                    
                    Relationship to XYZ Pharma Inc.: Patient or Caregiver
                    Reason for contacting: Adverse Event
                    
                    Message: Yes, I have been taking Mylan’s brand of Metroprolol for two years now and with no problem. I recently had my prescription refilled with the same Mylan Metoprolol and I’m having a hard time sleeping at night along with running nose. Did you possibly change something with the pill...possibly different fillers? The pharmacist at CVS didn’t have any information for me. Thank you, Nicole Moore", 
            "output": {{
                "drug_name":"Metroprolol",
                "adverse_events": ["hard time sleeping at night", "running nose"]
            }}
        }},
        {{
            "input": "Jack Ryan,
                    jack3rayan@gmail.com
                    120 Erwin RD, Canonsburg, PA 21391,
                    2133681441, United States
                    
                    Relationship to XYZ Pharma Inc.: Patient
                    Reason for contacting: Defective Product
                    
                    Message: I recently purchased a Wixela inhub 250/50 at my local CVS pharmacy and the inhaler appears to be defective. When I try and activate it, the yellow knob only goes down halfway. I just removed this one from the wrapper so I know it's not empty. The pharmacy wouldn't exchange it so I am contacting you to get a replacement. Thank you for your time and consideration in this matter",
            "output": {{
                "drug_name":"Wixela inhub 250/50",
                "adverse_events": ["defective inhaler"]
            }}
        }},
    ]

    Now create Adverse Event Reports in a similar way for the Drug - {drug_name}. 

    You have more information about the drug's use and its side effects below:
    {drug_report}

    Generate 15 different reports each with different side effects. Mention one or two side effects in each report at max. You have to prepare data for Entity Extraction of 2 entities: "drug_name" and "adverse_events" only.
    Followng the following format for the final output:

    [
        {{
        "input":"## Generated Report Here",
        "output": {{ "drug_name":"## Name of Drug", "adverse_events": ["side effect 1", "side effect 2"] }}
        }},
        {{
        "input":"## Generated Report Here",
        "output": {{ "drug_name":"## Name of Drug", "adverse_events": ["side effect 1", "side effect 2"] }}
        }},
    ]
    """

def create_dataset(folder_path):
    """
    Create a dataset of Adverse Event Reports for the Drugs using the OpenAI Chat Completions API.

    Args:
        folder_path (str): Path to the folder containing the Drug Information files

    Returns:
        None
    """
    # Iterate through the files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)

            # Read the contents of the file
            with open(file_path, 'r') as file:
                file_contents = file.read()

            # Get the name of the drug from the filename
            drug_name = filename.split('.')[0]
            # Get the information about the drug from the file contents
            drug_report = file_contents

            # Create a dynamic prompt
            prompt = create_prompt(drug_name, drug_report)

            # Generate Adverse Event Reports for the Drug
            reports = generate_adverse_event_report(prompt)

            # Convert the string response to a Python Dict object
            output_list = json.loads(reports)

            # Save the generated data as a JSON file
            with open(f"../data/entity_extraction_reports/{drug_name}.txt", 'w') as text_file:
                text_file.write(output_list)


if __name__ == '__main__':
    # Load the .env file with the API key
    load_dotenv()

    # Parse the arguments
    parser = argparse.ArgumentParser(description="Data Preparation Script")
    parser.add_argument('--folder-path', type=str, default='../data/raw_drug_info/', help="Path to the folder containing the raw Drug Information files scraped form web")
    args = parser.parse_args()

    create_dataset(args.folder_path)