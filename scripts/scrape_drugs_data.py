## This script scrapes the drugs.com website for the given drug name and saves the extracted text to a file.

import requests
from bs4 import BeautifulSoup
import html2text
from multiprocessing import Pool
import os

def extract_section_by_id(soup, section_id):
    """
    Extracts the section with the given ID from the HTML content. 

    Args:
        soup (BeautifulSoup): BeautifulSoup object containing the HTML content
        section_id (str): ID of the section to be extracted

    Returns:
        str: HTML content of the section
    """
    
    # Find the section with the given ID
    section = soup.find('h2', id=section_id)
    
    # If section is not found, return an error message
    if not section:
        return f"Section with ID '{section_id}' not found."

    # Extract the content of the section
    content = []
    for sibling in section.next_siblings:
        # Stop at the next section
        if sibling.name == 'h2':
            break
        # Extract the text if the sibling is a paragraph or list
        if sibling.name in ['p', 'ul', 'ol', 'div']:
            content.append(str(sibling))
    
    return ' '.join(content)

def scrape_website(args):
    """
    Scrapes the drugs.com website for the given drug name and saves the extracted text to a file.

    Args:
        args (tuple): Tuple containing the URL and drug name

    Returns:
        None
    """
    # Unpack the arguments
    url, drug_name = args
    print(f"[INFO] Processing: {drug_name}")
    
    # Fetch the HTML content
    try:
        response = requests.get(url)
        # If the request fails, return an error message
        if response.status_code != 200:
            print(f"[ERROR] Failed to fetch {url} with status code {response.status_code}")
            return
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
    except Exception as e:
        print(f"[ERROR] Exception while fetching {url}: {e}")
        return

    # Initialize html2text
    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_images = True

    # Section IDs to extract
    section_ids = ["uses", "side-effects"]

    # Extract and convert HTML to text for each section
    file_content = ''
    # Extract the sections one by one
    for section_id in section_ids:
        section_html = extract_section_by_id(soup, section_id)
        section_text = h.handle(section_html)
        file_content += section_text + "\n\n"

    # Write the extracted text to a file
    file_name = f"data/raw_drug_info/{drug_name}.txt"
    with open(file_name, 'w', encoding='utf-8') as file:
        file.write(file_content)

    print(f"[INFO] Processed {drug_name} Successfully")

if __name__ == "__main__":
    ## List of drugs to be scrape from drugs.com
    DRUG_LIST = ["abilify", "infliximab", "rituximab", "etanercept",
                "Humira", "Enbrel", "Remicade", "Rituxan",
                "Nexium", "Prevacid", "Prilosec", "Protonix",
                "Crestor", "Lipitor", "Zocor", "Vytorin", 
                "Victoza", "Byetta", "Januvia", "Onglyza",
                "Advair", "Symbicort", "Spiriva", "Singulair",
                "Cialis", "Viagra", "Levitra", "Staxyn",
                "AndroGel", "Prezista", "Doxycycline", "Cymbalta",
                "Neupogen", "Epogen", "Aranesp", "Neulasta",
                "Lunesta", "Ambien", "Provigil", "Nuvigil",
                "Metoprolol", "Lisinopril", "Amlodipine", "Atorvastatin",
                "Zoloft", "Lexapro", "Prozac", "Celexa",
                "Complera", "Atripla"]
    
    if not os.path.exists("data/raw_drug_info/"):
        os.makedirs("data/raw_drug_info/")

    # Prepare a list of tuples for the Pool
    tasks = [(f"https://www.drugs.com/{drug}.html", drug) for drug in DRUG_LIST]

    # Number of processes
    num_processes = 5  # Adjust this based on your machine's capability

    # Create a Pool of processes and map the function to the tasks
    with Pool(num_processes) as pool:
        pool.map(scrape_website, tasks)
