import requests
from bs4 import BeautifulSoup
import html2text
from multiprocessing import Pool
from config import DRUG_LIST

def extract_section_by_id(soup, section_id):
    section = soup.find('h2', id=section_id)
    if not section:
        return f"Section with ID '{section_id}' not found."

    content = []
    for sibling in section.next_siblings:
        if sibling.name == 'h2':
            break
        if sibling.name in ['p', 'ul', 'ol', 'div']:
            content.append(str(sibling))
    
    return ' '.join(content)

def scrape_website(args):
    url, drug_name = args
    print(f"[INFO] Processing: {drug_name}")
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"[ERROR] Failed to fetch {url} with status code {response.status_code}")
            return

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
    for section_id in section_ids:
        section_html = extract_section_by_id(soup, section_id)
        section_text = h.handle(section_html)
        file_content += section_text + "\n\n"

    # Write the extracted text to a file
    file_name = f"./data/{drug_name}.txt"
    with open(file_name, 'w', encoding='utf-8') as file:
        file.write(file_content)

    print(f"[INFO] Processed {drug_name} Successfully")

if __name__ == "__main__":
    # Prepare a list of tuples for the Pool
    tasks = [(f"https://www.drugs.com/{drug}.html", drug) for drug in DRUG_LIST]

    # Number of processes
    num_processes = 5  # Adjust this based on your machine's capability

    # Create a Pool of processes and map the function to the tasks
    with Pool(num_processes) as pool:
        pool.map(scrape_website, tasks)
