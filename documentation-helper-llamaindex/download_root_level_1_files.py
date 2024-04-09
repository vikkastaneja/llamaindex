import random

import requests
from bs4 import BeautifulSoup
import os


url = "https://docs.llamaindex.ai/en/stable/"
reqs = requests.get(url)
soup = BeautifulSoup(reqs.text, "html.parser")
output_dir = "./llamaindex-docs/"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

urls = []
for link in soup.find_all("a"):
    href = link.get("href")
    # print(link.get('href'))
    if not href.startswith("http"):
        print(url + href)
        file_response = requests.get(url + href)

        # Write it to a file
        file_name = os.path.join(
            output_dir, os.path.basename(href.replace("/", "") + "1.html")
        )
        with open(file_name, "w", encoding="utf-8") as file:
            file.write(file_response.text)
