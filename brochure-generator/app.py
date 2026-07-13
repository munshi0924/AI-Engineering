import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from scraper import fetch_website_links, fetch_website_contents

#Load the API from the .env file
load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')

# Initialize the OpenAI Client
openai = OpenAI()

# Using the standard model from the notebook setup
MODEL = 'gpt-4o-mini'

# 1. System Prompt for filtering links
link_system_prompt = """
You are provided with a list of links found on a webpage.
You are able to decide which of the links would be most relevant to include in a brochure about the company,
such as links to an About page, or a Company page, or Careers/Jobs pages.
You should respond in JSON as in this example:

{
    "links": [
        {"type": "about page", "url": "https://full.url/goes/here/about"},
        {"type": "careers page", "url": "https://another.full.url/careers"}
    ]
}
"""

def get_links_user_prompt(url):
    user_prompt = f"""
    Here is the list of links on the website {url} -
Please decide which of these are relevant web links for a brochure about the company, 
respond with the full https URL in JSON format.
Do not include Terms of Service, Privacy, email links.

Links (some might be relative links):

"""
    links = fetch_website_links(url)
    user_prompt += "\n".join(links)
    return user_prompt

def select_relevant_links(url):
    print(f"🤖 AI is reading website links for {url}...")
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": link_system_prompt},
            {"role": "user", "content": get_links_user_prompt(url)}
        ],
        response_format={"type": "json_object"}
    )
    result = response.choices[0].message.content
    return json.loads(result)
    
def fetch_page_and_all_relevant_links(url):
    contents = fetch_website_contents(url)
    relevant_links = select_relevant_links(url)
    result = f"## Landing Page:\n\n{contents}\n## Relevant Links:\n"
    
    # We will loop through the top 3 links found by AI to keep it quick
    for link in relevant_links['links'][:3]:
        print(f"📖 Scraping sub-page: {link['type']} -> {link['url']}")
        result += f"\n\n### Link: {link['type']}\n"
        result += fetch_website_contents(link["url"])
    return result

# 2. System Prompt for writing the final brochure
brochure_system_prompt = """
You are an assistant that analyzes the contents of several relevant pages from a company website
and creates a short brochure about the company for prospective customers, investors and recruits.
Respond in markdown without code blocks.
Include details of company culture, customers and careers/jobs if you have the information.
"""

def get_brochure_user_prompt(company_name, url):
    user_prompt = f"""
You are looking at a company called: {company_name}
Here are the contents of its landing page and other relevant pages;
use this information to build a short brochure of the company in markdown without code blocks.\n\n
"""
    user_prompt += fetch_page_and_all_relevant_links(url)
    return user_prompt[:5000] # Truncate to 5,000 characters maximum

def stream_brochure(company_name, url):
    print(f"\n🚀 Starting brochure compilation for {company_name}...")
    user_prompt = get_brochure_user_prompt(company_name, url)
    
    print("\n✍️ Generating brochure text stream:\n" + "="*40 + "\n")
    
    stream = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": brochure_system_prompt},
            {"role": "user", "content": user_prompt}
          ],
        stream=True
    )    
    
    # Print out chunks as they stream live from the API
    for chunk in stream:
        content = chunk.choices[0].delta.content or ''
        print(content, end="", flush=True)
        
    print("\n\n" + "="*40 + "\n✅ Brochure Complete!")

# --- EXECUTE THE PROGRAM ---
if __name__ == "__main__":
    stream_brochure("HuggingFace", "https://huggingface.co")