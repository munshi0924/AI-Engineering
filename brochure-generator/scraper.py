from bs4 import BeautifulSoup
import requests


#Standard headers to fetch a website like normal browser window 
headers = {
    "User-Agent": "Mozilla/5.0 (windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

def fetch_website_contents(url):
    """
    Return the tittle and contents of the website at the given url;
    trucate to 3000 characters as sensible limit
    """
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    tittle = soup.tittle.string if soup.tittle else "No tittle found"
    if soup.body:
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        text = soup.body.get_text(separator="\n", strip=True)
    else:
        text = ""
    return(tittle+ "\n\n"+ text)[:3000]

def fetch_website_links(url):
    """
    Retuns the links on the give4n url
    """
    response= requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    links = [link.get("herf") for link in soup.find_all("a")]
    return [link for link in links if link]