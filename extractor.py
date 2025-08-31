from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import spacy
import time

nlp = spacy.load("en_core_web_lg")

def fetch_article_text(url: str) -> tuple[str, str]:
    """Use Selenium to fetch article title and body from Times of India pages."""
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    try:
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()), options=options
        )
        driver.get(url)
        time.sleep(3)
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # 1) Extract title using the H1 tag with class HNMDR
        title_tag = soup.select_one("h1.HNMDR")
        title = title_tag.get_text(strip=True) if title_tag else "Title not found"

        # 2) Extract content from the primary article body div
        # The provided HTML structure has a main content container with a specific class.
        content_div = soup.find("div", class_="JuyWl CpKyX")
        paragraphs = []
        
        if content_div:
            # Find and extract text from direct children that are not specific ad/widget containers
            for element in content_div.find_all(recursive=False):
                # We'll collect text from direct text nodes and specific tags
                if element.name in ["div", "span"] and "class" not in element.attrs:
                    paragraphs.append(element.get_text(strip=True))
                elif element.name == "div" and "cdatainfo" in element.get("class", []):
                     paragraphs.append(element.get_text(strip=True))
                elif element.name == "div" and element.get("class") == ["vSlIC"]:
                    # This div seems to contain the main article text
                    article_body = element.find("div", class_="art_synopsis")
                    if article_body:
                        paragraphs.append(article_body.get_text(strip=True))
                    
                    # Also get the text from the main body div, excluding known ads
                    main_body_text_div = element.find("div", class_="_s30J clearfix")
                    if main_body_text_div:
                        for text_element in main_body_text_div.find_all(lambda tag: tag.name in ["p", "div", "span"]):
                            # This is a very specific, and possibly brittle, way to get the text.
                            # It's better to get the text of the entire div and then clean it.
                            full_text_content = main_body_text_div.get_text(separator="\n", strip=True)
                            paragraphs.append(full_text_content)
                            break # We only need to do this once.

        if paragraphs:
            # Join the paragraphs and clean up unwanted text
            content = "\n\n".join(paragraphs)
        else:
            content = "Content not found"

    except Exception as e:
        return f"❌ Error: Could not fetch or parse article from {url}. {e}", "Content not found"
    finally:
        if 'driver' in locals():
            driver.quit()

    return title, content

def extract_named_entities(text: str) -> list[tuple[str, str]]:
    """Extract named entities using spaCy."""
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

if __name__ == "__main__":
    sample_url = "https://timesofindia.indiatimes.com/entertainment/tamil/movies/news/coolie-star-rajinikanths-humble-flight-moment-goes-viral-actors-simplicity-steals-hearts/articleshow/123163168.cms"
    
    article_title, article_content = fetch_article_text(sample_url)

    print("\n--- Article Title ---\n", article_title)
    print("\n--- Article Content ---\n", article_content)

    print("\n--- Named Entities ---")
    if article_content != "Content not found":
        for ent, label in extract_named_entities(article_content):
            print(f"{ent}: {label}")
    else:
        print("Cannot extract entities without content.")