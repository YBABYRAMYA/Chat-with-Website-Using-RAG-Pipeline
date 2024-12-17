import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to scrape and clean website content
def scrape_website(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Extract only p and header tags
        main_content = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        text = "\n".join([element.get_text(strip=True) for element in main_content])
        print(f"Scraped content from {url}")
        return text
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""

# Function to filter relevant research content
def filter_research_content(content, keywords):
    lines = content.split("\n")
    filtered_lines = [line for line in lines if any(keyword.lower() in line.lower() for keyword in keywords)]
    return "\n".join(filtered_lines)

# Function to preprocess content into chunks
def preprocess_content(content, max_words=100):
    paragraphs = [p.strip() for p in content.split("\n") if p.strip()]
    chunks = []
    for paragraph in paragraphs:
        words = paragraph.split()
        for i in range(0, len(words), max_words):
            chunks.append(" ".join(words[i:i + max_words]))
    return chunks

# Function to find the most relevant content
def find_most_relevant(query, content_chunks, top_k=3):
    chunk_embeddings = model.encode(content_chunks, convert_to_tensor=True)
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, chunk_embeddings).squeeze(0)
    top_indices = scores.argsort(descending=True)[:top_k]
    relevant_chunks = [content_chunks[idx] for idx in top_indices]
    return "\n".join(relevant_chunks)

# Function to generate a response
def generate_response(query, content_chunks):
    relevant_content = find_most_relevant(query, content_chunks)
    if not relevant_content.strip():
        return "No relevant data found for your query."
    return f"Question: {query}\n\nBased on the available content, here is the response:\n\n{relevant_content}"

# Main function to handle dynamic input
def main():
    print("Enter the URLs you want to scrape (comma-separated):")
    urls = input().split(",")
    urls = [url.strip() for url in urls if url.strip()]

    print("\nEnter your query:")
    query = input().strip()

    keywords = ["research", "focus", "mission", "innovation", "academics", "knowledge"]

    combined_chunks = []
    for url in urls:
        content = scrape_website(url)
        if content:
            filtered_content = filter_research_content(content, keywords)
            chunks = preprocess_content(filtered_content)
            combined_chunks.extend(chunks)

    if combined_chunks:
        response = generate_response(query, combined_chunks)
        print("\nResponse:\n", response)
    else:
        print("\nNo content could be processed from the provided URLs.")

if __name__ == "__main__":
    main()
