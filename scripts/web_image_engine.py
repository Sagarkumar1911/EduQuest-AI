from duckduckgo_search import DDGS

def get_google_images(query):
    """
    Searches for images using DuckDuckGo (acting as a proxy for web search)
    to find relevant diagrams for the topic.
    """
    print(f"🌍 Searching Web Images for: {query}")
    try:
        results = DDGS().images(
            keywords=f"{query} scientific diagram labeled",
            region="wt-wt",
            safesearch="on",
            max_results=3
        )
        
        images = []
        for res in results:
            images.append({
                "path": res['image'],
                "description": res['title']
            })
            
        return images
    except Exception as e:
        print(f"❌ Web Image Search Error: {e}")
        return []