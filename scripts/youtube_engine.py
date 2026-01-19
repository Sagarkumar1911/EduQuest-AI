from googleapiclient.discovery import build

def get_relevant_video(query: str, api_key: str):  # <--- This is the missing function
    """
    Searches YouTube for the best educational video.
    """
    if not api_key or "PASTE" in api_key:
        print("⚠️ YouTube Engine Error: Invalid or missing API Key.")
        return None

    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        
        search_query = f"{query} education animation"
        
        request = youtube.search().list(
            q=search_query,
            part='snippet',
            type='video',
            maxResults=1,
            relevanceLanguage='en',
            order='relevance'
        )
        response = request.execute()

        if response.get('items'):
            item = response['items'][0]
            video_id = item['id']['videoId']
            
            return {
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "embed_url": f"https://www.youtube.com/embed/{video_id}",
                "title": item['snippet']['title'],
                "thumbnail": item['snippet']['thumbnails']['high']['url']
            }
            
    except Exception as e:
        print(f"❌ YouTube API Error: {e}")
        return None
    
    return None