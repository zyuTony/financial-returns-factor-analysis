from bs4 import BeautifulSoup

# Read the HTML content from a file
with open('content.html', 'r', encoding='utf-8') as file:
    html_content = file.read()

# Parse the HTML
soup = BeautifulSoup(html_content, 'html.parser')

# Find all elements containing the video views and their corresponding titles
views_elements = soup.find_all('strong', {'data-e2e': 'video-views'})
title_elements = soup.find_all('a', {'class': 'css-1wrhn5c-AMetaCaptionLine eih2qak0'})

# Extract and print the views count and titles for each element
if views_elements and title_elements:
    for idx, (view, title) in enumerate(zip(views_elements, title_elements), start=1):
        views_count = view.text
        video_title = title.get('title')
        print(f"Video {idx} Views: {views_count}, Title: {video_title}")
else:
    print("No video views or titles elements found.")
