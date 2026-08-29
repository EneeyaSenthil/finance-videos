import urllib.request
import xml.etree.ElementTree as ET

def fetch_real_us_finance_headlines():
    # Official Google News RSS feed locked strictly to US English and US region
    url = "https://news.google.com/rss/search?q=US+personal+finance+interest+rates&hl=en-US&gl=US&ceid=US:en"
    
    print("Fetching live data from US feeds...")
    
    try:
        # Request the raw XML data
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        response = urllib.request.urlopen(req)
        xml_data = response.read()
        
        # Parse the XML structure
        root = ET.fromstring(xml_data)
        
        headlines = []
        # Find all news items in the RSS feed
        for item in root.findall('.//item')[:15]: # Grab top 15 current items
            title = item.find('title')
            if title is not None and title.text:
                headlines.append(title.text)
                
        # Save straight to a local text file for your screen reader
        file_path = "live_us_finance.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("--- LIVE US FINANCE TRENDING HEADLINES ---\n\n")
            for i, h in enumerate(headlines, 1):
                f.write(f"{i}. {h}\n")
                
        print(f"Success! Saved real headlines to {file_path}. Open it with Notepad to read.")
        
    except Exception as e:
        print(f"Error fetching data: {e}")

if __name__ == "__main__":
    fetch_real_us_finance_headlines()