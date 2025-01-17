# src/docs_scraper/flame_docs.py

import requests
from pathlib import Path
from typing import List, Dict
import json
from datetime import datetime
from bs4 import BeautifulSoup
import re

class FlameScraper:
    """Scrapes and processes FlutterFlame documentation"""
    
    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.raw_dir = self.output_dir / "raw"
        self.raw_dir.mkdir(exist_ok=True)
        
        # Version tracking
        self.flame_version = self._get_flame_version()
        
        # Track processed documents
        self.processed_docs = []
    
    def test_connection(self) -> bool:
        """Test connections to all documentation sources"""
        sources = [
            "https://pub.dev/documentation/flame/latest/",
            "https://docs.flame-engine.org/latest/",
            "https://api.github.com/repos/flame-engine/flame/contents/doc"
        ]
        
        results = []
        for url in sources:
            try:
                response = requests.get(url)
                results.append(response.status_code == 200)
                print(f"Connection to {url}: {'Success' if response.status_code == 200 else 'Failed'}")
            except Exception as e:
                print(f"Failed to connect to {url}: {e}")
                results.append(False)
        
        return all(results)
    
    def _get_flame_version(self) -> str:
        """Get current Flame version from pub.dev"""
        try:
            response = requests.get("https://pub.dev/api/packages/flame")
            if response.status_code == 200:
                data = response.json()
                return data["latest"]["version"]
        except Exception as e:
            print(f"Failed to get Flame version: {e}")
        return "unknown"
    
    def fetch_github_docs(self) -> List[Dict[str, str]]:
        """Fetch documentation from GitHub repository"""
        print("Fetching GitHub documentation...")
        docs = []
        
        try:
            response = requests.get(
                "https://api.github.com/repos/flame-engine/flame/contents/doc"
            )
            if response.status_code == 200:
                files = response.json()
                for file in files:
                    if file["name"].endswith(".md"):
                        content = requests.get(file["download_url"]).text
                        doc_path = self.raw_dir / f"github_{file['name']}"
                        with open(doc_path, 'w') as f:
                            f.write(content)
                        docs.append({
                            "source": "github",
                            "filename": file["name"],
                            "path": str(doc_path)
                        })
                        print(f"Saved: {file['name']}")
        except Exception as e:
            print(f"Error fetching GitHub docs: {e}")
        
        return docs
    
    def fetch_flame_docs(self) -> List[Dict[str, str]]:
        """Fetch documentation from flame-engine.org"""
        print("Fetching Flame documentation...")
        docs = []
        
        try:
            response = requests.get("https://docs.flame-engine.org/latest/")
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find all documentation pages
                for link in soup.find_all('a', href=True):
                    if link['href'].endswith('.html'):
                        page_url = f"https://docs.flame-engine.org/latest/{link['href']}"
                        page_response = requests.get(page_url)
                        if page_response.status_code == 200:
                            page_name = link['href'].replace('/', '_')
                            doc_path = self.raw_dir / f"flame_docs_{page_name}"
                            with open(doc_path, 'w') as f:
                                f.write(page_response.text)
                            docs.append({
                                "source": "flame_docs",
                                "filename": page_name,
                                "path": str(doc_path)
                            })
                            print(f"Saved: {page_name}")
        except Exception as e:
            print(f"Error fetching Flame docs: {e}")
        
        return docs
    
    def fetch_pub_dev_docs(self) -> List[Dict[str, str]]:
        """Fetch documentation from pub.dev"""
        print("Fetching pub.dev documentation...")
        docs = []
        
        try:
            response = requests.get(f"https://pub.dev/documentation/flame/{self.flame_version}")
            if response.status_code == 200:
                doc_path = self.raw_dir / "pub_dev_docs.html"
                with open(doc_path, 'w') as f:
                    f.write(response.text)
                docs.append({
                    "source": "pub_dev",
                    "filename": "pub_dev_docs.html",
                    "path": str(doc_path)
                })
                print("Saved pub.dev documentation")
        except Exception as e:
            print(f"Error fetching pub.dev docs: {e}")
        
        return docs
    
    def fetch_all_docs(self):
        """Fetch documentation from all sources"""
        all_docs = []
        all_docs.extend(self.fetch_github_docs())
        all_docs.extend(self.fetch_flame_docs())
        all_docs.extend(self.fetch_pub_dev_docs())
        
        self.processed_docs = all_docs
        
        # Save documentation metadata
        metadata = {
            "scrape_date": datetime.now().isoformat(),
            "flame_version": self.flame_version,
            "processed_docs": all_docs
        }
        
        with open(self.output_dir / "docs_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nProcessed {len(all_docs)} documents")
        return all_docs

if __name__ == "__main__":
    scraper = FlameScraper()
    
    # Test connections
    if scraper.test_connection():
        print("Successfully connected to all documentation sources")
        print(f"Current Flame version: {scraper.flame_version}")
        
        # Fetch all documentation
        scraper.fetch_all_docs()
    else:
        print("Failed to connect to some documentation sources")