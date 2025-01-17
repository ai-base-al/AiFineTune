# src/docs_scraper/flame_docs.py

import requests
from pathlib import Path
from typing import List, Dict
import json
from datetime import datetime

class FlameScraper:
    """Scrapes and processes FlutterFlame documentation"""
    
    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Version tracking
        self.flame_version = self._get_flame_version()
        
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
    
    def save_metadata(self):
        """Save scraping metadata including timestamps and versions"""
        metadata = {
            "scrape_date": datetime.now().isoformat(),
            "flame_version": self.flame_version,
            "sources": {
                "pub_dev": "https://pub.dev/documentation/flame/latest/",
                "docs": "https://docs.flame-engine.org/latest/",
                "github": "https://github.com/flame-engine/flame/tree/main/doc"
            }
        }
        
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
            
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
            except Exception as e:
                print(f"Failed to connect to {url}: {e}")
                results.append(False)
        
        return all(results)

if __name__ == "__main__":
    scraper = FlameScraper()
    
    # Test connections
    if scraper.test_connection():
        print("Successfully connected to all documentation sources")
        print(f"Current Flame version: {scraper.flame_version}")
        
        # Save initial metadata
        scraper.save_metadata()
    else:
        print("Failed to connect to some documentation sources")