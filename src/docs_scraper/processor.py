from pathlib import Path
import json
from bs4 import BeautifulSoup
import re
from typing import List, Dict, Any

class FlameProcessor:
    def __init__(self, raw_dir: str = "data/raw", output_dir: str = "data/processed"):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.code_examples = []
        self.api_docs = []
        
    def process_markdown(self, content: str) -> List[Dict[str, str]]:
        examples = []
        code_blocks = re.finditer(r'```(?:dart)?\n(.*?)\n```', content, re.DOTALL)
        
        for block in code_blocks:
            code = block.group(1).strip()
            if 'Flame' in code or 'Game' in code:
                context = content[:block.start()].split('\n')[-2:]
                examples.append({
                    "context": ' '.join(context).strip(),
                    "code": code
                })
        
        return examples
    
    def process_html(self, content: str) -> List[Dict[str, str]]:
        examples = []
        soup = BeautifulSoup(content, 'html.parser')
        
        code_blocks = soup.find_all(['pre', 'code'])
        for block in code_blocks:
            code = block.get_text().strip()
            if 'Flame' in code or 'Game' in code:
                context = block.find_previous(['p', 'h1', 'h2', 'h3'])
                if context:
                    examples.append({
                        "context": context.get_text().strip(),
                        "code": code
                    })
        
        return examples
    
    def process_file(self, file_path: Path) -> List[Dict[str, str]]:
        with open(file_path, 'r') as f:
            content = f.read()
            
        if file_path.suffix == '.md':
            return self.process_markdown(content)
        else:
            return self.process_html(content)
    
    def process_all(self):
        all_examples = []
        
        for file_path in self.raw_dir.glob('*'):
            try:
                examples = self.process_file(file_path)
                all_examples.extend(examples)
                print(f"Processed {file_path.name}: {len(examples)} examples")
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")
        
        training_data = []
        for ex in all_examples:
            training_data.append({
                "prompt": f"Create Flame game code for: {ex['context']}",
                "completion": ex['code']
            })
            training_data.append({
                "prompt": f"How to implement {ex['context']} in FlutterFlame?",
                "completion": ex['code']
            })
        
        output_path = self.output_dir / "training_data.jsonl"
        with open(output_path, 'w') as f:
            for item in training_data:
                f.write(json.dumps(item) + '\n')
        
        print(f"\nProcessed {len(all_examples)} examples")
        print(f"Generated {len(training_data)} training samples")
        return training_data

if __name__ == "__main__":
    processor = FlameProcessor()
    processor.process_all()