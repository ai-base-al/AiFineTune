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
        
    def clean_code(self, code: str) -> str:
        lines = code.split('\n')
        cleaned_lines = []
        for line in lines:
            line = re.sub(r'^\s*\d+\s*', '', line)
            cleaned_lines.append(line)
        return '\n'.join(cleaned_lines)

    def clean_context(self, context: str) -> str:
        context = re.sub(r'\s+', ' ', context)
        context = context.replace('"', "'")
        return context.strip()
    
    def is_valid_code(self, code: str) -> bool:
        if not code:
            return False
        if len(code.strip().split('\n')) < 3:
            return False
        keywords = ['Flame', 'Game', 'Component', 'Vector2', 'import']
        return any(keyword in code for keyword in keywords)
    
    def process_markdown(self, content: str) -> List[Dict[str, str]]:
        examples = []
        sections = content.split('\n#')
        
        for section in sections:
            lines = section.split('\n')
            if lines:
                context = lines[0].strip('# ')
                code_blocks = re.finditer(r'```(?:dart)?\n(.*?)\n```', section, re.DOTALL)
                
                for block in code_blocks:
                    code = block.group(1).strip()
                    if self.is_valid_code(code):
                        local_context = '\n'.join(section[:block.start()].split('\n')[-3:])
                        examples.append({
                            "context": self.clean_context(f"{context}: {local_context}"),
                            "code": self.clean_code(code)
                        })
        
        return examples
    
    def process_html(self, content: str) -> List[Dict[str, str]]:
        examples = []
        soup = BeautifulSoup(content, 'html.parser')
        
        code_blocks = soup.find_all(['pre', 'code'])
        for block in code_blocks:
            code = block.get_text().strip()
            if self.is_valid_code(code):
                section = block.find_previous(['h1', 'h2', 'h3'])
                section_text = section.get_text() if section else ''
                
                context = block.find_previous(['p'])
                context_text = context.get_text() if context else ''
                
                examples.append({
                    "context": self.clean_context(f"{section_text}: {context_text}"),
                    "code": self.clean_code(code)
                })
        
        return examples
    
    def process_file(self, file_path: Path) -> List[Dict[str, str]]:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if file_path.suffix == '.md':
            return self.process_markdown(content)
        else:
            return self.process_html(content)
    
    def generate_prompts(self, context: str, code: str) -> List[Dict[str, str]]:
        prompts = [
            f"Create a Flutter Flame game component that {context}",
            f"Show me how to implement {context} using Flutter Flame",
            f"Write Flutter Flame code to {context}",
        ]
        
        return [{
            "prompt": prompt,
            "completion": code
        } for prompt in prompts]
    
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
            training_data.extend(self.generate_prompts(ex['context'], ex['code']))
        
        output_path = self.output_dir / "training_data.jsonl"
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in training_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"\nProcessed {len(all_examples)} examples")
        print(f"Generated {len(training_data)} training samples")
        
        sample_path = self.output_dir / "samples.jsonl"
        with open(sample_path, 'w', encoding='utf-8') as f:
            json.dump(all_examples[:5], f, ensure_ascii=False, indent=2)
        
        return training_data

if __name__ == "__main__":
    processor = FlameProcessor()
    processor.process_all()