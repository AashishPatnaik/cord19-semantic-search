import re
from typing import List, Dict
from itertools import combinations

class SimpleRE:
    def __init__(self):
        self.patterns = {
            'TREATS': [
                r'(\w+)\s+treat\w*\s+(\w+)',
                r'(\w+)\s+effective\s+against\s+(\w+)',
                r'(\w+)\s+therapy\s+for\s+(\w+)'
            ],
            'LOCATED_AT': [
                r'(\w+\s+\w+)\s+at\s+(\w+\s+\w+)',
                r'(\w+\s+\w+)\s+from\s+(\w+\s+\w+)'
            ],
            'ASSOCIATED_WITH': [
                r'(\w+)\s+associated\s+with\s+(\w+)',
                r'(\w+)\s+linked\s+to\s+(\w+)'
            ]
        }
    
    def extract_relations(self, text: str, entities: List[Dict]) -> List[Dict]:
        relations = []
        text = text.lower()
        
        for rel_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    relations.append({
                        'subject': match.group(1),
                        'predicate': rel_type,
                        'object': match.group(2),
                        'confidence': 0.8
                    })
        
        return relations[:10]  # Limit to 10 relations