import spacy
import json
from typing import List, Dict

class SimpleNER:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
    
    def extract_entities(self, paper_data: Dict) -> Dict:
        paper_id = paper_data.get('paper_id', 'unknown')
        
        # Get text
        title = paper_data.get('metadata', {}).get('title', '')
        abstract = ''
        body = ''
        
        # Extract abstract
        for section in paper_data.get('metadata', {}).get('abstract', []):
            abstract += section.get('text', '') + ' '
        
        # Extract body (first 3 sections)
        for section in paper_data.get('metadata', {}).get('body_text', [])[:3]:
            body += section.get('text', '') + ' '
        
        # Combine all text
        full_text = f"{title} {abstract} {body}"
        
        # Extract entities
        doc = self.nlp(full_text[:10000])  # Limit to 10k chars
        entities = []
        
        for ent in doc.ents:
            if len(ent.text.strip()) > 2 and ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT']:
                entities.append({
                    'text': ent.text.strip(),
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
        
        return {
            'paper_id': paper_id,
            'title': title,
            'entities': entities,
            'entity_count': len(entities)
        }