from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json
from typing import List, Dict
import pandas as pd

class SemanticVectorSearch:
    def __init__(self):
        print("🤖 Loading sentence transformer model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.documents = []
        self.embeddings = None
    
    def build_vector_index(self, semantic_results: List[Dict]):
        """Build FAISS vector index from paper abstracts and entities"""
        
        print("🔍 Building vector search index...")
        
        # Prepare documents
        documents = []
        for paper in semantic_results:
            # Create searchable text from title + entities + relations
            title = paper.get('title', '')
            entities = [e['text'] for e in paper.get('entities', [])]
            relations = [f"{r['subject']} {r['predicate']} {r['object']}" 
                        for r in paper.get('relations', [])]
            
            doc_text = f"{title}. Entities: {', '.join(entities)}. Relations: {', '.join(relations)}"
            
            documents.append({
                'paper_id': paper.get('paper_id', 'unknown'),
                'title': title,
                'text': doc_text,
                'entity_count': len(entities),
                'relation_count': len(relations)
            })
        
        self.documents = documents
        
        # Create embeddings
        texts = [doc['text'] for doc in documents]
        self.embeddings = self.model.encode(texts)
        
        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings.astype('float32'))
        
        print(f"✅ Vector index built: {len(documents)} documents, {dimension} dimensions")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar papers using semantic similarity"""
        
        if self.index is None:
            return []
        
        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                result = self.documents[idx].copy()
                result['similarity_score'] = float(score)
                result['rank'] = i + 1
                results.append(result)
        
        return results
    
    def get_similar_entities(self, target_entity: str, semantic_results: List[Dict], top_k: int = 10) -> List[Dict]:
        """Find entities similar to target entity"""
        
        # Collect all entities
        all_entities = []
        for paper in semantic_results:
            for entity in paper.get('entities', []):
                all_entities.append(entity['text'])
        
        # Remove duplicates
        unique_entities = list(set(all_entities))
        
        if target_entity not in unique_entities:
            return []
        
        # Create embeddings for entities
        entity_embeddings = self.model.encode(unique_entities)
        faiss.normalize_L2(entity_embeddings)
        
        # Build temporary index
        temp_index = faiss.IndexFlatIP(entity_embeddings.shape[1])
        temp_index.add(entity_embeddings.astype('float32'))
        
        # Find target entity index
        target_idx = unique_entities.index(target_entity)
        target_embedding = entity_embeddings[target_idx:target_idx+1]
        
        # Search for similar entities
        scores, indices = temp_index.search(target_embedding.astype('float32'), top_k + 1)
        
        results = []
        for score, idx in zip(scores[0][1:], indices[0][1:]):  # Skip first (self)
            results.append({
                'entity': unique_entities[idx],
                'similarity': float(score)
            })
        
        return results