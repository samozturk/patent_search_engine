import os
import json
import faiss
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer

class PatentRetrievalService:
    def __init__(self, dataset_path: str, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        """
        Initialize the Patent Retrieval Service with FAISS and Sentence Transformers
        
        Args:
            dataset_path (str): Path to the patent abstracts dataset
            model_name (str): Multilingual embedding model
        """
        self.dataset_path = dataset_path
        self.model_name = model_name
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(model_name)
        
        # Data storage
        self.abstracts = []
        self.embeddings = None
        self.index = None
        
        # Load and preprocess data
        self._load_dataset()
        self._create_faiss_index()
    
    def _load_dataset(self):
        """Load patent abstracts from the dataset file"""
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            self.abstracts = [line.strip() for line in f if line.strip()]
    
    def _create_faiss_index(self):
        """Create FAISS index for efficient similarity search"""
        # Generate embeddings for all abstracts
        self.embeddings = self.embedding_model.encode(
            self.abstracts,
            show_progress_bar=True
        )
        
        # Normalize embeddings
        faiss.normalize_L2(self.embeddings)
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)

    def retrieve_patents(
        self,
        keywords: List[str],
        precision_recall_balance: float = 0.5
    ) -> Dict[str, Any]:
        """
        Retrieve patent abstracts related to given keywords with precision-recall tradeoff
        
        Args:
            keywords: Input keywords
            precision_recall_balance: Value between 0 and 1
                - Higher values (>0.5) favor precision: returns fewer but more relevant results
                - Lower values (<0.5) favor recall: returns more results with varied relevance
            min_score_threshold: Minimum relevance score to include in results
        
        Returns:
            Dict containing retrieved patents and relevance scores
        """
        # Validate precision-recall balance
        if not 0 <= precision_recall_balance <= 1:
            raise ValueError("precision_recall_balance must be between 0 and 1")
            
        # Generate embedding for keywords
        keyword_embedding = self.embedding_model.encode([' '.join(keywords)])
        faiss.normalize_L2(keyword_embedding)
        
        # Get raw similarity scores
        distances, indices = self.index.search(
            keyword_embedding,
            k=len(self.abstracts)
        )

        # Apply precision-recall balance
        threshold = np.percentile(
            distances,
            (1 - precision_recall_balance) * 100
        )
        # Get indices of abstracts with scores above threshold
        relevant_indices = indices[distances >= threshold]

        # Store them in a dictionary
        results = {'abstract' : [],
        'relevance_score': [],
        }
        for i, idx in enumerate(relevant_indices):
            results['abstract'].append(self.abstracts[idx])
            results['relevance_score'].append(distances[0][i])

        # Prepare results with explanations
        results = {
            "metadata": {
                "keywords": keywords,
                "total_matches": len(relevant_indices),
                "precision_recall_balance": precision_recall_balance,
                "embedding_model": self.model_name
            }
        }
        
        return results
    
    def _generate_explanation(
        self,
        abstract_idx: int,
        keywords: List[str],
        precision_recall_balance: float
    ) -> str:
        """Generate an explanation for why an abstract is relevant"""
        abstract = self.abstracts[abstract_idx]
        
        # Find keyword matches
        keyword_matches = [
            kw for kw in keywords 
            if kw.lower() in abstract.lower()
        ]
        
        # Compute semantic similarity
        keyword_embedding = self.embedding_model.encode(keywords)
        abstract_embedding = self.embeddings[abstract_idx]
        semantic_similarities = [
            float(np.dot(keyword_emb, abstract_embedding)) 
            for keyword_emb in keyword_embedding
        ]
        
        # Add precision-recall context to explanation
        precision_context = (
            "high precision mode - focusing on closest matches"
            if precision_recall_balance > 0.7
            else "balanced precision-recall"
            if 0.3 <= precision_recall_balance <= 0.7
            else "high recall mode - including broader matches"
        )
        
        explanation = (
            f"Match Analysis ({precision_context}):\n"
            f"- Direct keyword matches: {len(keyword_matches)}\n"
            f"- Semantic similarity score: {np.mean(semantic_similarities):.2f}\n"
            f"- Matched keywords: {', '.join(keyword_matches) if keyword_matches else 'None'}"
        )
        return explanation
    
    def update_dataset(self, new_abstracts: List[str]):
        """Update dataset with new patent abstracts"""
        self.abstracts.extend(new_abstracts)
        
        # Generate embeddings for new abstracts
        new_embeddings = self.embedding_model.encode(new_abstracts)
        faiss.normalize_L2(new_embeddings)
        
        # Add to existing embeddings and index
        if self.embeddings is not None:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
            self.index.add(new_embeddings)
        else:
            self._create_faiss_index()

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)