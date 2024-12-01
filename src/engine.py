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
    
    def _calculate_relevance_scores(
        self,
        keyword_embedding: np.ndarray,
        precision_recall_balance: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate relevance scores with precision-recall adjustment
        
        Args:
            keyword_embedding: Encoded keywords
            precision_recall_balance: Value between 0 and 1, higher means more precision
        
        Returns:
            Tuple of (scores, indices)
        """
        # Get raw similarity scores
        distances, indices = self.index.search(
            keyword_embedding,
            k=len(self.abstracts)
        )
        
        # Calculate adjusted scores based on precision-recall balance
        avg_similarities = np.mean(distances, axis=0)
        
        # Apply exponential scaling based on precision-recall balance
        # Higher balance -> steeper curve favoring high similarity matches
        power = 1 + (precision_recall_balance)  # Ranges from 1 to 2
        adjusted_scores = np.power(avg_similarities, power)
        
        # Normalize scores
        adjusted_scores = (adjusted_scores - adjusted_scores.min()) / (adjusted_scores.max() - adjusted_scores.min())
        
        return adjusted_scores, indices
    
    def retrieve_patents(
        self,
        keywords: List[str],
        precision_recall_balance: float = 0.5,
        min_score_threshold: float = 0.1
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
        
        # Calculate relevance scores with precision-recall adjustment
        adjusted_scores, indices = self._calculate_relevance_scores(
            keyword_embedding,
            precision_recall_balance
        )
        
        # Filter results based on adjusted threshold
        dynamic_threshold = min_score_threshold * (1 + precision_recall_balance)
        relevant_mask = adjusted_scores >= dynamic_threshold
        
        relevant_indices = indices[0][relevant_mask]
        relevant_scores = adjusted_scores[relevant_mask]
        
        # Sort by relevance
        sorted_idx = np.argsort(relevant_scores)[::-1]
        ranked_results = [
            (relevant_indices[i], relevant_scores[i])
            for i in sorted_idx
        ]
        
        # Prepare results with explanations
        results = {
            "retrieved_patents": [
                {
                    "abstract": self.abstracts[idx],
                    "relevance_score": float(score),
                    "explanation": self._generate_explanation(
                        idx,
                        keywords,
                        precision_recall_balance
                    )
                } for idx, score in ranked_results
            ],
            "metadata": {
                "keywords": keywords,
                "total_matches": len(ranked_results),
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