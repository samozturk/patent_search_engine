import os
import json
import faiss
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from langdetect import detect

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
        """
        Load patent abstracts from the dataset file
        """
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            self.abstracts = [line.strip() for line in f if line.strip()]
    
    def _create_faiss_index(self):
        """
        Create FAISS index for efficient similarity search
        """
        # Generate embeddings for all abstracts
        self.embeddings = self.embedding_model.encode(
            self.abstracts,
            show_progress_bar=True
        )
        
        # Normalize embeddings
        faiss.normalize_L2(self.embeddings)
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product index for cosine similarity
        self.index.add(self.embeddings)
    
    def retrieve_patents(
        self, 
        keywords: List[str], 
        precision_recall_balance: float = 0.5,
        # top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Retrieve patent abstracts related to given keywords
        
        Args:
            keywords (List[str]): Input keywords
            precision_recall_balance (float): Controls precision-recall tradeoff
            top_k (int): Number of top results to return
        
        Returns:
            Dict containing retrieved patents and relevance scores
        """
        # Generate embedding for keywords
        keyword_embedding = self.embedding_model.encode(keywords)
        faiss.normalize_L2(keyword_embedding)
        
        # Perform similarity search
        distances, indices = self.index.search(keyword_embedding, k=len(self.abstracts)) # Use self.k or top_k maybe?
        
        # Aggregate similarities across keywords
        avg_similarities = np.mean(distances, axis=0)
        
        # Apply precision-recall balance
        threshold = np.percentile(
            avg_similarities, 
            (1 - precision_recall_balance) * 100
        )
        
        # Filter and rank results
        relevant_indices = np.where(avg_similarities >= threshold)[0]
        ranked_results = sorted(
            [(idx, avg_similarities[idx]) for idx in relevant_indices], 
            key=lambda x: x[1], 
            reverse=True
        )#[:top_k]
        
        # Prepare results with explanations
        results = {
            "retrieved_patents": [
                {
                    "abstract": self.abstracts[idx],
                    "relevance_score": float(score),
                    "explanation": self._generate_explanation(idx, keywords)
                } for idx, score in ranked_results
            ],
            "metadata": {
                "keywords": keywords,
                "total_matches": len(ranked_results),
                "embedding_model": self.model_name
            }
        }
        
        return results
    
    def _generate_explanation(self, abstract_idx: int, keywords: List[str]) -> str:
        """
        Generate an explanation for why an abstract is relevant
        
        Args:
            abstract_idx (int): Index of the patent abstract
            keywords (List[str]): Input keywords
        
        Returns:
            str: Explanation of relevance
        """
        abstract = self.abstracts[abstract_idx]
        
        # Find keyword matches and compute semantic similarity
        keyword_matches = [
            kw for kw in keywords 
            if kw.lower() in abstract.lower()
        ]
        
        # Compute semantic similarity between keyword and abstract
        keyword_embedding = self.embedding_model.encode(keywords)
        abstract_embedding = self.embeddings[abstract_idx]
        semantic_similarities = [
            float(np.dot(keyword_emb, abstract_embedding)) 
            for keyword_emb in keyword_embedding
        ]
        
        explanation = (
            f"Semantic Match: {len(keyword_matches)} direct keywords, "
            f"Average Semantic Similarity: {np.mean(semantic_similarities):.2f}"
        )
        return explanation
    
    def update_dataset(self, new_abstracts: List[str]):
        """
        Update dataset with new patent abstracts
        
        Args:
            new_abstracts (List[str]): New patent abstracts to add
        """
        # Append new abstracts
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
    """
    Load configuration from JSON file
    
    Args:
        config_path (str): Path to configuration file
    
    Returns:
        Dict containing configuration parameters
    """
    with open(config_path, 'r') as f:
        return json.load(f)

