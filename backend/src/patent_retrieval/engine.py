import os
import json
import faiss
import numpy as np
from numpy.linalg import norm
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

    def retrieve_patents(self, keywords: List[str], precision_recall_balance: float = 0.5):
        """
    Retrieve relevant patents based on keyword search using semantic similarity.

    Args:
        keywords (List[str]): List of search keywords to find relevant patents
        precision_recall_balance (float, optional): Controls the trade-off between precision and recall.
            Values closer to 1.0 favor recall (more results), while values closer to 0.0 favor precision
            (fewer but more relevant results). Defaults to 0.5.

    Returns:
        Tuple[Dict, Dict]: A tuple containing:
            - results (Dict): Dictionary with three lists:
                - abstract: List of retrieved patent abstracts
                - relevance_score: Corresponding similarity scores
                - degree_between: Angular distance between query and result embeddings
            - metadata (Dict): Search metadata containing the input keywords

    Raises:
        ValueError: If precision_recall_balance is not between 0 and 1

    Example:
        >>> service = PatentRetrievalService("patents.txt")
        >>> results, metadata = service.retrieve_patents(["autonomous", "vehicle"])
    """
        # Validate precision-recall balance
        if not 0 <= precision_recall_balance <= 1:
            raise ValueError("precision_recall_balance must be between 0 and 1")
            
        # Generate embedding for keywords
        keyword_embedding = self.embedding_model.encode([' '.join(keywords)])
        faiss.normalize_L2(keyword_embedding)
        
        # Get raw similarity scores
        distances, indices = self.index.search(keyword_embedding, k=len(self.abstracts))
        
        # Convert numpy arrays to native Python types
        distances = distances[0].tolist()
        indices = indices[0].tolist()
        
        # Apply precision-recall balance
        threshold = float(np.percentile(distances, (1 - precision_recall_balance) * 100))
        
        # Create results dictionary with native Python types
        results = {'abstract': [], 'relevance_score': [], 'degree_between': []}
        
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            if dist >= threshold:
                results['abstract'].append(self.abstracts[idx])
                results['relevance_score'].append(float(dist))
                results['degree_between'].append(float(self.get_degree_between(
                    keyword_embedding[0].tolist(), 
                    self.embeddings[idx].tolist()
                )))
                
        return results, {"keywords": keywords}

    def get_degree_between(self, a: np.array, b: np.array) -> float:
        cosine = np.dot(a,b) / (norm(a) * norm(b))
        return np.arccos(cosine) * 180 / np.pi
    
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

# def load_config(config_path: str) -> Dict[str, Any]:
#     """Load configuration from JSON file"""
#     with open(config_path, 'r') as f:
#         return json.load(f)