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

    def retrieve_patents(
        self,
        keywords: List[str],
        precision_recall_balance: float = 0.5
    ):
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
        'degree_between': []
        }
        for i, idx in enumerate(relevant_indices):
            results['abstract'].append(self.abstracts[idx])
            results['relevance_score'].append(float(distances[0][i]))
            results['degree_between'].append(float(self.get_degree_between(keyword_embedding[0], self.embeddings[idx])))
        # Prepare results with explanations
        metadata = {
                "keywords": keywords,
                "total_matches": len(relevant_indices),
                "precision_recall_balance": precision_recall_balance,
                "embedding_model": self.model_name
            }
        metadata_df = pd.DataFrame(metadata)
        
        return results, metadata_df

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

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)