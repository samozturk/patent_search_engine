import pytest
import numpy as np
import tempfile
import os
import json
from src.patent_retrieval.engine import PatentRetrievalService, load_config

@pytest.fixture
def test_abstracts():
    return [
        "A method for machine learning implementation",
        "Novel solar panel technology",
        "Improved battery storage system",
        "Artificial intelligence in healthcare"
    ]

@pytest.fixture
def temp_dataset_file(test_abstracts):
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
    for abstract in test_abstracts:
        temp_file.write(abstract + '\n')
    temp_file.close()
    yield temp_file.name
    os.unlink(temp_file.name)

@pytest.fixture
def patent_service(temp_dataset_file):
    return PatentRetrievalService(temp_dataset_file)

def test_initialization(patent_service, test_abstracts):
    assert len(patent_service.abstracts) == 4
    assert patent_service.embeddings is not None
    assert patent_service.index is not None

def test_retrieve_patents(patent_service):
    keywords = ["machine", "learning"]
    results, metadata = patent_service.retrieve_patents(keywords)
    
    assert 'abstract' in results
    assert 'relevance_score' in results
    assert 'degree_between' in results
    assert metadata['keywords'] == keywords

def test_precision_recall_balance_validation(patent_service):
    keywords = ["machine", "learning"]
    with pytest.raises(ValueError):
        patent_service.retrieve_patents(keywords, precision_recall_balance=1.5)

def test_update_dataset(patent_service):
    new_abstracts = ["New quantum computing method"]
    initial_length = len(patent_service.abstracts)
    
    patent_service.update_dataset(new_abstracts)
    
    assert len(patent_service.abstracts) == initial_length + 1
    assert patent_service.abstracts[-1] == new_abstracts[0]

def test_get_degree_between(patent_service):
    vector1 = np.array([1, 0, 0])
    vector2 = np.array([0, 1, 0])
    
    degree = patent_service.get_degree_between(vector1, vector2)
    assert pytest.approx(degree, abs=1e-5) == 90.0

def test_load_config():
    config_data = {
        "model_name": "test-model",
        "dataset_path": "test/path"
    }
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_config:
        temp_config.write(json.dumps(config_data))
    
    loaded_config = load_config(temp_config.name)
    assert loaded_config == config_data
    
    os.unlink(temp_config.name)
