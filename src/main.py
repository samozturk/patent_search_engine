from src.engine import PatentRetrievalService, load_config
import json

def main():
    # Load configuration
    config = load_config('config.json')
    
    # Initialize service
    service = PatentRetrievalService(
        dataset_path=config['dataset_path'],
        model_name=config.get('embedding_model', 'paraphrase-multilingual-MiniLM-L12-v2')
    )
    
    # Example usage
    keywords = ["artificial intelligence", "machine learning"]
    results = service.retrieve_patents(
        keywords, 
        precision_recall_balance=0.7
    )
    
    print(json.dumps(results, indent=2))

    # Demonstrate dataset update
    new_abstracts = [
        "A novel method for improving machine learning algorithms",
        "Artificial intelligence in medical diagnostics"
    ]
    service.update_dataset(new_abstracts)

if __name__ == "__main__":
    main()