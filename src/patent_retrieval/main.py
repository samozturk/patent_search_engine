from engine import PatentRetrievalService, load_config
import json

def main():
    # Load configuration
    config = load_config('config.json')
    
    # Initialize service
    service = PatentRetrievalService(
        dataset_path=config['dataset_path'],
        model_name=config['embedding_model']
    )
    
    # Example usage
    keywords = ["artificial intelligence", "machine learning"]
    results, metadata = service.retrieve_patents(
        keywords, 
        precision_recall_balance=0.001
    )
    
    print(results)

    # Demonstrate dataset update
    new_abstracts = [
        "A novel method for improving machine learning algorithms",
        "Artificial intelligence in medical diagnostics"
    ]
    service.update_dataset(new_abstracts)

if __name__ == "__main__":
    main()