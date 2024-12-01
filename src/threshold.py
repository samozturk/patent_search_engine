import numpy as np
import matplotlib.pyplot as plt

class MeanThresholdController:
    def __init__(self, similarity_scores):
        """
        Initialize with similarity scores
        
        :param similarity_scores: Array of similarity scores
        """
        self.scores = np.array(similarity_scores)
        self.mean_score = np.mean(self.scores)
        self.std_score = np.std(self.scores)
    
    def calculate_threshold(self, precision_recall_param=1.0):
        """
        Calculate threshold based on mean with adjustment parameter
        
        :param precision_recall_param: Threshold adjustment
                                       < 1.0: Favor precision
                                       > 1.0: Favor recall
        :return: Calculated threshold
        """
        # Threshold calculation
        threshold = self.mean_score + (self.std_score * precision_recall_param)
        return threshold
    
    def apply_threshold(self, precision_recall_param=1.0):
        """
        Apply threshold and return filtered results
        
        :param precision_recall_param: Precision-recall control parameter
        :return: Dictionary of retrieval results
        """
        # Calculate threshold
        threshold = self.calculate_threshold(precision_recall_param)
        
        # Apply threshold
        mask = self.scores >= threshold
        
        return {
            'retrieved_count': np.sum(mask),
            'retrieval_ratio': np.sum(mask) / len(self.scores),
            'threshold': threshold,
            'mean_retrieved_score': np.mean(self.scores[mask]) if np.sum(mask) > 0 else 0
        }
    
    def visualize_threshold_impact(self, precision_recall_params=[0.5, 1.0, 1.5]):
        """
        Visualize impact of different precision-recall parameters
        
        :param precision_recall_params: Parameters to evaluate
        """
        plt.figure(figsize=(10, 6))
        
        # Plot score distribution
        plt.hist(self.scores, bins=30, alpha=0.7, label='Similarity Scores')
        
        # Plot thresholds
        colors = ['red', 'green', 'blue']
        for param, color in zip(precision_recall_params, colors):
            threshold = self.mean_score + (self.std_score * param)
            plt.axvline(x=threshold, color=color, 
                        linestyle='--', 
                        label=f'Threshold (param={param})')
        
        plt.title('Similarity Score Distribution with Thresholds')
        plt.xlabel('Similarity Score')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

# Demonstration function
def demonstrate_mean_threshold():
    # Simulated similarity scores (e.g., from document embeddings)
    np.random.seed(42)
    similarity_scores = np.random.normal(0.5, 0.2, 1000)
    
    # Initialize controller
    controller = MeanThresholdController(similarity_scores)
    
    # Demonstrate different precision-recall scenarios
    scenarios = [
        (0.5, "Precision-Focused"),
        (1.0, "Balanced"),
        (1.5, "Recall-Focused")
    ]
    
    print("Precision-Recall Tradeoff Analysis:")
    for param, scenario_name in scenarios:
        results = controller.apply_threshold(param)
        print(f"\n{scenario_name} Scenario (param = {param}):")
        print(f"Threshold: {results['threshold']:.4f}")
        print(f"Retrieved Count: {results['retrieved_count']}")
        print(f"Retrieval Ratio: {results['retrieval_ratio']:.2%}")
        print(f"Mean Retrieved Score: {results['mean_retrieved_score']:.4f}")
    
    # Visualize threshold impact
    controller.visualize_threshold_impact()