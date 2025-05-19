import numpy as np

def process_data(filename: str) -> float:
    """
    Process a text file containing integer:float pairs.
    
    Args:
        filename (str): Path to input file
        
    Returns:
        float: Mean value of filtered floats
        
    Raises:
        ValueError: If file format is invalid
        FileNotFoundError: If file doesn't exist
    """
    try:
        with open(filename, 'r') as f:
            # Parse lines and filter in single comprehension
            values = [float(line.split(':')[1]) 
                     for line in f 
                     if len(line.split(':')) == 2]
            
            # Filter values >= 0.5 and compute mean
            filtered_values = np.array([v for v in values if v >= 0.4])
            
            if len(filtered_values) == 0:
                raise ValueError("No values >= 0.5 found in input")
                
            # return np.median(filtered_values)
            return np.mean(filtered_values)
            
    except ValueError as e:
        raise ValueError(f"Invalid file format: {str(e)}")
        
if __name__ == "__main__":
    try:
        result = process_data("/home/xycoord/output-e2e/mask_predictions/test_mix_5966/iou.txt")
        print(f"Mean of filtered values: {result:.6f}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {str(e)}")