import json
import sys
import os
from typing import List, Dict, Any

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def read_booksum_data(file_path: str = "data/benchmark_base_chapters_example.json") -> List[Dict[str, Any]]:
    """
    Read booksum data from the benchmark base example JSON file.
    
    Args:
        file_path: Path to the JSON file containing booksum data
        
    Returns:
        List of dictionaries containing booksum data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Transform the data to match the expected format
        booksum_data = []
        for i, item in enumerate(data):
            booksum_data.append({
                "chapter_id": item["chapter_id"],
                "chapter_text": item["chapter_text"],
            })
        
        return booksum_data
    
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {file_path}.")
        return []
    except Exception as e:
        print(f"Error reading booksum data: {str(e)}")
        return []

if __name__ == "__main__":
    # Example usage
    booksum_data = read_booksum_data()
    
    if booksum_data:
        print(f"Successfully loaded {len(booksum_data)} chapters.")
        print(f"First chapter ID: {booksum_data[0]['chapter_id']}")
        print(f"First chapter text length: {len(booksum_data[0]['chapter_text'])} characters")
    else:
        print("Failed to load booksum data.")
