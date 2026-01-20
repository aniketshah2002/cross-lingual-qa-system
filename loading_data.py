from datasets import load_dataset
import pandas as pd

def main():
    """
    Step 1: Load Real-World Amazon Reviews (Multilingual)
    Fixed for 'mteb/amazon_reviews_multi' schema.
    """
    print("Starting Step 1: Loading the Amazon Multilingual dataset...")

    # We will load reviews from these 4 languages
    target_languages = ['en', 'de', 'fr', 'es'] 
    
    # We will collect all reviews here
    all_reviews = []

    try:
        for lang in target_languages:
            print(f"Downloading {lang} reviews...")
            
            # Loading the MTEB version of the Amazon dataset
            dataset = load_dataset("mteb/amazon_reviews_multi", lang, split="train")
            
            # Taking 1,000 reviews per language
            subset = dataset.select(range(1000))
            
            for item in subset:
                # MAPPING FIX: 
                # 'text' contains the review
                # 'label' contains the stars (0-4 scale, so we add 1 to make it 1-5)
                
                all_reviews.append({
                    "text": item['text'],  # Changed from 'review_body'
                    "language": lang,
                    "stars": item['label'] + 1, # Changed from 'stars' and adjusted scale
                    "category": "General" # Category is missing in this version, setting default
                })
        
        print(f"\nSuccessfully loaded {len(all_reviews)} total reviews.")
        
        # Let's peek at the data to make sure it looks "real"
        print("\n--- Data Preview ---")
        # Show first 3 (English) and last 3 (Spanish)
        examples = all_reviews[:3] + all_reviews[-3:]
        
        for i, ex in enumerate(examples):
            print(f"[{ex['language'].upper()}] ({ex['stars']} stars): {ex['text'][:100]}...")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Detailed error info: Check if 'text' or 'label' keys exist in the dataset.")

if __name__ == "__main__":
    main()