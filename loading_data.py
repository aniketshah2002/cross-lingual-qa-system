from datasets import load_dataset
<<<<<<< HEAD
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
=======

def main():
    """
    Main function to load the dataset and display some examples.
    """
    print("Starting Step 1: Loading the dataset...")

    try:
        """
        we will use the 'tatoeba dataset.
        it's a collection of parallet sentences in many languages.
        the second argument, 'de-en', specifies that we want the German-English language pair.
        The 'split' argument tells the function we want the 'train' part of the dataset."""
        print("Downloading the Tatoeba dataset for German-English...")
        dataset = load_dataset("tatoeba", lang1="de", lang2="en", split="train")
        print("Dataset downloaded successfully!")

        #The dataset object now holds our data. It behaves a lot like a python list.
        print(f"\nNumber of sentence pairs in the dataset: {len(dataset)}")

        #let's look at the first 5 sentence pairs to understand the structure.
        print(f"\nHere are the first 5 examples from the dataset:")

        #Each item in the dataset is a dictionary.
        # The dictionary has a key 'translation' whcih itself contains another.
        # dictionary with the language codes ('de' for german, 'en' for English).
        for i in range(5):
            example = dataset[i]
            german_sentence = example['translation']['de']
            english_sentence = example['translation']['en']

            print(f"\n-- Example {i+1} --")
            print(f" German: {german_sentence}")
            print(f" English: {english_sentence}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure you have an internet connection and have installed the 'datasets' library.")
        print("You can install it with: pip install datasets")
>>>>>>> 12a97993d1425d5103c71a93c9ac1423827b0633

if __name__ == "__main__":
    main()