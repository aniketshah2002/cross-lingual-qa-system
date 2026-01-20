import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import os
import json

def create_embeddings(model, documents):
    """
    Generates embeddings for a list of documents.
    """
    print(f"Generating embeddings for {len(documents)} documents...")
import os 

def create_embeddings(model, documents):
    """
    Generates embedings for a list of documents.
    
    Args:
        model: The SenenceTransformer model.
        documents (lists): A list of strings (sentences).
        
    Returns:
        np.ndarray: An array of embeddings.
    """
    print(f"Generating embeddings for {len(documents)} documemnts...")
    # The model.encode() method takes a list of sentences and returns
    # a list of their corresponding embeddings. We can specify and returns
    # to process multiple sentences at once, which is much faster.
    embeddings = model.encode(documents, show_progress_bar=True, batch_size=64)
    print("Embeddings generated successfully.")
    return embeddings

def main():
    print("Starting Step 2: Generating Embeddings (BALANCED VERSION)...")

    # --- 1. Load the Data (Balanced) ---
    target_languages = ['en', 'de', 'fr', 'es']
    documents = [] 
    metadatas = [] 

    print("Loading datasets and balancing classes...")
    try:
        for lang in target_languages:
            print(f"Processing {lang}...")
            # Load the full dataset for this language
            dataset = load_dataset("mteb/amazon_reviews_multi", lang, split="train")
            
            # We want 50 reviews for EACH star rating (0-4 in dataset, which is 1-5 stars)
            # This ensures we have exactly 250 reviews per language
            for star_label in range(5): # 0, 1, 2, 3, 4
                
                # Filter the dataset to find reviews with this specific star rating
                # We use .filter() to find matching rows, then take the first 50
                filtered = dataset.filter(lambda x: x['label'] == star_label).select(range(50))
                
                for item in filtered:
                    review_text = item['text']
                    star_rating = item['label'] + 1 # Convert 0-4 to 1-5
                    
                    documents.append(review_text)
                    metadatas.append({
                        "language": lang,
                        "stars": star_rating
                    })
            print(f"  -> Added 250 balanced reviews for {lang}")
                
        print(f"\nTotal Loaded: {len(documents)} reviews.")

    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- 2. Load the Model ---
    model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
    print(f"Loading model: {model_name}...")
    model = SentenceTransformer(model_name)

    # --- 3. Generate Embeddings ---
    embeddings = create_embeddings(model, documents)

    # --- 4. Save Everything ---
    output_dir = "knowledge_base"
    os.makedirs(output_dir, exist_ok=True)

    embeddings_path = os.path.join(output_dir, "embeddings.npy")
    sentences_path = os.path.join(output_dir, "sentences.txt")
    metadata_path = os.path.join(output_dir, "metadata.json")

    print(f"Saving embeddings to {embeddings_path}")
    np.save(embeddings_path, embeddings)

    print(f"Saving sentences to {sentences_path}")
    with open(sentences_path, "w", encoding="utf-8") as f:
        for sentence in documents:
            clean_sentence = sentence.replace("\n", " ")
            f.write(clean_sentence + "\n")

    print(f"Saving metadata to {metadata_path}")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadatas, f)

    print("\nStep 2 complete! Balanced Knowledge Base created.")
    """
    Main function to load data, chose a model, create embeddings.
    and save them to disk.
    """
    print("Starting Step 2: Generating Text Embeddings...")

    # --1. Load the dataset --
    # We will use tha same dataset as before.
    print("Loading the Tatoeba dataset...")
    dataset = load_dataset("tatoeba", lang1="de", lang2 = "en", split="train")
    print("Dataset Loaded.")

    # --2. Select a Pre trained model --
    # We choose a model from the sentence-transformer library.
    # 'paraphrase-multilingual-MiniLM-L12-v2' is a great choice for this task.
    # It's powerful, fast and supports over 50 Languages.
    model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
    print(f"Loading the mode: {model_name}...")
    model = SentenceTransformer(model_name)
    print("Model Loaded successfully.")

    # --3. Prepare the Documents --
    # To make this example run faster, we'll only use the first 10,000
    # German sentences as our "knowledge base" that we will search through.
    num_documents = 10000
    german_sentences = [ex['translation']['de'] for ex in dataset.select(range(num_documents))]
    print(f"Prepared {len(german_sentences)} German sentences as our knowledge base.")

    # --4. Generate and Save Embeddings --
    # this is the code step where the model cconverts text to numbers.
    # this might take a few minutes depending on your computer's hardware.
    german_embeddings = create_embeddings(model, german_sentences)

    # we will save our work so we don't have to re-run this step.
    # we'll save the embeddings (the vectors) and the original sentences.
    output_dir = "knowledge_base"
    os.makedirs(output_dir, exist_ok=True) #create a directoery to store our files.

    embeddings_path = os.path.join(output_dir, "embeddings.npy")
    sentence_path = os.path.join(output_dir, "sentences.txt")

    print(f"Saving embeddings to {embeddings_path}")
    np.save(embeddings_path, german_embeddings)

    print(f"Saving sentences to {sentence_path}")
    with open(sentence_path, "w", encoding="utf-8") as f:
        for sentence in german_sentences:
            f.write(sentence + "\n")

    print("\nStep 2 complete! We now have a knowledge base of sentences and their embeddings.")

if __name__ == "__main__":
    main()