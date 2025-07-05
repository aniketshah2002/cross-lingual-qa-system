import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
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