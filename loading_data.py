from datasets import load_dataset

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

if __name__ == "__main__":
    main()