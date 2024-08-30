from transformers import CLIPTokenizer
import argparse

def check_word_in_tokenizer(tokenizer, word):
    # Get the token ID for the word
    token_id = tokenizer.convert_tokens_to_ids(word)
    
    # If the token ID is the same as the unknown token ID, the word is not in the vocabulary
    if token_id == tokenizer.unk_token_id:
        return False
    else:
        return True

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Check if a word exists in CLIP's tokenizer vocabulary")
    parser.add_argument("-w", "--words", nargs='+', required=True, help="Word(s) to check")

    args = parser.parse_args()
    tokenizer = CLIPTokenizer.from_pretrained("checkpoints/stable-diffusion-v1-4", subfolder="tokenizer")

    # Check each word
    for word in args.words:
        exists = check_word_in_tokenizer(tokenizer, word)
        if exists:
            print(f"The word '{word}' exists in the CLIP tokenizer vocabulary.")
        else:
            print(f"The word '{word}' does NOT exist in the CLIP tokenizer vocabulary.")

if __name__ == "__main__":
    main()