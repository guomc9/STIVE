from transformers import CLIPTokenizer
import argparse

def check_word_in_tokenizer(tokenizer, word):
    # Get the token ID for the word
    tokens = tokenizer.tokenize(word)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(f'tokenize {word} to tokens: {tokens}, and token_ids: {token_ids}')

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Check if a word exists in CLIP's tokenizer vocabulary")
    parser.add_argument("-w", "--words", nargs='+', required=True, help="Word(s) to check")

    args = parser.parse_args()
    tokenizer = CLIPTokenizer.from_pretrained("checkpoints/stable-diffusion-v1-4", subfolder="tokenizer")

    # Check each word
    for word in args.words:
        check_word_in_tokenizer(tokenizer, word)

if __name__ == "__main__":
    main()