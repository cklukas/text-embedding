"""calculate and save embeddings for text files (txt,md,java,pas,c,cc,go) in a directory and subdirectories"""
#
# Text-Embedding demo / test
# (c) 2023 by C. Klukas
#
# see https://huggingface.co/thenlper/gte-large for details on the used embedding model
#
# license: MIT license
#
# calculate embeddings:
#    python file_embedding.py [directory] embeddings.txt
#
# search embeddings:
#   python file_embedding.py search "search text"
#   output:
# load stored embeddings
# instantiate tokenizer and model
# calculate text embedding
# perform search
# 0.894: /dir/file.md,2,6,142
# 0.882: /dir/file.md,4,7,184
#
# ^ [1]  ^ [2]        ^[3,4,5]
# [1] cosine similarity
# [2] file path
# [3] heading index (heading index, 0 means before first heading)
# [4] paragraph index (paragraph # after heading index)
# [5] line index (line number after hit)
#
# paragraphs in the input are detected as being separated by empty lines
# headings are lines starting with # (markdown files)
# non-markdown files are treated as if they had no headings (thus heading # is always 0)
#
# embeddings.txt file has the following format:
# file_path,heading_index,paragraph_index,line_index,embedding
#
# embedding is a comma-separated list of floats (1024 floats)
#
import os
import sys
import argparse
from typing import Any, List, Dict, Tuple, Optional

import tqdm
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
import numpy as np

def average_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Average the hidden states according to the attention mask"""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def text_to_embedding(tokenizer: AutoTokenizer, model: AutoModel, text: str, chunk_size: int = 512) -> torch.Tensor:
    """Tokenize the input text"""
    tokens = tokenizer(text, padding=True, truncation=False, return_tensors='pt')

    embeddings_list = []

    for i in range(0, tokens['input_ids'].shape[1], chunk_size):
        chunk = {
            key: value[:, i:i+chunk_size] for key, value in tokens.items()
        }

        outputs = model(**chunk)
        embeddings = average_pool(outputs.last_hidden_state, chunk['attention_mask'])

        # Normalize embeddings
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1).detach()
        embeddings_list.append(normalized_embeddings)

    # Combine all chunk embeddings into a single vector by averaging
    combined_embedding = torch.mean(torch.stack(embeddings_list), dim=0)
    return combined_embedding


def get_tokenizer_and_model(tokenizer: Optional[AutoTokenizer] = None, model: Optional[AutoModel] = None) \
    -> Tuple[AutoTokenizer, AutoModel]:
    """see https://huggingface.co/thenlper/gte-large"""
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-large")

    if model is None:
        model = AutoModel.from_pretrained("thenlper/gte-large")

    return tokenizer, model


def process_file(file_path: str, tokenizer: Optional[AutoTokenizer] = None, model: Optional[AutoModel] = None, \
                 max_length: int = 512) \
    -> Tuple[Any, Any, Dict[int, Dict[int, Tuple[int, List[float]]]]]:
    """process a single file"""
    tokenizer, model = get_tokenizer_and_model(tokenizer, model)
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        # Process the text
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='ISO-8859-1') as file:
            text = file.read()

    if file_path.endswith(".md"):
        heading2paragraphs = parse_markdown_headings_and_paragraphs(text, detect_headings=True)
    else:
        heading2paragraphs = parse_markdown_headings_and_paragraphs(text, detect_headings=False)

    idx_heading_2_idx_paragraph_2_embedding = {}
    idx_heading = 0
    for _, paragraphs_with_line_idx in heading2paragraphs.items():
        idx_paragraph = 0
        for lineidx,paragraph in paragraphs_with_line_idx:
            emb = text_to_embedding(tokenizer, model, paragraph, max_length)
            emb = emb.detach().numpy().tolist()[0]

            if idx_heading not in idx_heading_2_idx_paragraph_2_embedding:
                idx_heading_2_idx_paragraph_2_embedding[idx_heading] = {}

            idx_heading_2_idx_paragraph_2_embedding[idx_heading][idx_paragraph] = (lineidx,emb)
            idx_paragraph = idx_paragraph + 1

        idx_heading = idx_heading + 1

    return tokenizer, model, idx_heading_2_idx_paragraph_2_embedding


def process_dir(directory: str, max_length: int, extensions: str, out_file: str) -> None:
    """process all files in a directory"""
    valid_extensions = extensions.split(',')
    tokenizer, model = get_tokenizer_and_model(None, None)
    todo_files = []
    n = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith("."+ext.lower()) for ext in valid_extensions):
                file_path = os.path.join(root, file)
                todo_files.append(file_path)

    with open(out_file, 'w', encoding='utf-8') as outf:
        for file_path in tqdm.tqdm(todo_files):
            try:
                tokenizer, model, idx_heading_2_idx_paragraph_2_embedding = \
                    process_file(file_path, tokenizer, model, max_length)

                for idx_heading, idx_paragraph_2_embedding in idx_heading_2_idx_paragraph_2_embedding.items():
                    for idx_paragraph, embedding_with_line_idx in idx_paragraph_2_embedding.items():
                        idx_line, embedding = embedding_with_line_idx
                        vals = embedding
                        valss = ",".join([f"{v}" for v in vals])
                        print(file_path, idx_heading, idx_paragraph, idx_line, valss, sep=",", file=outf)
                n = n + 1
                if n>3:
                    model = None
            except Exception as e:
                print(f"\nerror, file '{file_path} could not be processed: {e}", file=sys.stderr)
                model = None


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate the cosine similarity between two vectors.
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def sort_files_by_similarity(vector_y: List[float], embeddings: Dict[str, List[float]]) -> List[Tuple[str, float]]:
    """
    Sorts filenames based on their embeddings' cosine similarity to a given vector.
    """
    # Calculate similarity scores for each file
    similarity_scores = [(filename, cosine_similarity(vector_y, embedding)) \
                         for filename, embedding in embeddings.items()]

    # Sort the files based on similarity scores
    sorted_files = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    return sorted_files


def load_embeddings(file_path: str) -> Dict[str, List[float]]:
    """load embeddings from file"""
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) > 1:
                filename = parts[0].strip()
                idx_heading = int(parts[1].strip())
                idx_paragraph = int(parts[2].strip())
                idx_line = int(parts[3].strip())
                vector = [float(val) for val in parts[4:]]
                embeddings[filename+","+str(idx_heading)+","+str(idx_paragraph)+","+str(idx_line)] = vector

    return embeddings

def search_file(text: str) -> None:
    """search files by text embedding similarity"""
    print("load stored embeddings")
    emb_dict = load_embeddings("embeddings.txt")

    print("instantiate tokenizer and model")
    tokenizer, model = get_tokenizer_and_model(None, None)

    print("calculate text embedding")
    search_emb = text_to_embedding(tokenizer, model, text)

    print("perform search")
    res = sort_files_by_similarity(search_emb, emb_dict)

    n = 0
    for k,v in res:
        v = v[0]
        print(f"{v:.3f}: {k}")
        n = n + 1
        if n>=5:
            break


def parse_markdown_headings_and_paragraphs(markdown_text: str, detect_headings: bool = True) \
    -> Dict[str, List[Tuple[int, str]]]:
    """Returns a dictionary of headings and their corresponding paragraphs"""
    lines = markdown_text.split('\n')

    # Dictionary to store headings and their corresponding paragraphs
    headings = {}
    current_heading = ""  # Virtual heading for the start of the document
    current_paragraph = []
    idx_line = 0
    for line in lines:
        idx_line = idx_line + 1
        # Check if the line is a heading
        if line.startswith('#') and detect_headings:
            # Save the current heading and its paragraphs
            if current_heading is not None and current_paragraph:
                paragraph_text = ' '.join(current_paragraph).strip()
                headings.setdefault(current_heading, []).append((idx_line,paragraph_text))
            # Reset for the new heading
            current_heading = line.strip('# ').strip()
            current_paragraph = []
        else:
            # Check if the line is empty (indicating the end of a paragraph)
            if line.strip() == '':
                if current_paragraph:
                    paragraph_text = ' '.join(current_paragraph).strip()
                    headings.setdefault(current_heading, []).append((idx_line,paragraph_text))
                    current_paragraph = []
            else:
                # Non-empty lines are part of the current paragraph
                current_paragraph.append(line)

    # Add the last paragraph if it exists
    if current_paragraph:
        paragraph_text = ' '.join(current_paragraph).strip()
        headings.setdefault(current_heading, []).append((idx_line,paragraph_text))

    return headings


def main():
    """main function"""
    parser = argparse.ArgumentParser(description="Process files and print embeddings.")
    subparsers = parser.add_subparsers(dest='command', help='Sub-command help')

    # Subparser for the "process-file" command
    parser_file = subparsers.add_parser('process-file', \
                                        help='Process a single file')
    parser_file.add_argument("file_path", \
                             help="Path to the input file to process")
    parser_file.add_argument("--max_length", type=int, default=512, \
                             help="Maximum token length for each chunk")

    # Subparser for the "process-dir" command
    parser_dir = subparsers.add_parser('process-dir', \
                                       help='Process all files in a directory')
    parser_dir.add_argument("dir_path", \
                            help="Path to the directory to process")
    parser_dir.add_argument("out_file", \
                            help="Path to the output file")
    parser_dir.add_argument("--max_length", type=int, default=512, \
                            help="Maximum token length for each chunk")
    parser_dir.add_argument("--extensions", default="txt,md,java,pas,c,cc,go", \
                            help="Comma-separated list of valid file extensions (e.g., 'txt,md')")

    parser_search = subparsers.add_parser("search", help="Search files by text embedding similarity")
    parser_search.add_argument("text", help="The search text")

    args = parser.parse_args()

    if args.command == "process-file":
        _, _, idx_heading_2_idx_paragraph_2_embedding = process_file(None, None, args.file_path, args.max_length)
        for idx_heading, idx_paragraph_2_embedding in idx_heading_2_idx_paragraph_2_embedding.items():
            for idx_paragraph, embedding in idx_paragraph_2_embedding.items():
                vals = embedding
                valss = ",".join([f"{v}" for v in vals])
                print(args.file_path, idx_heading, idx_paragraph, valss, sep=",")

    elif args.command == "process-dir":
        # python file_embedding.py process-dir /Volumes/S8T embeddings.txt
        process_dir(args.dir_path, args.max_length, args.extensions, args.out_file)

    elif args.command == "search":
        search_file(args.text)


if __name__ == "__main__":
    main()
