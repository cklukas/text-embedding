# Text-Embedding demo / test

(c) 2023 by C. Klukas

see https://huggingface.co/thenlper/gte-large for details on the used embedding model

license: MIT license

## calculate embeddings

```bash
python file_embedding.py [directory] embeddings.txt
```

## search files by embedding similarity

```bash
python file_embedding.py search "search text"
```
output:

```
load stored embeddings
instantiate tokenizer and model
calculate text embedding
perform search
0.894: /dir/file.md,2,6,142
0.882: /dir/file.md,4,7,184
#
# ^ [1]  ^ [2]        ^[3,4,5]
```

* [1] cosine similarity
* [2] file path
* [3] heading index (heading index, 0 means before first heading)
* [4] paragraph index (paragraph # after heading index)
* [5] line index (line number after hit)

Paragraphs in the input are detected as being separated by empty lines,
headings are lines starting with # (markdown files),
non-markdown files are treated as if they had no headings (thus heading # is always 0).

embeddings.txt file has the following format:
```
file_path,heading_index,paragraph_index,line_index,embedding
```

embedding is a comma-separated list of floats (1024 floats)
