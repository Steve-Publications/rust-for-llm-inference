# 1. Write a synthetic English corpus to data/corpus.txt (~120 KB).
cargo run --release -- gen-data

# 2. Train the tiny model on that corpus -> data/tiny.llmw (~50s on CPU).
#    (train auto-generates the corpus if data/corpus.txt is missing.)
cargo run --release -- train --steps 2000

# 3. Generate REAL text from the trained weights (greedy = deterministic).
cargo run --release -- generate --weights data/tiny.llmw --prompt "the" --temperature 0
