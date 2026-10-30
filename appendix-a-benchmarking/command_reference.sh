# gen-data: write a synthetic corpus
cargo run --release -- gen-data [--out data/corpus.txt] [--bytes 120000] [--seed 1]

# train: fit the tiny model, save weights
cargo run --release -- train [--corpus data/corpus.txt] [--out data/tiny.llmw] \
                             [--steps 4000] [--lr 0.001] [--seq-len 48] [--seed 1]

# generate: tokenize -> prefill -> decode
cargo run --release -- generate [--prompt TEXT] [--max-tokens 64] [--temperature 0.8] \
                                [--top-p 0.9] [--seed 42] [--model tiny|gpt2|llama3] \
                                [--weights PATH]

# info: print architecture + param count (from a preset or a trained file)
cargo run --release -- info [--model tiny|gpt2|llama3] [--weights PATH]
