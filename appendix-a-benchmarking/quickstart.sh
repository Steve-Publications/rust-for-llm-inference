cargo test                                    # 43 tests: unit + integration + gradient checks
cargo build --release                         # build the `llm-rs` binary
cargo run --release -- info --model llama3    # print architecture + param count
