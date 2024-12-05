## BinDash: Binwise Densified Minhash

One Permutation MinHash with Optimal/Faster Densification in Rust

## Install
```bash
git clone https://github.com/jianshu93/bindash-rs
cd bindash-rs
cargo build --release
./target/release/bindash -h

```

## Usage
```bash


 ************** initializing logger *****************

Binwise Densifed MinHash for Genome/Metagenome/Pangenome Comparisons

Usage: bindash [OPTIONS] --query_list <QUERY_LIST_FILE> --reference_list <REFERENCE_LIST_FILE>

Options:
  -q, --query_list <QUERY_LIST_FILE>
          Query genome list file (one FASTA/FNA file path per line)
  -r, --reference_list <REFERENCE_LIST_FILE>
          Reference genome list file (one FASTA/FNA file path per line)
  -k, --kmer_size <KMER_SIZE>
          K-mer size [default: 16]
  -s, --sketch_size <SKETCH_SIZE>
          Sketch size [default: 2048]
  -t, --threads <THREADS>
          Number of threads [default: 1]
  -o, --output <OUTPUT_FILE>
          Output file (defaults to stdout)
  -h, --help
          Print help
  -V, --version
          Print version

```


