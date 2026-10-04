[![Latest Version](https://img.shields.io/crates/v/bindash?style=for-the-badge&color=mediumpurple&logo=rust)](https://crates.io/crates/bindash)

## BinDash: Binwise Densified Minhash

One Permutation MinHash with Optimal/Faster Densification in Rust

## Install
### Compile from source

The default build is the CPU implementation:

```bash
git clone https://github.com/jianshu93/bindash-rs
cd bindash-rs
cargo build --release
./target/release/bindash -h
```

On macOS, build the 16-bit Apple Metal distance backend with:

```bash
cargo build --release --features metal --bin bindash-metal
./target/release/bindash-metal -h
```

On Linux with an NVIDIA CUDA toolkit, build the 16-bit CUDA distance backend with:

```bash
cargo build --release --features cuda --bin bindash-cuda
./target/release/bindash-cuda -h
```

## Usage
```bash
************** initializing logger *****************

Binwise Densified MinHash for Genome/Metagenome/Pangenome Comparisons

Usage: bindash [OPTIONS] --query_list <QUERY_LIST_FILE>

Options:
  -q, --query_list <QUERY_LIST_FILE>
          Query genome list file (one FASTA/FNA file path per line, .gz supported)
  -r, --reference_list <REFERENCE_LIST_FILE>
          Reference genome list file (one FASTA/FNA file path per line, .gz supported). If omitted, query_list is reused for self-comparison
  -k, --kmer_size <KMER_SIZE>
          K-mer size [default: 16]
  -s, --sketch_size <SKETCH_SIZE>
          MinHash sketch size [default: 2048]
  -d, --densification <DENS_OPT>
          Densification strategy, 0 = optimal densification, 1 = reverse optimal/faster densification [default: 0]
  -T, --threads <THREADS>
          Number of threads, default all logical cores
      --matrix
          Write dense rectangular matrix output
  -o, --output <OUTPUT_FILE>
          Output file (zstd-compressed by default)
  -h, --help
          Print help
  -V, --version
          Print version
```


## References
1.Li, P., Owen, A. and Zhang, C.H., 2012. One permutation hashing. Advances in Neural Information Processing Systems, 25.

2.Shrivastava, A., 2017, July. Optimal densification for fast and accurate minwise hashing. In International Conference on Machine Learning (pp. 3154-3163). PMLR.

3.Mai, T., Rao, A., Kapilevich, M., Rossi, R., Abbasi-Yadkori, Y. and Sinha, R., 2020, August. On densification for minwise hashing. In Uncertainty in Artificial Intelligence (pp. 831-840). PMLR.
