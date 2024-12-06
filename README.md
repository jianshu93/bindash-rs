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
          Query genome list file (one FASTA/FNA file path per line, .gz supported)
  -r, --reference_list <REFERENCE_LIST_FILE>
          Reference genome list file (one FASTA/FNA file path per line, .gz supported)
  -k, --kmer_size <KMER_SIZE>
          K-mer size [default: 16]
  -s, --sketch_size <SKETCH_SIZE>
          MinHash sketch size [default: 2048]
  -d, --densification <DENS_OPT>
          Densification strategy, 0 for optimal densification, 1 for reverse optimal/faster densification [default: 0]
  -t, --threads <THREADS>
          Number of threads to use in parallel [default: 1]
  -o, --output <OUTPUT_FILE>
          Output file (defaults to stdout)
  -h, --help
          Print help
  -V, --version
          Print version
```


## References
1.Li, P., Owen, A. and Zhang, C.H., 2012. One permutation hashing. Advances in Neural Information Processing Systems, 25.
2.Shrivastava, A., 2017, July. Optimal densification for fast and accurate minwise hashing. In International Conference on Machine Learning (pp. 3154-3163). PMLR.
3.Mai, T., Rao, A., Kapilevich, M., Rossi, R., Abbasi-Yadkori, Y. and Sinha, R., 2020, August. On densification for minwise hashing. In Uncertainty in Artificial Intelligence (pp. 831-840). PMLR.


