use clap::{Arg, ArgAction, Command};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use needletail::{Sequence,parse_fastx_file};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use kmerutils::sketcharg::{SeqSketcherParams, SketchAlgo};
use kmerutils::base::{kmergenerator::*, Kmer32bit, Kmer16b32bit, Kmer64bit, CompressedKmerT};
use kmerutils::sketching::setsketchert::*;
use kmerutils::sketcharg::DataType;
use kmerutils::base::alphabet::Alphabet2b;
use kmerutils::base::sequence::Sequence as SequenceStruct;
use anndists::dist::Distance;
use anndists::dist::DistHamming;
use num;

fn ascii_to_seq(bases: &[u8]) -> Result<SequenceStruct, ()> {
    let alphabet = Alphabet2b::new();
    let mut seq = SequenceStruct::with_capacity(2, bases.len());
    seq.encode_and_add(bases, &alphabet);
    Ok(seq)
} // end of ascii_to_seq

fn main() {
        // Initialize logger
    println!("\n ************** initializing logger *****************\n");
    let _ = env_logger::Builder::from_default_env().init();

    let matches = Command::new("BinDash")
        .version("0.1.0")
        .about("Binwise Densifed MinHash for Genome/Metagenome/Pangenome Comparisons")
        .arg(
            Arg::new("query_list")
                .short('q')
                .long("query_list")
                .value_name("QUERY_LIST_FILE")
                .help("Query genome list file (one FASTA/FNA file path per line)")
                .required(true)
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("reference_list")
                .short('r')
                .long("reference_list")
                .value_name("REFERENCE_LIST_FILE")
                .help("Reference genome list file (one FASTA/FNA file path per line)")
                .required(true)
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("kmer_size")
                .short('k')
                .long("kmer_size")
                .value_name("KMER_SIZE")
                .help("K-mer size")
                .default_value("16")
                .value_parser(clap::value_parser!(usize))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("sketch_size")
                .short('s')
                .long("sketch_size")
                .value_name("SKETCH_SIZE")
                .help("Sketch size")
                .default_value("2048")
                .value_parser(clap::value_parser!(usize))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("threads")
                .short('t')
                .long("threads")
                .value_name("THREADS")
                .help("Number of threads")
                .default_value("1")
                .value_parser(clap::value_parser!(usize))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .value_name("OUTPUT_FILE")
                .help("Output file (defaults to stdout)")
                .required(true)
                .action(ArgAction::Set),
        )
        .get_matches();

    let query_list = matches
        .get_one::<String>("query_list")
        .expect("Query list is required")
        .to_string();
    let reference_list = matches
        .get_one::<String>("reference_list")
        .expect("Reference list is required")
        .to_string();
    let kmer_size = *matches.get_one::<usize>("kmer_size").unwrap();
    let sketch_size = *matches.get_one::<usize>("sketch_size").unwrap();
    let threads = *matches.get_one::<usize>("threads").unwrap();
    let output = matches.get_one::<String>("output").cloned();

    ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();

    let query_genomes = {
        let file = File::open(&query_list).expect("Cannot open query genome list file");
        let reader = BufReader::new(file);
        reader
            .lines()
            .map(|line| line.expect("Error reading query genome list"))
            .collect::<Vec<String>>()
    };

    let reference_genomes = {
        let file = File::open(&reference_list).expect("Cannot open reference genome list file");
        let reader = BufReader::new(file);
        reader
            .lines()
            .map(|line| line.expect("Error reading reference genome list"))
            .collect::<Vec<String>>()
    };

    let sketch_args = SeqSketcherParams::new(
        kmer_size,
        sketch_size,
        SketchAlgo::OPTDENS,
        DataType::DNA,
    );

    if kmer_size <= 14 {
        let sketcher = OptDensHashSketch::<Kmer32bit, f32>::new(&sketch_args);
        let nb_alphabet_bits = 2;
        let kmer_hash_fn_32bit = | kmer : &Kmer32bit | -> <Kmer32bit as CompressedKmerT>::Val {
            let mask : <Kmer32bit as CompressedKmerT>::Val = num::NumCast::from::<u64>((0b1 << nb_alphabet_bits*kmer.get_nb_base()) - 1).unwrap();
            let hashval = kmer.get_compressed_value() & mask;
            hashval
        };

        println!("Sketching query genomes...");
        let query_sketches = {
            let sketches = query_genomes
                .par_iter()
                .map(|path| {
                    println!("Sketching genome: {}", path);
                    let mut sequences = Vec::new();
                    let mut reader = parse_fastx_file(path).expect("Invalid FASTA/Q file");
                    while let Some(record) = reader.next() {
                        let seq_record = record.expect("Error reading sequence record");
                        let seq_seq = seq_record.normalize(false).into_owned();
                        let seq = ascii_to_seq(&seq_seq).unwrap();
                        sequences.push(seq);
                    }
                    let sequences_ref: Vec<&SequenceStruct> = sequences.iter().collect();
                    let signature = sketcher.sketch_compressedkmer_seqs(&sequences_ref, &kmer_hash_fn_32bit);
                    (path.clone(), signature[0].clone())
                })
                .collect::<HashMap<String, Vec<f32>>>();
            sketches
        };

        println!("Sketching reference genomes...");
        let reference_sketches = {
            let sketches = reference_genomes
                .par_iter()
                .map(|path| {
                    println!("Sketching genome: {}", path);
                    let mut sequences = Vec::new();
                    let mut reader = parse_fastx_file(path).expect("Invalid FASTA/Q file");
                    while let Some(record) = reader.next() {
                        let seq_record = record.expect("Error reading sequence record");
                        let seq_seq = seq_record.normalize(false).into_owned();
                        let seq = ascii_to_seq(&seq_seq).unwrap();
                        sequences.push(seq);
                    }
                    let sequences_ref: Vec<&SequenceStruct> = sequences.iter().collect();
                    let signature = sketcher.sketch_compressedkmer_seqs(&sequences_ref, &kmer_hash_fn_32bit);
                    (path.clone(), signature[0].clone())
                })
                .collect::<HashMap<String, Vec<f32>>>();
            sketches
        };

        let writer: Box<dyn Write + Send> = match output {
            Some(filename) => {
                let file = File::create(filename).expect("Cannot create output file");
                Box::new(BufWriter::new(file))
            }
            None => Box::new(BufWriter::new(io::stdout())),
        };

        let writer = Arc::new(Mutex::new(writer));

        println!("Performing pairwise comparisons...");
        query_genomes.par_iter().for_each(|query_path| {
            let query_signature = &query_sketches[query_path];

            for reference_path in &reference_genomes {
                let reference_signature = &reference_sketches[reference_path];

                let dist_hamming = DistHamming;
                let hamming_distance = dist_hamming.eval(&query_signature, &reference_signature);
                let j = 1.0 - hamming_distance;

                let numerator = 2.0 * j;
                let denominator = 1.0 + j;
                let fraction: f32 = numerator / denominator;
                let distance = -fraction.ln() / (kmer_size as f32);

                let mut writer = writer.lock().unwrap();
                writeln!(
                    writer,
                    "{}\t{}\t{}",
                    query_path, reference_path, distance
                )
                .expect("Error writing output");
            }
        });

    } else if kmer_size == 16 {
        let sketcher = OptDensHashSketch::<Kmer16b32bit, f32>::new(&sketch_args);
        let nb_alphabet_bits = 2;
        let kmer_hash_fn_16b32bit = | kmer : &Kmer16b32bit | -> <Kmer16b32bit as CompressedKmerT>::Val {
            let canonical =  kmer.reverse_complement().min(*kmer);
            let mask : <Kmer16b32bit as CompressedKmerT>::Val = num::NumCast::from::<u64>((0b1 << nb_alphabet_bits*kmer.get_nb_base()) - 1).unwrap();
            let hashval = canonical.get_compressed_value() & mask;
            hashval
        };

        println!("Sketching query genomes...");
        let query_sketches = {
            let sketches = query_genomes
                .par_iter()
                .map(|path| {
                    println!("Sketching genome: {}", path);
                    let mut sequences = Vec::new();
                    let mut reader = parse_fastx_file(path).expect("Invalid FASTA/Q file");
                    while let Some(record) = reader.next() {
                        let seq_record = record.expect("Error reading sequence record");
                        let seq_seq = seq_record.normalize(false).into_owned();
                        let seq = ascii_to_seq(&seq_seq).unwrap();
                        sequences.push(seq);
                    }
                    let sequences_ref: Vec<&SequenceStruct> = sequences.iter().collect();
                    let signature = sketcher.sketch_compressedkmer_seqs(&sequences_ref, &kmer_hash_fn_16b32bit);
                    (path.clone(), signature[0].clone())
                })
                .collect::<HashMap<String, Vec<f32>>>();
            sketches
        };

        println!("Sketching reference genomes...");
        let reference_sketches = {
            let sketches = reference_genomes
                .par_iter()
                .map(|path| {
                    println!("Sketching genome: {}", path);
                    let mut sequences = Vec::new();
                    let mut reader = parse_fastx_file(path).expect("Invalid FASTA/Q file");
                    while let Some(record) = reader.next() {
                        let seq_record = record.expect("Error reading sequence record");
                        let seq_seq = seq_record.normalize(false).into_owned();
                        let seq = ascii_to_seq(&seq_seq).unwrap();
                        sequences.push(seq);
                    }
                    let sequences_ref: Vec<&SequenceStruct> = sequences.iter().collect();
                    let signature = sketcher.sketch_compressedkmer_seqs(&sequences_ref, &kmer_hash_fn_16b32bit);
                    (path.clone(), signature[0].clone())
                })
                .collect::<HashMap<String, Vec<f32>>>();
            sketches
        };

        let writer: Box<dyn Write + Send> = match output {
            Some(filename) => {
                let file = File::create(filename).expect("Cannot create output file");
                Box::new(BufWriter::new(file))
            }
            None => Box::new(BufWriter::new(io::stdout())),
        };

        let writer = Arc::new(Mutex::new(writer));

        println!("Performing pairwise comparisons...");
        query_genomes.par_iter().for_each(|query_path| {
            let query_signature = &query_sketches[query_path];

            for reference_path in &reference_genomes {
                let reference_signature = &reference_sketches[reference_path];

                let dist_hamming = DistHamming;
                let hamming_distance = dist_hamming.eval(&query_signature, &reference_signature);
                let j = 1.0 - hamming_distance;

                let numerator = 2.0 * j;
                let denominator = 1.0 + j;
                let fraction: f32  = numerator / denominator;
                let distance = -fraction.ln() / (kmer_size as f32);

                let mut writer = writer.lock().unwrap();
                writeln!(
                    writer,
                    "{}\t{}\t{}",
                    query_path, reference_path, distance
                )
                .expect("Error writing output");
            }
        });

    } else if kmer_size <= 32 {
        let sketcher = OptDensHashSketch::<Kmer64bit, f32>::new(&sketch_args);
        let nb_alphabet_bits = 2;
        let kmer_hash_fn_64bit = | kmer : &Kmer64bit | -> <Kmer64bit as CompressedKmerT>::Val {
            let canonical =  kmer.reverse_complement().min(*kmer);
            let mask : <Kmer64bit as CompressedKmerT>::Val = num::NumCast::from::<u64>((0b1 << nb_alphabet_bits*kmer.get_nb_base()) - 1).unwrap();
            let hashval = canonical.get_compressed_value() & mask;
            hashval
        };

        println!("Sketching query genomes...");
        let query_sketches = {
            let sketches = query_genomes
                .par_iter()
                .map(|path| {
                    println!("Sketching genome: {}", path);
                    let mut sequences = Vec::new();
                    let mut reader = parse_fastx_file(path).expect("Invalid FASTA/Q file");
                    while let Some(record) = reader.next() {
                        let seq_record = record.expect("Error reading sequence record");
                        let seq_seq = seq_record.normalize(false).into_owned();
                        let seq = ascii_to_seq(&seq_seq).unwrap();
                        sequences.push(seq);
                    }
                    let sequences_ref: Vec<&SequenceStruct> = sequences.iter().collect();
                    let signature = sketcher.sketch_compressedkmer_seqs(&sequences_ref, &kmer_hash_fn_64bit);
                    (path.clone(), signature[0].clone())
                })
                .collect::<HashMap<String, Vec<f32>>>();
            sketches
        };

        println!("Sketching reference genomes...");
        let reference_sketches = {
            let sketches = reference_genomes
                .par_iter()
                .map(|path| {
                    println!("Sketching genome: {}", path);
                    let mut sequences = Vec::new();
                    let mut reader = parse_fastx_file(path).expect("Invalid FASTA/Q file");
                    while let Some(record) = reader.next() {
                        let seq_record = record.expect("Error reading sequence record");
                        let seq_seq = seq_record.normalize(false).into_owned();
                        let seq = ascii_to_seq(&seq_seq).unwrap();
                        sequences.push(seq);
                    }
                    let sequences_ref: Vec<&SequenceStruct> = sequences.iter().collect();
                    let signature = sketcher.sketch_compressedkmer_seqs(&sequences_ref, &kmer_hash_fn_64bit);
                    (path.clone(), signature[0].clone())
                })
                .collect::<HashMap<String, Vec<f32>>>();
            sketches
        };

        let writer: Box<dyn Write + Send> = match output {
            Some(filename) => {
                let file = File::create(filename).expect("Cannot create output file");
                Box::new(BufWriter::new(file))
            }
            None => Box::new(BufWriter::new(io::stdout())),
        };

        let writer = Arc::new(Mutex::new(writer));

        println!("Performing pairwise comparisons...");
        query_genomes.par_iter().for_each(|query_path| {
            let query_signature = &query_sketches[query_path];

            for reference_path in &reference_genomes {
                let reference_signature = &reference_sketches[reference_path];

                let dist_hamming = DistHamming;
                let hamming_distance = dist_hamming.eval(&query_signature, &reference_signature);
                let j = 1.0 - hamming_distance;

                let numerator = 2.0 * j;
                let denominator = 1.0 + j;
                let fraction: f32 = numerator / denominator;
                let distance = - fraction.ln() / (kmer_size as f32);

                let mut writer = writer.lock().unwrap();
                writeln!(
                    writer,
                    "{}\t{}\t{}",
                    query_path, reference_path, distance
                )
                .expect("Error writing output");
            }
        });

    } else {
        panic!("kmers cannot be 15 or greater than 32");
    }
}
