use clap::{Arg, ArgAction, Command};
use env_logger;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::collections::HashMap;
use std::fmt::Debug;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use needletail::{parse_fastx_file, Sequence};
use num;

use kmerutils::base::{
    alphabet::Alphabet2b,
    kmergenerator::*,
    sequence::Sequence as SequenceStruct,
    CompressedKmerT,
    Kmer16b32bit,
    Kmer32bit,
    Kmer64bit,
    KmerBuilder,
};
use kmerutils::sketcharg::{DataType, SeqSketcherParams, SketchAlgo};
use kmerutils::sketching::setsketchert::{
    OptDensHashSketch,
    RevOptDensHashSketch,
    SeqSketcherT, // trait
};
use ryu;
use zstd;
use std::time::Instant;

#[cfg(not(feature = "cuda"))]
use anndists::dist::{Distance, DistHamming};

#[cfg(feature = "cuda")]
mod disthamming_gpu;

#[cfg(feature = "cuda")]
use disthamming_gpu::pairwise_hamming_rect_multi_gpu_u16;

/// Converts ASCII-encoded bases (from Needletail) into our `SequenceStruct`.
fn ascii_to_seq(bases: &[u8]) -> Result<SequenceStruct, ()> {
    let alphabet = Alphabet2b::new();
    let mut seq = SequenceStruct::with_capacity(2, bases.len());
    seq.encode_and_add(bases, &alphabet);
    Ok(seq)
}

/// Reads a list of file paths (one per line) from a text file.
fn read_genome_list(filepath: &str) -> Vec<String> {
    let file = File::open(filepath).expect("Cannot open genome list file");
    let reader = BufReader::new(file);
    reader
        .lines()
        .map(|line| line.expect("Error reading genome list"))
        .collect()
}


fn sketch_files<Kmer, Sketcher, F>(
    file_paths: &[String],
    sketcher: &Sketcher,
    kmer_hash_fn: F,
) -> HashMap<String, Vec<<Sketcher as SeqSketcherT<Kmer>>::Sig>>
where
    Kmer: CompressedKmerT + KmerBuilder<Kmer> + Send + Sync,
    <Kmer as CompressedKmerT>::Val: num::PrimInt + Send + Sync + Debug,
    KmerGenerator<Kmer>: KmerGenerationPattern<Kmer>,
    Sketcher: SeqSketcherT<Kmer> + Sync,
    <Sketcher as SeqSketcherT<Kmer>>::Sig: Send + Sync + Clone,
    F: Fn(&Kmer) -> <Kmer as CompressedKmerT>::Val + Send + Sync + Copy,
{
    let t0 = Instant::now();
    log::info!(
        "sketch_files: starting sketch of {} files with k={} sketch_size={} algo={:?}",
        file_paths.len(),
        sketcher.get_kmer_size(),
        sketcher.get_sketch_size(),
        sketcher.get_algo()
    );

    let out: HashMap<String, Vec<<Sketcher as SeqSketcherT<Kmer>>::Sig>> = file_paths
        .par_iter()
        .map(|path| {
            let file_t0 = Instant::now();
            log::debug!("sketch_files: start {}", path);

            let mut sequences = Vec::new();
            let mut reader = parse_fastx_file(path).expect("Invalid FASTA/Q file");

            while let Some(record) = reader.next() {
                let seq_record = record.expect("Error reading sequence record");
                let seq_seq = seq_record.normalize(false).into_owned();
                let seq = ascii_to_seq(&seq_seq).unwrap();
                sequences.push(seq);
            }

            let sequences_ref: Vec<&SequenceStruct> = sequences.iter().collect();

            let signature = sketcher.sketch_compressedkmer_seqs(&sequences_ref, kmer_hash_fn);

            log::debug!(
                "sketch_files: finished {} in {:.3}s",
                path,
                file_t0.elapsed().as_secs_f64()
            );

            (path.clone(), signature[0].clone())
        })
        .collect();

    log::info!(
        "sketch_files: finished {} files in {:.3}s",
        file_paths.len(),
        t0.elapsed().as_secs_f64()
    );

    out
}

/// Computes the distance between two sketches (as `Vec<u64>`) using DistHamming,
/// then applies your transformation:
///
///    distance = -ln( (2*j) / (1 + j) ) / kmer_size
///
/// where j = 1 - hamming_distance (assuming DistHamming returns normalized distance).
fn compute_distance_from_hamming(h: f64, kmer_size: usize) -> f64 {
    let h = if h <= 0.0 { 0.0 } else if h >= 1.0 { 1.0 } else { h };

    let mut j = 1.0 - h;
    if j <= 0.0 {
        j = f64::MIN_POSITIVE;
    }

    let fraction = (2.0 * j) / (1.0 + j);
    -fraction.ln() / (kmer_size as f64)
}

#[cfg(not(feature = "cuda"))]
fn compute_distance<Sig>(query_sig: &[Sig], reference_sig: &[Sig], kmer_size: usize) -> f64
where
    Sig: Send + Sync,
    DistHamming: Distance<Sig>,
{
    let dist_hamming = DistHamming;
    let h: f64 = dist_hamming.eval(query_sig, reference_sig) as f64;
    compute_distance_from_hamming(h, kmer_size)
}


fn make_zstd_output_writer(output: Option<String>) -> Box<dyn Write> {
    match output {
        Some(filename) => {
            let file = File::create(&filename).expect("Cannot create output file");
            let mut enc = zstd::Encoder::new(file, 0).expect("Cannot create zstd encoder");
            let zstd_threads = rayon::current_num_threads() as u32;
            if zstd_threads > 1 {
                enc.multithread(zstd_threads)
                    .expect("Cannot enable zstd multithreading");
            }
            Box::new(BufWriter::with_capacity(16 << 20, enc.auto_finish()))
        }
        None => {
            panic!("Refusing to write zstd-compressed output to stdout; please provide -o");
        }
    }
}


#[cfg(not(feature = "cuda"))]
fn write_results<Sig>(
    output: Option<String>,
    matrix_output: bool,
    query_genomes: &[String],
    reference_genomes: &[String],
    query_sketches: &HashMap<String, Vec<Sig>>,
    reference_sketches: &HashMap<String, Vec<Sig>>,
    kmer_size: usize,
) where
    Sig: Send + Sync + Clone,
    DistHamming: Distance<Sig>,
{
    let total_t0 = Instant::now();

    log::info!(
        "CPU distance: starting pairwise comparisons for {} query x {} reference = {} pairs",
        query_genomes.len(),
        reference_genomes.len(),
        query_genomes.len() * reference_genomes.len()
    );

    let mut output_writer = make_zstd_output_writer(output);

    let nq = query_genomes.len();
    let nr = reference_genomes.len();

    let block_rows = ((nq as f64).sqrt() as usize).max(1);
    log::info!(
        "CPU distance: block-wise zstd writing with block_rows = {} (nq = {}, matrix_output = {})",
        block_rows,
        nq,
        matrix_output
    );

    let total_pairs = nq * nr;
    let hamming_t0 = Instant::now();

    let raw_results: Vec<Vec<f32>> = (0..nq)
        .into_par_iter()
        .map(|qi| {
            let q_path = &query_genomes[qi];
            let query_signature = &query_sketches[q_path];
            let mut row = Vec::with_capacity(nr);

            for r_path in reference_genomes {
                let reference_signature = &reference_sketches[r_path];
                let dist_hamming = DistHamming;
                let h = dist_hamming.eval(query_signature, reference_signature) as f32;
                row.push(h);
            }

            row
        })
        .collect();

    log::info!(
        "CPU distance: raw Hamming stage finished in {:.3}s for {} pairs",
        hamming_t0.elapsed().as_secs_f64(),
        total_pairs
    );

    let write_t0 = Instant::now();

    if matrix_output {
        let mut header = String::with_capacity(nr * 32);
        header.push_str("ID");
        for r_path in reference_genomes {
            header.push('\t');
            header.push_str(r_path);
        }
        header.push('\n');
        output_writer
            .write_all(header.as_bytes())
            .expect("Error writing matrix header");

        let mut q0 = 0usize;
        while q0 < nq {
            let q1 = (q0 + block_rows).min(nq);

            let lines: Vec<String> = (q0..q1)
                .into_par_iter()
                .map(|qi| {
                    let q_path = &query_genomes[qi];
                    let query_basename = Path::new(q_path)
                        .file_name()
                        .and_then(|os_str| os_str.to_str())
                        .unwrap_or(q_path);

                    let mut line = String::with_capacity(nr * 16 + q_path.len() + 1);
                    let row = &raw_results[qi];
                    let mut fmt = ryu::Buffer::new();

                    line.push_str(q_path);

                    for (ri, r_path) in reference_genomes.iter().enumerate() {
                        let reference_basename = Path::new(r_path)
                            .file_name()
                            .and_then(|os_str| os_str.to_str())
                            .unwrap_or(r_path);

                        let mut distance =
                            compute_distance_from_hamming(row[ri] as f64, kmer_size);

                        if query_basename == reference_basename {
                            distance = 0.0;
                        }

                        line.push('\t');
                        line.push_str(fmt.format_finite(distance));
                    }

                    line.push('\n');
                    line
                })
                .collect();

            for line in &lines {
                output_writer
                    .write_all(line.as_bytes())
                    .expect("Error writing matrix row block");
            }

            q0 = q1;
        }
    } else {
        writeln!(output_writer, "Query\tReference\tDistance").expect("Error writing header");

        let mut q0 = 0usize;
        while q0 < nq {
            let q1 = (q0 + block_rows).min(nq);

            let lines: Vec<String> = (q0..q1)
                .into_par_iter()
                .map(|qi| {
                    let q_path = &query_genomes[qi];
                    let query_basename = Path::new(q_path)
                        .file_name()
                        .and_then(|os_str| os_str.to_str())
                        .unwrap_or(q_path);

                    let mut line_block = String::with_capacity(nr * 64);
                    let row = &raw_results[qi];
                    let mut fmt = ryu::Buffer::new();

                    for (ri, r_path) in reference_genomes.iter().enumerate() {
                        let reference_basename = Path::new(r_path)
                            .file_name()
                            .and_then(|os_str| os_str.to_str())
                            .unwrap_or(r_path);

                        let mut distance =
                            compute_distance_from_hamming(row[ri] as f64, kmer_size);

                        if query_basename == reference_basename {
                            distance = 0.0;
                        }

                        line_block.push_str(q_path);
                        line_block.push('\t');
                        line_block.push_str(r_path);
                        line_block.push('\t');
                        line_block.push_str(fmt.format_finite(distance));
                        line_block.push('\n');
                    }

                    line_block
                })
                .collect();

            for line in &lines {
                output_writer
                    .write_all(line.as_bytes())
                    .expect("Error writing output block");
            }

            q0 = q1;
        }
    }

    output_writer.flush().expect("Error flushing output");

    log::info!(
        "CPU distance: transform + zstd write stage finished in {:.3}s",
        write_t0.elapsed().as_secs_f64()
    );
    log::info!(
        "CPU distance: total compute + zstd write completed in {:.3}s",
        total_t0.elapsed().as_secs_f64()
    );
}

#[cfg(feature = "cuda")]
fn write_results_gpu_u16(
    output: Option<String>,
    matrix_output: bool,
    query_genomes: &[String],
    reference_genomes: &[String],
    query_sketches: &HashMap<String, Vec<u16>>,
    reference_sketches: &HashMap<String, Vec<u16>>,
    kmer_size: usize,
) {
    let total_t0 = Instant::now();
    log::info!(
        "GPU distance: starting pairwise comparisons for {} query x {} reference = {} pairs",
        query_genomes.len(),
        reference_genomes.len(),
        query_genomes.len() * reference_genomes.len()
    );

    let mut output_writer = make_zstd_output_writer(output);

    if query_genomes.is_empty() || reference_genomes.is_empty() {
        log::info!("GPU distance: empty query or reference set, nothing to do");
        return;
    }

    let k = query_sketches[&query_genomes[0]].len();

    for q in query_genomes {
        let len = query_sketches[q].len();
        assert!(
            len == k,
            "Query sketch length mismatch for {}: got {}, expected {}",
            q,
            len,
            k
        );
    }

    for r in reference_genomes {
        let len = reference_sketches[r].len();
        assert!(
            len == k,
            "Reference sketch length mismatch for {}: got {}, expected {}",
            r,
            len,
            k
        );
    }

    let nq = query_genomes.len();
    let nr = reference_genomes.len();

    let flatten_t0 = Instant::now();
    let mut query_flat = Vec::<u16>::with_capacity(nq * k);
    for q in query_genomes {
        query_flat.extend_from_slice(&query_sketches[q]);
    }

    let mut reference_flat = Vec::<u16>::with_capacity(nr * k);
    for r in reference_genomes {
        reference_flat.extend_from_slice(&reference_sketches[r]);
    }

    log::info!(
        "GPU distance: flatten stage finished in {:.3}s (nq={} nr={} k={})",
        flatten_t0.elapsed().as_secs_f64(),
        nq,
        nr,
        k
    );

    let mut hamming_rect = vec![0.0f32; nq * nr];

    let block_cols = 2048usize.min(nr.max(1));
    let block_rows_gpu = 2048usize.min(nq.max(1));

    let hamming_t0 = Instant::now();
    pairwise_hamming_rect_multi_gpu_u16(
        &query_flat,
        nq,
        &reference_flat,
        nr,
        k,
        &mut hamming_rect,
        block_rows_gpu,
        block_cols,
    )
    .expect("GPU rectangular Hamming failed");

    log::info!(
        "GPU distance: raw GPU Hamming stage finished in {:.3}s",
        hamming_t0.elapsed().as_secs_f64()
    );

    let write_t0 = Instant::now();

    let block_rows_write = ((nq as f64).sqrt() as usize).max(1);
    log::info!(
        "GPU distance: block-wise zstd writing with block_rows = {} (nq = {}, matrix_output = {})",
        block_rows_write,
        nq,
        matrix_output
    );

    if matrix_output {
        let mut header = String::with_capacity(nr * 32);
        header.push_str("ID");
        for r_path in reference_genomes {
            header.push('\t');
            header.push_str(r_path);
        }
        header.push('\n');
        output_writer
            .write_all(header.as_bytes())
            .expect("Error writing matrix header");

        let mut q0 = 0usize;
        while q0 < nq {
            let q1 = (q0 + block_rows_write).min(nq);

            let lines: Vec<String> = (q0..q1)
                .into_par_iter()
                .map(|qi| {
                    let q_path = &query_genomes[qi];
                    let query_basename = Path::new(q_path)
                        .file_name()
                        .and_then(|os_str| os_str.to_str())
                        .unwrap_or(q_path);

                    let mut line = String::with_capacity(nr * 16 + q_path.len() + 1);
                    let row_base = qi * nr;
                    let mut fmt = ryu::Buffer::new();

                    line.push_str(q_path);

                    for (ri, r_path) in reference_genomes.iter().enumerate() {
                        let reference_basename = Path::new(r_path)
                            .file_name()
                            .and_then(|os_str| os_str.to_str())
                            .unwrap_or(r_path);

                        let mut distance = compute_distance_from_hamming(
                            hamming_rect[row_base + ri] as f64,
                            kmer_size,
                        );

                        if query_basename == reference_basename {
                            distance = 0.0;
                        }

                        line.push('\t');
                        line.push_str(fmt.format_finite(distance));
                    }

                    line.push('\n');
                    line
                })
                .collect();

            for line in &lines {
                output_writer
                    .write_all(line.as_bytes())
                    .expect("Error writing matrix row block");
            }

            q0 = q1;
        }
    } else {
        writeln!(output_writer, "Query\tReference\tDistance").expect("Error writing header");

        let mut q0 = 0usize;
        while q0 < nq {
            let q1 = (q0 + block_rows_write).min(nq);

            let lines: Vec<String> = (q0..q1)
                .into_par_iter()
                .map(|qi| {
                    let q_path = &query_genomes[qi];
                    let query_basename = Path::new(q_path)
                        .file_name()
                        .and_then(|os_str| os_str.to_str())
                        .unwrap_or(q_path);

                    let mut line_block = String::with_capacity(nr * 64);
                    let row_base = qi * nr;
                    let mut fmt = ryu::Buffer::new();

                    for (ri, r_path) in reference_genomes.iter().enumerate() {
                        let reference_basename = Path::new(r_path)
                            .file_name()
                            .and_then(|os_str| os_str.to_str())
                            .unwrap_or(r_path);

                        let mut distance = compute_distance_from_hamming(
                            hamming_rect[row_base + ri] as f64,
                            kmer_size,
                        );

                        if query_basename == reference_basename {
                            distance = 0.0;
                        }

                        line_block.push_str(q_path);
                        line_block.push('\t');
                        line_block.push_str(r_path);
                        line_block.push('\t');
                        line_block.push_str(fmt.format_finite(distance));
                        line_block.push('\n');
                    }

                    line_block
                })
                .collect();

            for line in &lines {
                output_writer
                    .write_all(line.as_bytes())
                    .expect("Error writing output block");
            }

            q0 = q1;
        }
    }

    output_writer.flush().expect("Error flushing output");

    log::info!(
        "GPU distance: transform + zstd write stage finished in {:.3}s",
        write_t0.elapsed().as_secs_f64()
    );
    log::info!(
        "GPU distance: total compute + zstd write completed in {:.3}s",
        total_t0.elapsed().as_secs_f64()
    );
}

/// Runs the pipeline for a given Kmer type and densification type (`dens`).
///
/// - `dens = 0` => `OptDensHashSketch`
/// - `dens = 1` => `RevOptDensHashSketch`
///
/// We instantiate the internal minhash with `S = f32`,
/// but the returned signature type is `u64` (per your kmerutils implementation).
#[cfg(not(feature = "cuda"))]
fn sketching_kmerType<Kmer, F>(
    query_genomes: &[String],
    reference_genomes: &[String],
    sketch_args: &SeqSketcherParams,
    kmer_hash_fn: F,
    dens: usize,
    output: Option<String>,
    matrix_output: bool,
    kmer_size: usize,
) where
    Kmer: CompressedKmerT + KmerBuilder<Kmer> + Send + Sync,
    <Kmer as CompressedKmerT>::Val: num::PrimInt + Send + Sync + Debug,
    KmerGenerator<Kmer>: KmerGenerationPattern<Kmer>,
    F: Fn(&Kmer) -> <Kmer as CompressedKmerT>::Val + Send + Sync + Copy,
{
    let total_t0 = Instant::now();
    log::info!(
        "pipeline(CPU): starting with {} query genomes and {} reference genomes",
        query_genomes.len(),
        reference_genomes.len()
    );

    match dens {
        0 => {
            let sketcher = OptDensHashSketch::<Kmer, f32>::new(sketch_args);

            let tq = Instant::now();
            println!("Sketching query genomes with OptDens...");
            let query_sketches = sketch_files(query_genomes, &sketcher, kmer_hash_fn);
            log::info!("pipeline(CPU): query sketching finished in {:.3}s", tq.elapsed().as_secs_f64());

            let tr = Instant::now();
            println!("Sketching reference genomes with OptDens...");
            let reference_sketches = sketch_files(reference_genomes, &sketcher, kmer_hash_fn);
            log::info!("pipeline(CPU): reference sketching finished in {:.3}s", tr.elapsed().as_secs_f64());

            let td = Instant::now();
            println!("Performing pairwise comparisons...");
            write_results(
                output,
                matrix_output,
                query_genomes,
                reference_genomes,
                &query_sketches,
                &reference_sketches,
                kmer_size,
            );
            log::info!("pipeline(CPU): distance stage finished in {:.3}s", td.elapsed().as_secs_f64());
        }
        1 => {
            let sketcher = RevOptDensHashSketch::<Kmer, f32>::new(sketch_args);

            let tq = Instant::now();
            println!("Sketching query genomes with RevOptDens...");
            let query_sketches = sketch_files(query_genomes, &sketcher, kmer_hash_fn);
            log::info!("pipeline(CPU): query sketching finished in {:.3}s", tq.elapsed().as_secs_f64());

            let tr = Instant::now();
            println!("Sketching reference genomes with RevOptDens...");
            let reference_sketches = sketch_files(reference_genomes, &sketcher, kmer_hash_fn);
            log::info!("pipeline(CPU): reference sketching finished in {:.3}s", tr.elapsed().as_secs_f64());

            let td = Instant::now();
            println!("Performing pairwise comparisons...");
            write_results(
                output,
                matrix_output,
                query_genomes,
                reference_genomes,
                &query_sketches,
                &reference_sketches,
                kmer_size,
            );
            log::info!("pipeline(CPU): distance stage finished in {:.3}s", td.elapsed().as_secs_f64());
        }
        _ => panic!("Only densification = 0 or 1 are supported!"),
    }

    log::info!(
        "pipeline(CPU): total runtime {:.3}s",
        total_t0.elapsed().as_secs_f64()
    );
}

#[cfg(feature = "cuda")]
fn sketching_kmerType<Kmer, F>(
    query_genomes: &[String],
    reference_genomes: &[String],
    sketch_args: &SeqSketcherParams,
    kmer_hash_fn: F,
    dens: usize,
    output: Option<String>,
    matrix_output: bool,
    kmer_size: usize,
) where
    Kmer: CompressedKmerT + KmerBuilder<Kmer> + Send + Sync,
    <Kmer as CompressedKmerT>::Val: num::PrimInt + Send + Sync + Debug,
    KmerGenerator<Kmer>: KmerGenerationPattern<Kmer>,
    F: Fn(&Kmer) -> <Kmer as CompressedKmerT>::Val + Send + Sync + Copy,
    OptDensHashSketch<Kmer, f32>: SeqSketcherT<Kmer, Sig = u16>,
    RevOptDensHashSketch<Kmer, f32>: SeqSketcherT<Kmer, Sig = u16>,
{
    let total_t0 = Instant::now();
    log::info!(
        "pipeline(GPU): starting with {} query genomes and {} reference genomes",
        query_genomes.len(),
        reference_genomes.len()
    );

    match dens {
        0 => {
            let sketcher = OptDensHashSketch::<Kmer, f32>::new(sketch_args);

            let tq = Instant::now();
            println!("Sketching query genomes with OptDens...");
            let query_sketches = sketch_files(query_genomes, &sketcher, kmer_hash_fn);
            log::info!("pipeline(GPU): query sketching finished in {:.3}s", tq.elapsed().as_secs_f64());

            let tr = Instant::now();
            println!("Sketching reference genomes with OptDens...");
            let reference_sketches = sketch_files(reference_genomes, &sketcher, kmer_hash_fn);
            log::info!("pipeline(GPU): reference sketching finished in {:.3}s", tr.elapsed().as_secs_f64());

            let td = Instant::now();
            println!("Performing GPU pairwise comparisons...");
            write_results_gpu_u16(
                output,
                matrix_output,
                query_genomes,
                reference_genomes,
                &query_sketches,
                &reference_sketches,
                kmer_size,
            );
            log::info!("pipeline(GPU): distance stage finished in {:.3}s", td.elapsed().as_secs_f64());
        }
        1 => {
            let sketcher = RevOptDensHashSketch::<Kmer, f32>::new(sketch_args);

            let tq = Instant::now();
            println!("Sketching query genomes with RevOptDens...");
            let query_sketches = sketch_files(query_genomes, &sketcher, kmer_hash_fn);
            log::info!("pipeline(GPU): query sketching finished in {:.3}s", tq.elapsed().as_secs_f64());

            let tr = Instant::now();
            println!("Sketching reference genomes with RevOptDens...");
            let reference_sketches = sketch_files(reference_genomes, &sketcher, kmer_hash_fn);
            log::info!("pipeline(GPU): reference sketching finished in {:.3}s", tr.elapsed().as_secs_f64());

            let td = Instant::now();
            println!("Performing GPU pairwise comparisons...");
            write_results_gpu_u16(
                output,
                matrix_output,
                query_genomes,
                reference_genomes,
                &query_sketches,
                &reference_sketches,
                kmer_size,
            );
            log::info!("pipeline(GPU): distance stage finished in {:.3}s", td.elapsed().as_secs_f64());
        }
        _ => panic!("Only densification = 0 or 1 are supported!"),
    }

    log::info!(
        "pipeline(GPU): total runtime {:.3}s",
        total_t0.elapsed().as_secs_f64()
    );
}

fn main() {
    println!("\n ************** initializing logger *****************\n");
    let _ = env_logger::Builder::from_default_env().init();

    let matches = Command::new("BinDash")
        .version("0.3.3")
        .about("Binwise Densified MinHash for Genome/Metagenome/Pangenome Comparisons")
        .arg(
            Arg::new("query_list")
                .short('q')
                .long("query_list")
                .value_name("QUERY_LIST_FILE")
                .help("Query genome list file (one FASTA/FNA file path per line, .gz supported)")
                .required(true)
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("reference_list")
                .short('r')
                .long("reference_list")
                .value_name("REFERENCE_LIST_FILE")
                .help("Reference genome list file (one FASTA/FNA file path per line, .gz supported)")
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
                .help("MinHash sketch size")
                .default_value("2048")
                .value_parser(clap::value_parser!(usize))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("dens_opt")
                .short('d')
                .long("densification")
                .value_name("DENS_OPT")
                .help("Densification strategy, 0 = optimal densification, 1 = reverse optimal/faster densification")
                .default_value("0")
                .value_parser(clap::value_parser!(usize))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("threads")
                .short('t')
                .long("threads")
                .value_name("THREADS")
                .help("Number of threads to use in parallel")
                .default_value("1")
                .value_parser(clap::value_parser!(usize))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("matrix")
                .long("matrix")
                .help("Write dense rectangular matrix output")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .value_name("OUTPUT_FILE")
                .help("Output file (zstd-compressed by default)")
                .required(false)
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
    let dens = *matches.get_one::<usize>("dens_opt").unwrap();
    let threads = *matches.get_one::<usize>("threads").unwrap();
    let matrix_output = matches.get_flag("matrix");
    let output = matches.get_one::<String>("output").cloned();

    log::info!(
        "main: query_list={} reference_list={} kmer_size={} sketch_size={} dens={} threads={} output={:?} matrix_output={}",
        query_list,
        reference_list,
        kmer_size,
        sketch_size,
        dens,
        threads,
        output,
        matrix_output
    );

    ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();

    let query_genomes = read_genome_list(&query_list);
    let reference_genomes = read_genome_list(&reference_list);
    log::info!(
        "main: loaded {} query genomes and {} reference genomes",
        query_genomes.len(),
        reference_genomes.len()
    );

    let sketch_args = SeqSketcherParams::new(
        kmer_size,
        sketch_size,
        SketchAlgo::OPTDENS, // actual algo chosen by sketcher type
        DataType::DNA,
    );

    if kmer_size <= 14 {
        let nb_alphabet_bits = 2;
        let kmer_hash_fn_32bit = move |kmer: &Kmer32bit| -> <Kmer32bit as CompressedKmerT>::Val {
            let mask: <Kmer32bit as CompressedKmerT>::Val =
                num::NumCast::from::<u64>((1u64 << (nb_alphabet_bits * kmer.get_nb_base())) - 1)
                    .unwrap();
            kmer.get_compressed_value() & mask
        };

        sketching_kmerType::<Kmer32bit, _>(
            &query_genomes,
            &reference_genomes,
            &sketch_args,
            kmer_hash_fn_32bit,
            dens,
            output,
            matrix_output,
            kmer_size,
        );
    } else if kmer_size == 16 {
        let nb_alphabet_bits = 2;
        let kmer_hash_fn_16b32bit =
            move |kmer: &Kmer16b32bit| -> <Kmer16b32bit as CompressedKmerT>::Val {
                let canonical = kmer.reverse_complement().min(*kmer);
                let mask: <Kmer16b32bit as CompressedKmerT>::Val =
                    num::NumCast::from::<u64>(
                        (1u64 << (nb_alphabet_bits * kmer.get_nb_base())) - 1,
                    )
                    .unwrap();
                canonical.get_compressed_value() & mask
            };

        sketching_kmerType::<Kmer16b32bit, _>(
            &query_genomes,
            &reference_genomes,
            &sketch_args,
            kmer_hash_fn_16b32bit,
            dens,
            output,
            matrix_output,
            kmer_size,
        );
    } else if kmer_size <= 32 {
        let nb_alphabet_bits = 2;
        let kmer_hash_fn_64bit = move |kmer: &Kmer64bit| -> <Kmer64bit as CompressedKmerT>::Val {
            let canonical = kmer.reverse_complement().min(*kmer);
            let mask: <Kmer64bit as CompressedKmerT>::Val =
                num::NumCast::from::<u64>((1u64 << (nb_alphabet_bits * kmer.get_nb_base())) - 1)
                    .unwrap();
            canonical.get_compressed_value() & mask
        };

        sketching_kmerType::<Kmer64bit, _>(
            &query_genomes,
            &reference_genomes,
            &sketch_args,
            kmer_hash_fn_64bit,
            dens,
            output,
            matrix_output,
            kmer_size,
        );
    } else {
        panic!("kmer_size must not be 15 and cannot exceed 32!");
    }
}