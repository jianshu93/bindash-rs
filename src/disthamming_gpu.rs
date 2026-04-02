//! CUDA rectangular in-memory Hamming distance for u16 sketches.
//!
//! Computes a full query x reference distance matrix on one or more GPUs.
//! Input sketch matrices are row-major:
//!   query_sketches: [nq * k]
//!   reference_sketches: [nr * k]
//! Output distance matrix is row-major:
//!   out: [nq * nr]
//! where out[i * nr + j] = normalized_hamming(query_i, reference_j)
//
// Distances are stored as f32.

use anyhow::{bail, Context, Result};
use cudarc::driver::{CudaContext, CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use log::{debug, info, warn};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

const KERNEL_SRC: &str = r#"
#ifndef BK16
#define BK16 64
#endif

#ifndef STRIDE16
#define STRIDE16 (BK16 + 1)
#endif

__device__ __forceinline__ unsigned long long pack_u16x4(const unsigned short* p) {
    return  (unsigned long long)p[0]
         | ((unsigned long long)p[1] << 16)
         | ((unsigned long long)p[2] << 32)
         | ((unsigned long long)p[3] << 48);
}

__device__ __forceinline__ unsigned mismatch_u16x4_from_xor(unsigned long long x) {
    unsigned long long m = (x - 0x0101010101010101ULL) & ~x & 0x8080808080808080ULL;
    unsigned long long w = (m & (m >> 8)) & 0x0080008000800080ULL;
    unsigned zeros = __popcll(w);
    return 4u - zeros;
}

extern "C" __global__
void hamming_rect_u16_packed(
    const unsigned short* __restrict__ query_sketches,   // [nq * k]
    const unsigned short* __restrict__ ref_sketches,     // [nr * k]
    int nq,
    int nr,
    int k,
    int i0,
    int j0,
    int bw,
    int bh,
    float* __restrict__ out // [bw * bh], row-major
){
    const int jj = blockIdx.x * blockDim.x + threadIdx.x; // local ref col
    const int ii = blockIdx.y * blockDim.y + threadIdx.y; // local query row

    const int qi = i0 + ii;
    const int rj = j0 + jj;

    const int k4   = (k >> 2);
    const int krem = (k & 3);

    extern __shared__ __align__(16) unsigned char smem_raw[];
    unsigned long long* As = reinterpret_cast<unsigned long long*>(smem_raw);
    unsigned long long* Bs = As + (size_t)blockDim.y * (size_t)STRIDE16;

    unsigned int diff = 0u;

    const int tpb = blockDim.x * blockDim.y;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    for (int t0 = 0; t0 < k4; t0 += BK16) {
        const int bk = min(BK16, k4 - t0);

        // Load query slab: blockDim.y rows x bk packed u64 words
        const int totalA = blockDim.y * bk;
        for (int idx = tid; idx < totalA; idx += tpb) {
            const int r = idx / bk;
            const int t = idx - r * bk;

            const int tile_row_base = i0 + (int)blockIdx.y * (int)blockDim.y;
            const int gq = tile_row_base + r;

            unsigned long long v = 0ULL;
            if (gq < nq) {
                const int off = ((t0 + t) << 2);
                const unsigned short* base =
                    query_sketches + (size_t)gq * (size_t)k + (size_t)off;
                v = pack_u16x4(base);
            }
            As[(size_t)r * (size_t)STRIDE16 + (size_t)t] = v;
        }

        // Load ref slab: blockDim.x cols x bk packed u64 words
        const int totalB = blockDim.x * bk;
        for (int idx = tid; idx < totalB; idx += tpb) {
            const int c = idx / bk;
            const int t = idx - c * bk;

            const int tile_col_base = j0 + (int)blockIdx.x * (int)blockDim.x;
            const int gr = tile_col_base + c;

            unsigned long long v = 0ULL;
            if (gr < nr) {
                const int off = ((t0 + t) << 2);
                const unsigned short* base =
                    ref_sketches + (size_t)gr * (size_t)k + (size_t)off;
                v = pack_u16x4(base);
            }
            Bs[(size_t)c * (size_t)STRIDE16 + (size_t)t] = v;
        }

        __syncthreads();

        if (ii < bw && jj < bh && qi < nq && rj < nr) {
            const size_t arow = (size_t)threadIdx.y * (size_t)STRIDE16;
            const size_t brow = (size_t)threadIdx.x * (size_t)STRIDE16;

            #pragma unroll
            for (int t = 0; t < bk; ++t) {
                unsigned long long x = As[arow + (size_t)t] ^ Bs[brow + (size_t)t];
                diff += mismatch_u16x4_from_xor(x);
            }
        }

        __syncthreads();
    }

    if (krem && ii < bw && jj < bh && qi < nq && rj < nr) {
        const int base = (k4 << 2);
        const unsigned short* a = query_sketches + (size_t)qi * (size_t)k + (size_t)base;
        const unsigned short* b = ref_sketches   + (size_t)rj * (size_t)k + (size_t)base;
        #pragma unroll
        for (int t = 0; t < 3; ++t) {
            if (t < krem) diff += (a[t] != b[t]);
        }
    }

    if (ii < bw && jj < bh && qi < nq && rj < nr) {
        out[(size_t)ii * (size_t)bh + (size_t)jj] = ((float)diff / (float)k);
    }
}
"#;

pub fn device_count() -> Result<usize> {
    Ok(CudaContext::device_count()? as usize)
}

#[inline]
fn mib(x: usize) -> f64 {
    (x as f64) / (1024.0 * 1024.0)
}

#[inline]
fn gib(x: usize) -> f64 {
    (x as f64) / (1024.0 * 1024.0 * 1024.0)
}

#[inline]
fn shared_mem_bytes_u16(blk_x: usize, blk_y: usize) -> u32 {
    let stride_words = 64usize + 1; // STRIDE16
    ((stride_words * (blk_y + blk_x)) * 8usize) as u32
}

fn pairwise_hamming_rect_single_gpu_u16(
    query_sketches: &[u16],
    nq: usize,
    ref_sketches: &[u16],
    nr: usize,
    k: usize,
    out: &mut [f32],
    block_rows: usize,
    block_cols: usize,
    gpu_id: usize,
) -> Result<()> {
    if query_sketches.len() != nq * k {
        bail!(
            "query_sketches length mismatch: got {}, expected {}",
            query_sketches.len(),
            nq * k
        );
    }
    if ref_sketches.len() != nr * k {
        bail!(
            "ref_sketches length mismatch: got {}, expected {}",
            ref_sketches.len(),
            nr * k
        );
    }
    if out.len() != nq * nr {
        bail!(
            "out length mismatch: got {}, expected {}",
            out.len(),
            nq * nr
        );
    }
    if k == 0 {
        bail!("k must be > 0");
    }

    let mut block_rows = block_rows.max(1).min(nq.max(1));
    let mut block_cols = block_cols.max(1).min(nr.max(1));

    let cap = 4096usize;
    if block_rows > cap {
        warn!("capping block_rows from {} to {}", block_rows, cap);
        block_rows = cap;
    }
    if block_cols > cap {
        warn!("capping block_cols from {} to {}", block_cols, cap);
        block_cols = cap;
    }

    info!(
        "single-GPU rect: nq={} nr={} k={} block_rows={} block_cols={} query={:.2} GiB ref={:.2} GiB out={:.2} GiB",
        nq,
        nr,
        k,
        block_rows,
        block_cols,
        gib(query_sketches.len() * std::mem::size_of::<u16>()),
        gib(ref_sketches.len() * std::mem::size_of::<u16>()),
        gib(out.len() * std::mem::size_of::<f32>())
    );

    let ctx = CudaContext::new(gpu_id)?;
    let stream = ctx.default_stream();

    let ptx = compile_ptx(KERNEL_SRC)?;
    let module = ctx.load_module(ptx)?;
    let func = module
        .load_function("hamming_rect_u16_packed")
        .context("load function hamming_rect_u16_packed")?;

    let d_query: CudaSlice<u16> = stream.clone_htod(query_sketches)?;
    let d_ref: CudaSlice<u16> = stream.clone_htod(ref_sketches)?;

    info!(
        "single-GPU rect: uploaded query {:.2} MiB, ref {:.2} MiB",
        mib(query_sketches.len() * std::mem::size_of::<u16>()),
        mib(ref_sketches.len() * std::mem::size_of::<u16>())
    );

    let max_bw = block_rows.min(nq.max(1));
    let max_bh = block_cols.min(nr.max(1));
    let scratch_elems = max_bw * max_bh;

    let mut d_tile: CudaSlice<f32> = stream
        .alloc_zeros(scratch_elems)
        .with_context(|| format!("alloc d_tile: {:.2} MiB", mib(scratch_elems * 4)))?;
    let mut h_tile = vec![0.0f32; scratch_elems];

    let nq_i32 = nq as i32;
    let nr_i32 = nr as i32;
    let k_i32 = k as i32;

    let nbq = nq.div_ceil(block_rows);
    let nbr = nr.div_ceil(block_cols);

    for bi in 0..nbq {
        let i0 = bi * block_rows;
        let bw = (nq - i0).min(block_rows);

        for bj in 0..nbr {
            let j0 = bj * block_cols;
            let bh = (nr - j0).min(block_cols);

            let blk_x = 64usize;
            let blk_y = 8usize;
            let smem_bytes = shared_mem_bytes_u16(blk_x, blk_y);

            let cfg = LaunchConfig {
                grid_dim: (
                    bh.div_ceil(blk_x) as u32,
                    bw.div_ceil(blk_y) as u32,
                    1,
                ),
                block_dim: (blk_x as u32, blk_y as u32, 1),
                shared_mem_bytes: smem_bytes,
            };

            let i0_i32 = i0 as i32;
            let j0_i32 = j0 as i32;
            let bw_i32 = bw as i32;
            let bh_i32 = bh as i32;

            debug!(
                "single-GPU rect: tile bi={} bj={} i0={} j0={} bw={} bh={} smem={}",
                bi, bj, i0, j0, bw, bh, smem_bytes
            );

            let mut launch = stream.launch_builder(&func);
            launch.arg(&d_query);
            launch.arg(&d_ref);
            launch.arg(&nq_i32);
            launch.arg(&nr_i32);
            launch.arg(&k_i32);
            launch.arg(&i0_i32);
            launch.arg(&j0_i32);
            launch.arg(&bw_i32);
            launch.arg(&bh_i32);
            launch.arg(&mut d_tile);

            unsafe { launch.launch(cfg) }?;
            stream.memcpy_dtoh(&d_tile, &mut h_tile)?;

            let base_ptr = out.as_mut_ptr();
            unsafe {
                for ii in 0..bw {
                    let qi = i0 + ii;
                    for jj in 0..bh {
                        let rj = j0 + jj;
                        let d = h_tile[ii * bh + jj];
                        *base_ptr.add(qi * nr + rj) = d;
                    }
                }
            }
        }
    }

    Ok(())
}

fn pairwise_hamming_rect_multi_gpu_u16_impl(
    query_sketches: &[u16],
    nq: usize,
    ref_sketches: &[u16],
    nr: usize,
    k: usize,
    out: &mut [f32],
    block_rows: usize,
    block_cols: usize,
) -> Result<()> {
    if query_sketches.len() != nq * k {
        bail!(
            "query_sketches length mismatch: got {}, expected {}",
            query_sketches.len(),
            nq * k
        );
    }
    if ref_sketches.len() != nr * k {
        bail!(
            "ref_sketches length mismatch: got {}, expected {}",
            ref_sketches.len(),
            nr * k
        );
    }
    if out.len() != nq * nr {
        bail!(
            "out length mismatch: got {}, expected {}",
            out.len(),
            nq * nr
        );
    }
    if k == 0 {
        bail!("k must be > 0");
    }

    let ng = device_count()?;
    if ng == 0 {
        bail!("No CUDA devices available");
    }
    if ng == 1 {
        return pairwise_hamming_rect_single_gpu_u16(
            query_sketches,
            nq,
            ref_sketches,
            nr,
            k,
            out,
            block_rows,
            block_cols,
            0,
        );
    }

    let mut block_rows = block_rows.max(1).min(nq.max(1));
    let mut block_cols = block_cols.max(1).min(nr.max(1));

    let cap = 4096usize;
    if block_rows > cap {
        warn!("multi-GPU: capping block_rows from {} to {}", block_rows, cap);
        block_rows = cap;
    }
    if block_cols > cap {
        warn!("multi-GPU: capping block_cols from {} to {}", block_cols, cap);
        block_cols = cap;
    }

    let nbq = nq.div_ceil(block_rows);
    let nbr = nr.div_ceil(block_cols);

    let mut tiles = Vec::<(usize, usize)>::with_capacity(nbq * nbr);
    for bi in 0..nbq {
        for bj in 0..nbr {
            tiles.push((bi, bj));
        }
    }

    info!(
        "multi-GPU rect: ng={} nq={} nr={} k={} block_rows={} block_cols={} tiles={}",
        ng, nq, nr, k, block_rows, block_cols, tiles.len()
    );

    let ptx = Arc::new(compile_ptx(KERNEL_SRC)?);
    let tiles = Arc::new(tiles);
    let next = Arc::new(AtomicUsize::new(0));

    let q_arc: Arc<Vec<u16>> = Arc::new(query_sketches.to_vec());
    let r_arc: Arc<Vec<u16>> = Arc::new(ref_sketches.to_vec());

    let out_addr = out.as_mut_ptr() as usize;

    std::thread::scope(|scope| {
        for dev_id in 0..ng {
            let ptx = Arc::clone(&ptx);
            let tiles = Arc::clone(&tiles);
            let next = Arc::clone(&next);
            let q_arc = Arc::clone(&q_arc);
            let r_arc = Arc::clone(&r_arc);

            scope.spawn(move || {
                let inner = || -> Result<()> {
                    let ctx = CudaContext::new(dev_id)?;
                    let stream = ctx.default_stream();

                    let module = ctx.load_module((*ptx).clone())?;
                    let func = module
                        .load_function("hamming_rect_u16_packed")
                        .context("load function hamming_rect_u16_packed")?;

                    let d_query: CudaSlice<u16> = stream.clone_htod(&q_arc[..])?;
                    let d_ref: CudaSlice<u16> = stream.clone_htod(&r_arc[..])?;

                    let max_bw = block_rows.min(nq.max(1));
                    let max_bh = block_cols.min(nr.max(1));

                    let mut d_tile: CudaSlice<f32> = stream.alloc_zeros(max_bw * max_bh)?;
                    let mut h_tile = vec![0.0f32; max_bw * max_bh];

                    let nq_i32 = nq as i32;
                    let nr_i32 = nr as i32;
                    let k_i32 = k as i32;

                    loop {
                        let tix = next.fetch_add(1, Ordering::Relaxed);
                        if tix >= tiles.len() {
                            break;
                        }

                        let (bi, bj) = tiles[tix];

                        let i0 = bi * block_rows;
                        let j0 = bj * block_cols;
                        let bw = (nq - i0).min(block_rows);
                        let bh = (nr - j0).min(block_cols);

                        let blk_x = 64usize;
                        let blk_y = 8usize;
                        let smem_bytes = shared_mem_bytes_u16(blk_x, blk_y);

                        let cfg = LaunchConfig {
                            grid_dim: (
                                bh.div_ceil(blk_x) as u32,
                                bw.div_ceil(blk_y) as u32,
                                1,
                            ),
                            block_dim: (blk_x as u32, blk_y as u32, 1),
                            shared_mem_bytes: smem_bytes,
                        };

                        let i0_i32 = i0 as i32;
                        let j0_i32 = j0 as i32;
                        let bw_i32 = bw as i32;
                        let bh_i32 = bh as i32;

                        let mut launch = stream.launch_builder(&func);
                        launch.arg(&d_query);
                        launch.arg(&d_ref);
                        launch.arg(&nq_i32);
                        launch.arg(&nr_i32);
                        launch.arg(&k_i32);
                        launch.arg(&i0_i32);
                        launch.arg(&j0_i32);
                        launch.arg(&bw_i32);
                        launch.arg(&bh_i32);
                        launch.arg(&mut d_tile);

                        unsafe { launch.launch(cfg) }?;
                        stream.memcpy_dtoh(&d_tile, &mut h_tile)?;

                        let base_ptr = out_addr as *mut f32;
                        unsafe {
                            for ii in 0..bw {
                                let qi = i0 + ii;
                                for jj in 0..bh {
                                    let rj = j0 + jj;
                                    let d = h_tile[ii * bh + jj];
                                    *base_ptr.add(qi * nr + rj) = d;
                                }
                            }
                        }
                    }

                    Ok(())
                };

                if let Err(e) = inner() {
                    panic!("GPU worker {} failed: {e:?}", dev_id);
                }
            });
        }
    });

    Ok(())
}

pub fn pairwise_hamming_rect_multi_gpu_u16(
    query_sketches: &[u16],
    nq: usize,
    ref_sketches: &[u16],
    nr: usize,
    k: usize,
    out: &mut [f32],
    block_rows: usize,
    block_cols: usize,
) -> Result<()> {
    pairwise_hamming_rect_multi_gpu_u16_impl(
        query_sketches,
        nq,
        ref_sketches,
        nr,
        k,
        out,
        block_rows,
        block_cols,
    )
}