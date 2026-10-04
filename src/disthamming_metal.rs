//! Minimal in-memory Metal backend for rectangular u16 Hamming distance.

use anyhow::{anyhow, bail, Context, Result};
use log::info;
use metal::objc::rc::autoreleasepool;
use metal::{
    CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize, NSUInteger,
};
use std::ffi::c_void;
use std::time::Instant;

const SLOT_BYTES: usize = std::mem::size_of::<u64>();

const SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Params {
    int nq;
    int nr;
    int k;
    int bk;
    int stride;
};

inline ushort4 load_slot(device const ushort *p) {
    return ushort4(p[0], p[1], p[2], p[3]);
}

inline uint lane_diff(ushort4 a, ushort4 b) {
    ushort4 ne = select(ushort4(0), ushort4(1), a != b);
    return uint(ne.x) + uint(ne.y) + uint(ne.z) + uint(ne.w);
}

kernel void hamming_rect_u16(
    device const ushort *query_sketches [[buffer(0)]],
    device const ushort *ref_sketches   [[buffer(1)]],
    constant Params &p                  [[buffer(2)]],
    device float *out                   [[buffer(3)]],
    threadgroup ushort4 *smem           [[threadgroup(0)]],
    uint2 tgpos [[threadgroup_position_in_grid]],
    uint2 tpt   [[thread_position_in_threadgroup]],
    uint2 tgdim [[threads_per_threadgroup]])
{
    const int rj = int(tgpos.x * tgdim.x + tpt.x);
    const int qi = int(tgpos.y * tgdim.y + tpt.y);
    const int nslot = p.k >> 2;
    const int krem = p.k & 3;

    threadgroup ushort4 *query_slab = smem;
    threadgroup ushort4 *ref_slab = query_slab + int(tgdim.y) * p.stride;

    const int threads_per_group = int(tgdim.x * tgdim.y);
    const int tid = int(tpt.y * tgdim.x + tpt.x);
    const bool active = qi < p.nq && rj < p.nr;
    uint diff = 0u;

    for (int t0 = 0; t0 < nslot; t0 += p.bk) {
        const int slab = min(p.bk, nslot - t0);
        const int query_base = int(tgpos.y * tgdim.y);
        const int ref_base = int(tgpos.x * tgdim.x);

        for (int idx = tid; idx < int(tgdim.y) * slab; idx += threads_per_group) {
            const int row = idx / slab;
            const int t = idx - row * slab;
            const int global_row = query_base + row;
            ushort4 value = ushort4(0);
            if (global_row < p.nq) {
                device const ushort *src = query_sketches
                    + ulong(global_row) * ulong(p.k)
                    + ulong((t0 + t) << 2);
                value = load_slot(src);
            }
            query_slab[row * p.stride + t] = value;
        }

        for (int idx = tid; idx < int(tgdim.x) * slab; idx += threads_per_group) {
            const int col = idx / slab;
            const int t = idx - col * slab;
            const int global_col = ref_base + col;
            ushort4 value = ushort4(0);
            if (global_col < p.nr) {
                device const ushort *src = ref_sketches
                    + ulong(global_col) * ulong(p.k)
                    + ulong((t0 + t) << 2);
                value = load_slot(src);
            }
            ref_slab[col * p.stride + t] = value;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active) {
            const int query_offset = int(tpt.y) * p.stride;
            const int ref_offset = int(tpt.x) * p.stride;
            for (int t = 0; t < slab; ++t) {
                diff += lane_diff(query_slab[query_offset + t], ref_slab[ref_offset + t]);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active && krem != 0) {
        const int tail = nslot << 2;
        device const ushort *a = query_sketches + ulong(qi) * ulong(p.k) + ulong(tail);
        device const ushort *b = ref_sketches + ulong(rj) * ulong(p.k) + ulong(tail);
        for (int t = 0; t < krem; ++t) {
            diff += uint(a[t] != b[t]);
        }
    }

    if (active) {
        out[ulong(qi) * ulong(p.nr) + ulong(rj)] = float(diff) / float(p.k);
    }
}
"#;

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    nq: i32,
    nr: i32,
    k: i32,
    bk: i32,
    stride: i32,
}

#[derive(Clone, Copy, Debug)]
struct Tile {
    x: usize,
    y: usize,
    bk: usize,
    stride: usize,
    smem: usize,
}

fn choose_tile(pipeline: &ComputePipelineState, max_smem: usize) -> Result<Tile> {
    let width = pipeline.thread_execution_width() as usize;
    let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
    if width == 0 || max_threads == 0 {
        bail!("Metal reported invalid threadgroup limits");
    }

    let threads = (max_threads / 2).max(width);
    let x = width.min(threads);
    let y = (threads / x).max(1);
    let budget = max_smem / 2;

    let mut bk = 1usize;
    while bk < 256 && (2 * bk + 1) * (x + y) * SLOT_BYTES <= budget {
        bk *= 2;
    }

    let stride = bk + 1;
    let smem = stride * (x + y) * SLOT_BYTES;
    if smem > max_smem {
        bail!("Metal tile needs {smem} bytes of threadgroup memory, device allows {max_smem}");
    }

    Ok(Tile {
        x,
        y,
        bk,
        stride,
        smem,
    })
}

pub fn pairwise_hamming_rect_metal_u16(
    query_sketches: &[u16],
    nq: usize,
    ref_sketches: &[u16],
    nr: usize,
    k: usize,
    out: &mut [f32],
) -> Result<()> {
    if query_sketches.len() != nq * k {
        bail!(
            "query sketch length mismatch: got {}, expected {}",
            query_sketches.len(),
            nq * k
        );
    }
    if ref_sketches.len() != nr * k {
        bail!(
            "reference sketch length mismatch: got {}, expected {}",
            ref_sketches.len(),
            nr * k
        );
    }
    if out.len() != nq * nr {
        bail!(
            "output length mismatch: got {}, expected {}",
            out.len(),
            nq * nr
        );
    }
    if k == 0 {
        bail!("sketch size must be greater than zero");
    }
    if nq == 0 || nr == 0 {
        return Ok(());
    }

    let device = Device::system_default().context("no Metal device found")?;
    let queue = device.new_command_queue();

    let compile_t0 = Instant::now();
    let options = CompileOptions::new();
    options.set_fast_math_enabled(false);
    let library = device
        .new_library_with_source(SHADER, &options)
        .map_err(|e| anyhow!("Metal shader compilation failed: {e}"))?;
    let function = library
        .get_function("hamming_rect_u16", None)
        .map_err(|e| anyhow!("Metal function hamming_rect_u16 not found: {e}"))?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow!("Metal compute pipeline creation failed: {e}"))?;
    let tile = choose_tile(&pipeline, device.max_threadgroup_memory_length() as usize)?;

    info!(
        "Metal device {}: tile {}x{}, slab {}, threadgroup memory {} bytes, pipeline ready in {:.3}s",
        device.name(),
        tile.x,
        tile.y,
        tile.bk,
        tile.smem,
        compile_t0.elapsed().as_secs_f64()
    );

    let query_buffer = device.new_buffer_with_data(
        query_sketches.as_ptr() as *const c_void,
        std::mem::size_of_val(query_sketches) as NSUInteger,
        MTLResourceOptions::StorageModeShared,
    );
    let ref_buffer = device.new_buffer_with_data(
        ref_sketches.as_ptr() as *const c_void,
        std::mem::size_of_val(ref_sketches) as NSUInteger,
        MTLResourceOptions::StorageModeShared,
    );
    let out_buffer = device.new_buffer(
        std::mem::size_of_val(out) as NSUInteger,
        MTLResourceOptions::StorageModeShared,
    );
    let params = Params {
        nq: nq as i32,
        nr: nr as i32,
        k: k as i32,
        bk: tile.bk as i32,
        stride: tile.stride as i32,
    };

    let kernel_t0 = Instant::now();
    autoreleasepool(|| {
        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&query_buffer), 0);
        encoder.set_buffer(1, Some(&ref_buffer), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<Params>() as NSUInteger,
            &params as *const Params as *const c_void,
        );
        encoder.set_buffer(3, Some(&out_buffer), 0);
        encoder.set_threadgroup_memory_length(0, tile.smem as NSUInteger);
        encoder.dispatch_thread_groups(
            MTLSize::new(
                nr.div_ceil(tile.x) as NSUInteger,
                nq.div_ceil(tile.y) as NSUInteger,
                1,
            ),
            MTLSize::new(tile.x as NSUInteger, tile.y as NSUInteger, 1),
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    });
    info!(
        "Metal rectangular Hamming kernel finished in {:.3}s",
        kernel_t0.elapsed().as_secs_f64()
    );

    let gpu_output =
        unsafe { std::slice::from_raw_parts(out_buffer.contents() as *const f32, nq * nr) };
    out.copy_from_slice(gpu_output);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::pairwise_hamming_rect_metal_u16;

    #[test]
    fn metal_matches_scalar_u16_hamming_with_tail() {
        if metal::Device::system_default().is_none() {
            eprintln!("skipping Metal test because no Metal device is visible");
            return;
        }

        let nq = 5usize;
        let nr = 7usize;
        let k = 19usize;
        let query: Vec<u16> = (0..nq * k)
            .map(|i| ((i * 7919 + i / 3) & 0xffff) as u16)
            .collect();
        let reference: Vec<u16> = (0..nr * k)
            .map(|i| ((i * 3571 + i / 5 + 1) & 0xffff) as u16)
            .collect();
        let mut actual = vec![0.0f32; nq * nr];

        pairwise_hamming_rect_metal_u16(&query, nq, &reference, nr, k, &mut actual)
            .expect("Metal Hamming failed");

        for qi in 0..nq {
            for rj in 0..nr {
                let a = &query[qi * k..(qi + 1) * k];
                let b = &reference[rj * k..(rj + 1) * k];
                let diff = a.iter().zip(b).filter(|(x, y)| x != y).count();
                let expected = diff as f32 / k as f32;
                assert_eq!(actual[qi * nr + rj], expected, "pair ({qi}, {rj})");
            }
        }
    }
}
