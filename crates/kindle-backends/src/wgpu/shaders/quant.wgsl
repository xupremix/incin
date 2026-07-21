// WGSL Quantization for Q8_0
// Q8_0 block is 34 bytes: f16 d (2 bytes) + 32x i8 qs (32 bytes).
// Two blocks are 68 bytes = 17 u32 words.
// We map 1 thread to process up to 2 blocks to avoid byte-write data races on u32 word boundaries.

@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_buf: array<u32>;

struct PushConstants {
    num_blocks: u32,
};
var<push_constant> pc: PushConstants;

fn pack2x16float(x: vec2<f32>) -> u32 {
    return pack2x16float(x); // Native WGSL builtin
}

@compute @workgroup_size(64, 1, 1)
fn quantize_q8_0(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pair_idx = global_id.x;
    let b0_idx = pair_idx * 2u;
    let b1_idx = pair_idx * 2u + 1u;

    if (b0_idx >= pc.num_blocks) {
        return;
    }

    // Process block 0
    var d0: f32 = 0.0;
    var qs0: array<u32, 8>; // 8 words = 32 bytes
    {
        let offset = b0_idx * 32u;
        var max_abs = 0.0;
        for (var i = 0u; i < 32u; i++) {
            let val = inp[offset + i];
            let abs_val = select(-val, val, val >= 0.0);
            if (abs_val > max_abs) { max_abs = abs_val; }
        }
        d0 = max_abs / 127.0;
        let inv_d = select(1.0 / d0, 0.0, d0 == 0.0);
        
        for (var w = 0u; w < 8u; w++) {
            var word = 0u;
            for (var b = 0u; b < 4u; b++) {
                let val = inp[offset + w * 4u + b];
                let q = i32(round(val * inv_d));
                let uq = u32(q) & 0xFFu;
                word = word | (uq << (b * 8u));
            }
            qs0[w] = word;
        }
    }

    // Process block 1 (if exists)
    var d1: f32 = 0.0;
    var qs1: array<u32, 8>;
    let has_b1 = b1_idx < pc.num_blocks;
    if (has_b1) {
        let offset = b1_idx * 32u;
        var max_abs = 0.0;
        for (var i = 0u; i < 32u; i++) {
            let val = inp[offset + i];
            let abs_val = select(-val, val, val >= 0.0);
            if (abs_val > max_abs) { max_abs = abs_val; }
        }
        d1 = max_abs / 127.0;
        let inv_d = select(1.0 / d1, 0.0, d1 == 0.0);
        
        for (var w = 0u; w < 8u; w++) {
            var word = 0u;
            for (var b = 0u; b < 4u; b++) {
                let val = inp[offset + w * 4u + b];
                let q = i32(round(val * inv_d));
                let uq = u32(q) & 0xFFu;
                word = word | (uq << (b * 8u));
            }
            qs1[w] = word;
        }
    }

    // Now write to out_buf cleanly
    let word_start = pair_idx * 17u; // 17 words per 2 blocks

    // Word 0: d0 (f16 in lower half, qs0[0] first 2 bytes in upper half)
    let d0_f16 = pack2x16float(vec2<f32>(d0, 0.0)); // lower 16 bits is d0
    let qs0_0 = qs0[0];
    out_buf[word_start + 0u] = (d0_f16 & 0xFFFFu) | ((qs0_0 & 0xFFFFu) << 16u);

    // Word 1: qs0[0] upper 2 bytes, qs0[1] lower 2 bytes
    out_buf[word_start + 1u] = (qs0_0 >> 16u) | ((qs0[1] & 0xFFFFu) << 16u);
    // Word 2: qs0[1] upper 2 bytes, qs0[2] lower 2 bytes
    out_buf[word_start + 2u] = (qs0[1] >> 16u) | ((qs0[2] & 0xFFFFu) << 16u);
    // Word 3: qs0[2] upper 2 bytes, qs0[3] lower 2 bytes
    out_buf[word_start + 3u] = (qs0[2] >> 16u) | ((qs0[3] & 0xFFFFu) << 16u);
    // Word 4: qs0[3] upper 2 bytes, qs0[4] lower 2 bytes
    out_buf[word_start + 4u] = (qs0[3] >> 16u) | ((qs0[4] & 0xFFFFu) << 16u);
    // Word 5: qs0[4] upper 2 bytes, qs0[5] lower 2 bytes
    out_buf[word_start + 5u] = (qs0[4] >> 16u) | ((qs0[5] & 0xFFFFu) << 16u);
    // Word 6: qs0[5] upper 2 bytes, qs0[6] lower 2 bytes
    out_buf[word_start + 6u] = (qs0[5] >> 16u) | ((qs0[6] & 0xFFFFu) << 16u);
    // Word 7: qs0[6] upper 2 bytes, qs0[7] lower 2 bytes
    out_buf[word_start + 7u] = (qs0[6] >> 16u) | ((qs0[7] & 0xFFFFu) << 16u);

    // Word 8 is special: qs0[7] upper 2 bytes, and IF has_b1, d1 lower 2 bytes
    if (has_b1) {
        let d1_f16 = pack2x16float(vec2<f32>(d1, 0.0));
        out_buf[word_start + 8u] = (qs0[7] >> 16u) | ((d1_f16 & 0xFFFFu) << 16u);

        let qs1_0 = qs1[0];
        out_buf[word_start + 9u] = (qs1_0 & 0xFFFFu) | ((qs1_0 >> 16u) << 16u); // Wait, qs1_0 is shifted? No.
        // Wait, word 8 has qs0[7] upper 16 bits and d1 16 bits.
        // So word 9 is just qs1[0] directly?
        // Let's trace bytes.
        // Block 1 starts at byte 34.
        // Word 8: bytes 32, 33 (qs0[7] upper 16), bytes 34, 35 (d1).
        // Word 9: bytes 36, 37, 38, 39, which is EXACTLY qs1[0] (bytes 0-3 of qs1).
        // Yes! Block 1 qs array is perfectly aligned to word boundaries from word 9 to word 16!
        out_buf[word_start + 9u]  = qs1[0];
        out_buf[word_start + 10u] = qs1[1];
        out_buf[word_start + 11u] = qs1[2];
        out_buf[word_start + 12u] = qs1[3];
        out_buf[word_start + 13u] = qs1[4];
        out_buf[word_start + 14u] = qs1[5];
        out_buf[word_start + 15u] = qs1[6];
        out_buf[word_start + 16u] = qs1[7];
    } else {
        out_buf[word_start + 8u] = (qs0[7] >> 16u);
    }
}
