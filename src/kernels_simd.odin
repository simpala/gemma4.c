package main

import "core:simd"

// ----------------------------------------------------------------------------
// SIMD Kernels (AVX2 primary using core:simd with alignment-safe operations)

matmul_int8_simd :: proc(
	output: []f32,
	input_q: []i8,
	input_scales: []f32,
	weight: ^Tensor,
	rows: int,
) {
	out_dim := int(weight.shape[0])
	block_rows :: 16
	num_output_blocks := out_dim / block_rows

	MatmulTaskData :: struct {
		output: []f32,
		input_q: []i8,
		input_scales: []f32,
		weight: ^Tensor,
		rows: int,
	}
	data := MatmulTaskData{output, input_q, input_scales, weight, rows}

	parallel_for(num_output_blocks, proc(start, end: int, user_data: rawptr) {
		td := cast(^MatmulTaskData)user_data
		in_dim := int(td.weight.shape[1])
		out_dim := int(td.weight.shape[0])
		groups_per_row := in_dim / 64
		packed_weights := cast([^]i8)td.weight.data
		scales := td.weight.scales

		for output_block in start ..< end {
			out_block_weights := packed_weights[output_block * 16 * in_dim:]
			for row in 0 ..< td.rows {
				in_row_q := td.input_q[row * in_dim:]
				in_row_scales := td.input_scales[row * groups_per_row:]

				for r in 0 ..< 16 {
					out_row_idx := output_block * 16 + r
					sum_acc: f32 = 0.0

					for group in 0 ..< groups_per_row {
						group_scale_f16 := scales[(output_block * groups_per_row + group) * 16 + r]
						group_scale := f16_to_f32(group_scale_f16) * in_row_scales[group]
						dot: i32 = 0

						for chunk in 0 ..< 16 {
							base_idx := group * 64 + chunk * 4
							w_idx := group * 1024 + chunk * 64 + (r / 8) * 32 + (r % 8)

							in0 := i32(in_row_q[base_idx + 0])
							in1 := i32(in_row_q[base_idx + 1])
							in2 := i32(in_row_q[base_idx + 2])
							in3 := i32(in_row_q[base_idx + 3])

							w0 := i32(out_block_weights[w_idx + 0 * 8])
							w1 := i32(out_block_weights[w_idx + 1 * 8])
							w2 := i32(out_block_weights[w_idx + 2 * 8])
							w3 := i32(out_block_weights[w_idx + 3 * 8])

							dot += in0 * w0 + in1 * w1 + in2 * w2 + in3 * w3
						}
						sum_acc += f32(dot) * group_scale
					}
					td.output[row * out_dim + out_row_idx] = sum_acc
				}
			}
		}
	}, &data)
}

attention_scores_simd :: proc(
	scores: []f32,
	query: []f32,
	key_cache: []f32,
	first_key: int,
	num_keys: int,
	cache_mask: int,
	head_dim: int,
) {
	for key_index in 0 ..< num_keys {
		key_pos := (first_key + key_index) & cache_mask
		key := key_cache[key_pos * head_dim:]
		sum_vec: #simd[8]f32 = {}
		j := 0
		for j + 8 <= head_dim {
			qv := simd.from_slice(#simd[8]f32, query[j:])
			kv := simd.from_slice(#simd[8]f32, key[j:])
			sum_vec += qv * kv
			j += 8
		}
		arr := simd.to_array(sum_vec)
		sum := arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7]
		for j < head_dim {
			sum += query[j] * key[j]
			j += 1
		}
		scores[key_index] = sum
	}
}

weighted_value_sum_simd :: proc(
	output: []f32,
	probabilities: []f32,
	value_cache: []f32,
	first_key: int,
	num_keys: int,
	cache_mask: int,
	head_dim: int,
) {
	for j in 0 ..< head_dim {
		output[j] = 0.0
	}

	for key_index in 0 ..< num_keys {
		prob := probabilities[key_index]
		prob_vec: #simd[8]f32 = prob
		key_pos := (first_key + key_index) & cache_mask
		val := value_cache[key_pos * head_dim:]

		j := 0
		for j + 8 <= head_dim {
			out_v := simd.from_slice(#simd[8]f32, output[j:])
			val_v := simd.from_slice(#simd[8]f32, val[j:])
			res_v := out_v + prob_vec * val_v
			simd.to_slice(output[j:], res_v)
			j += 8
		}
		for j < head_dim {
			output[j] += prob * val[j]
			j += 1
		}
	}
}
