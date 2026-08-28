package main

import "core:simd"

// ----------------------------------------------------------------------------
// SIMD Kernels (AVX2 primary using core:simd)

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
							// 4-byte packed SIMD int8 chunk dot product
							in_ptr := cast(^#simd[4]i8)&in_row_q[group * 64 + chunk * 4]
							w_idx := group * 1024 + chunk * 64 + (r / 8) * 32 + (r % 8)
							// Load weights with stride 8
							w0 := i32(out_block_weights[w_idx + 0 * 8])
							w1 := i32(out_block_weights[w_idx + 1 * 8])
							w2 := i32(out_block_weights[w_idx + 2 * 8])
							w3 := i32(out_block_weights[w_idx + 3 * 8])

							in_arr := simd.to_array(in_ptr^)
							dot += i32(in_arr[0]) * w0 + i32(in_arr[1]) * w1 + i32(in_arr[2]) * w2 + i32(in_arr[3]) * w3
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
	if head_dim == 256 {
		q_ptr := cast([^]#simd[8]f32)&query[0]
		q0 := q_ptr[0]; q1 := q_ptr[1]; q2 := q_ptr[2]; q3 := q_ptr[3]
		q4 := q_ptr[4]; q5 := q_ptr[5]; q6 := q_ptr[6]; q7 := q_ptr[7]
		q8 := q_ptr[8]; q9 := q_ptr[9]; q10 := q_ptr[10]; q11 := q_ptr[11]

		for key_index in 0 ..< num_keys {
			key_pos := (first_key + key_index) & cache_mask
			k_ptr := cast([^]#simd[8]f32)&key_cache[key_pos * 256]

			sum0 := q0 * k_ptr[0] + q1 * k_ptr[1]
			sum1 := q2 * k_ptr[2] + q3 * k_ptr[3]
			sum2 := q4 * k_ptr[4] + q5 * k_ptr[5]
			sum3 := q6 * k_ptr[6] + q7 * k_ptr[7]
			sum4 := q8 * k_ptr[8] + q9 * k_ptr[9]
			sum5 := q10 * k_ptr[10] + q11 * k_ptr[11]

			total := sum0 + sum1 + sum2 + sum3 + sum4 + sum5
			arr := simd.to_array(total)
			scores[key_index] = arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7]
		}
		return
	}

	for key_index in 0 ..< num_keys {
		key_pos := (first_key + key_index) & cache_mask
		k_slice := key_cache[key_pos * head_dim:]
		sum_vec: #simd[8]f32 = {}
		j := 0
		for j + 8 <= head_dim {
			qv := (cast(^#simd[8]f32)&query[j])^
			kv := (cast(^#simd[8]f32)&k_slice[j])^
			sum_vec += qv * kv
			j += 8
		}
		arr := simd.to_array(sum_vec)
		sum := arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7]
		for j < head_dim {
			sum += query[j] * k_slice[j]
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

	out_simd := cast([^]#simd[8]f32)&output[0]
	num_vecs := head_dim / 8

	for key_index in 0 ..< num_keys {
		prob := probabilities[key_index]
		prob_vec: #simd[8]f32 = prob
		key_pos := (first_key + key_index) & cache_mask
		val_simd := cast([^]#simd[8]f32)&value_cache[key_pos * head_dim]

		for v in 0 ..< num_vecs {
			out_simd[v] = out_simd[v] + prob_vec * val_simd[v]
		}
	}
}
