package main

import "core:simd"

attention_scores_simd :: proc(
	scores: []f32,
	query: []f32,
	key_cache: []f32,
	first_key: int,
	num_keys: int,
	cache_mask: int,
	head_dim: int,
) {
	q_ptr := cast([^]#simd[8]f32)&query[0]

	if head_dim == 256 {
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
