package main

import "core:math"

f16_to_f32 :: proc(h: u16) -> f32 {
	sign := u32(h >> 15) & 0x0001
	exp  := u32(h >> 10) & 0x001F
	mant := u32(h) & 0x03FF

	if exp == 0 {
		if mant == 0 {
			return sign != 0 ? -0.0 : 0.0
		}
		for (mant & 0x0400) == 0 {
			mant <<= 1
			exp -= 1
		}
		exp += 1
		mant &= 0x03FF
	} else if exp == 31 {
		if mant == 0 {
			return sign != 0 ? -math.INF_F32 : math.INF_F32
		}
		return math.nan_f32()
	}

	exp = exp + (127 - 15)
	mant = mant << 13
	bits := (sign << 31) | (exp << 23) | mant
	return transmute(f32)bits
}

matmul_int8_scalar :: proc(
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
							for c in 0 ..< 4 {
								in_val := i32(in_row_q[group * 64 + chunk * 4 + c])
								w_idx := group * 1024 + chunk * 64 + (r / 8) * 32 + c * 8 + (r % 8)
								w_val := i32(out_block_weights[w_idx])
								dot += in_val * w_val
							}
						}
						sum_acc += f32(dot) * group_scale
					}
					td.output[row * out_dim + out_row_idx] = sum_acc
				}
			}
		}
	}, &data)
}

quantize_scalar :: proc(quantized: []i8, scales: []f32, input: []f32, rows: int, width: int) {
	groups_per_row := width / 64
	total_groups := rows * groups_per_row

	QuantData :: struct {
		quantized: []i8,
		scales: []f32,
		input: []f32,
	}
	data := QuantData{quantized, scales, input}

	parallel_for(total_groups, proc(start, end: int, user_data: rawptr) {
		td := cast(^QuantData)user_data
		for group_index in start ..< end {
			group := td.input[group_index * 64:]
			max_abs: f32 = 0.0
			for j in 0 ..< 64 {
				val := math.abs(group[j])
				if val > max_abs {
					max_abs = val
				}
			}
			scale := max_abs / 127.0
			inv_scale: f32 = scale > 0.0 ? 1.0 / scale : 0.0
			for j in 0 ..< 64 {
				td.quantized[group_index * 64 + j] = i8(math.round(group[j] * inv_scale))
			}
			td.scales[group_index] = scale
		}
	}, &data)
}

attention_scores_scalar :: proc(
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
		sum: f32 = 0.0
		for j in 0 ..< head_dim {
			sum += query[j] * key[j]
		}
		scores[key_index] = sum
	}
}

weighted_value_sum_scalar :: proc(
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
		key_pos := (first_key + key_index) & cache_mask
		val := value_cache[key_pos * head_dim:]
		for j in 0 ..< head_dim {
			output[j] += prob * val[j]
		}
	}
}

geglu_scalar :: proc(
	gate: []f32,
	up: []f32,
	rows: int,
	width: int,
	up_stride: int,
	gelu_table: ^Tensor,
) {
	table := cast([^]f32)gelu_table.data
	table_size := int(gelu_table.shape[0])
	lower := f32(gelu_table.shape[1])
	upper := f32(gelu_table.shape[2])
	scale := f32(table_size - 1) / (upper - lower)

	GegluData :: struct {
		gate: []f32,
		up: []f32,
		width: int,
		up_stride: int,
		table: [^]f32,
		lower: f32,
		upper: f32,
		scale: f32,
	}
	data := GegluData{gate, up, width, up_stride, table, lower, upper, scale}

	parallel_for(rows, proc(start, end: int, user_data: rawptr) {
		td := cast(^GegluData)user_data
		for row in start ..< end {
			for i in 0 ..< td.width {
				x := td.gate[row * td.width + i]
				if x <= td.lower {
					x = td.table[0]
				} else if !(x >= td.upper) {
					pos := (x - td.lower) * td.scale
					idx := int(pos)
					fraction := pos - f32(idx)
					x = td.table[idx] + fraction * (td.table[idx + 1] - td.table[idx])
				}
				td.gate[row * td.width + i] = x * td.up[row * td.up_stride + i]
			}
		}
	}, &data)
}

rmsnorm_scalar :: proc(
	output: []f32,
	input: []f32,
	weight: ^Tensor,
	width: int,
	epsilon: f32,
	row_count: int,
) {
	weights := weight != nil ? cast([^]f32)weight.data : nil

	RmsData :: struct {
		output: []f32,
		input: []f32,
		weights: [^]f32,
		width: int,
		epsilon: f32,
	}
	data := RmsData{output, input, weights, width, epsilon}

	parallel_for(row_count, proc(start, end: int, user_data: rawptr) {
		td := cast(^RmsData)user_data
		for row in start ..< end {
			input_row := td.input[row * td.width:]
			output_row := td.output[row * td.width:]
			sum_sq: f32 = 0.0
			for i in 0 ..< td.width {
				sum_sq += input_row[i] * input_row[i]
			}
			inv_rms := 1.0 / math.sqrt(sum_sq / f32(td.width) + td.epsilon)
			for i in 0 ..< td.width {
				w := td.weights != nil ? td.weights[i] : 1.0
				output_row[i] = w * (inv_rms * input_row[i])
			}
		}
	}, &data)
}

add_and_scale_scalar :: proc(output: []f32, addend: []f32, count: int, scale: f32) {
	AddScaleData :: struct {
		output: []f32,
		addend: []f32,
		scale: f32,
	}
	data := AddScaleData{output, addend, scale}

	parallel_for(count, proc(start, end: int, user_data: rawptr) {
		td := cast(^AddScaleData)user_data
		for i in start ..< end {
			td.output[i] = (td.output[i] + td.addend[i]) * td.scale
		}
	}, &data)
}

apply_rope_scalar :: proc(
	cosines: ^Tensor,
	sines: ^Tensor,
	vectors: []f32,
	num_heads: int,
	head_dim: int,
	start_pos: int,
	token_count: int,
) {
	pairs := int(cosines.shape[1])
	cos_data := cast([^]f32)cosines.data
	sin_data := cast([^]f32)sines.data

	RopeData :: struct {
		cos_data: [^]f32,
		sin_data: [^]f32,
		vectors: []f32,
		pairs: int,
		num_heads: int,
		head_dim: int,
		start_pos: int,
	}
	data := RopeData{cos_data, sin_data, vectors, pairs, num_heads, head_dim, start_pos}

	parallel_for(token_count, proc(start, end: int, user_data: rawptr) {
		td := cast(^RopeData)user_data
		for token in start ..< end {
			cosine := td.cos_data[(td.start_pos + token) * td.pairs:]
			sine := td.sin_data[(td.start_pos + token) * td.pairs:]
			for head in 0 ..< td.num_heads {
				vector := td.vectors[(token * td.num_heads + head) * td.head_dim:]
				for j in 0 ..< td.pairs {
					first := vector[j]
					second := vector[j + td.head_dim / 2]
					vector[j] = first * cosine[j] - second * sine[j]
					vector[j + td.head_dim / 2] = second * cosine[j] + first * sine[j]
				}
			}
		}
	}, &data)
}

softmax_scalar :: proc(values: []f32) {
	if len(values) == 0 {
		return
	}
	max_val := values[0]
	sum: f32 = 1.0
	for i in 1 ..< len(values) {
		if values[i] > max_val {
			sum = sum * math.exp(max_val - values[i]) + 1.0
			max_val = values[i]
		} else {
			sum += math.exp(values[i] - max_val)
		}
	}
	for i in 0 ..< len(values) {
		values[i] = math.exp(values[i] - max_val) / sum
	}
}

embedding_scalar :: proc(output: []f32, table: ^Tensor, tokens: []i32, token_count: int, multiplier: f32) {
	block_rows :: 16
	width := int(table.shape[1])
	groups := width / 64
	table_data := cast([^]i8)table.data
	table_scales := table.scales

	EmbedData :: struct {
		output: []f32,
		tokens: []i32,
		multiplier: f32,
		table_data: [^]i8,
		table_scales: [^]u16,
		width: int,
		groups: int,
	}
	data := EmbedData{output, tokens, multiplier, table_data, table_scales, width, groups}

	parallel_for(token_count, proc(start, end: int, user_data: rawptr) {
		td := cast(^EmbedData)user_data
		for token in start ..< end {
			tok_id := int(td.tokens[token])
			block := tok_id / 16
			row := tok_id % 16
			vector := td.output[token * td.width:]
			block_data := td.table_data[block * 16 * td.width:]
			block_scales := td.table_scales[block * td.groups * 16:]

			for group_index in 0 ..< td.groups {
				group := block_data[group_index * 16 * 64:]
				scale := f16_to_f32(block_scales[group_index * 16 + row]) * td.multiplier
				for j in 0 ..< 64 {
					chunk := j / 4
					offset := j % 4
					byte_val := i32(group[chunk * 16 * 4 + row * 4 + offset])
					vector[group_index * 64 + j] = f32(byte_val) * scale
				}
			}
		}
	}, &data)
}
