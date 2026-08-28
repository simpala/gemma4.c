package main

import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"
import "core:os"
import "core:sys/linux"
import "core:time"

// ----------------------------------------------------------------------------
// Model Loading & Mapping

map_model_file :: proc(path: string) -> (model: ^Model, file_size: int, err: bool) {
	fd, open_err := os.open(path, os.O_RDONLY)
	if open_err != nil {
		fmt.eprintfln("Failed to open model file '%s': %v", path, open_err)
		return nil, 0, true
	}
	defer os.close(fd)

	st, stat_err := os.fstat(fd, context.allocator)
	if stat_err != nil {
		fmt.eprintfln("Failed to stat '%s': %v", path, stat_err)
		return nil, 0, true
	}
	file_size = int(st.size)

	// Memory map using Linux sys call
	mmap_res, mmap_errno := linux.mmap(
		0,
		uint(file_size),
		{.READ, .WRITE},
		{.PRIVATE},
		linux.Fd(os.fd(fd)),
		0,
	)
	if mmap_errno != .NONE {
		fmt.eprintfln("mmap failed: %v", mmap_errno)
		return nil, 0, true
	}

	model = cast(^Model)mmap_res
	magic_str := string(model.magic[:3])
	if magic_str != "MOG" {
		fmt.eprintfln("bad model file magic: %s", magic_str)
		linux.munmap(mmap_res, uint(file_size))
		return nil, 0, true
	}

	// Fix pointers for all tensors in model.weights
	tensors := cast([^]Tensor)&model.weights
	tensor_count := size_of(ModelWeights) / size_of(Tensor)
	base_addr := uintptr(model)

	for i in 0 ..< tensor_count {
		if tensors[i].data != nil {
			tensors[i].data = rawptr(base_addr + uintptr(tensors[i].data))
		}
		if tensors[i].scales != nil {
			tensors[i].scales = cast([^]u16)(base_addr + uintptr(tensors[i].scales))
		}
	}

	return model, file_size, false
}

unmap_model_file :: proc(model: ^Model, file_size: int) {
	if model != nil {
		linux.munmap(rawptr(model), uint(file_size))
	}
}

// ----------------------------------------------------------------------------
// Forward Pass & Transformer Loop

attention_forward :: proc(
	state: ^InferenceState,
	layers: []LayerWeights,
	layer: int,
	start_pos: int,
	token_count: int,
	scores: []f32,
) {
	weights := &layers[layer]
	full_attention := (layer % 5) == 4
	cache_len := full_attention ? MAX_CONTEXT : SLIDING_WINDOW + BATCH_SIZE
	cache_mask := cache_len - 1
	head_dim := int(weights.q_norm.shape[0])
	query_width := int(weights.q_proj.shape[0])

	cache_owner := layer
	for layers[cache_owner].k_proj.data == nil || (cache_owner % 5 == 4) != full_attention {
		cache_owner -= 1
	}

	key_cache := full_attention ? state.full_cache[cache_owner / 5][:] : state.sliding_cache[cache_owner / 5][cache_owner % 5][:]
	value_cache := key_cache[cache_len * head_dim:]

	quantize_scalar(state.quantized[:], state.activation_scales[:], state.hidden[:], token_count, int(weights.q_proj.shape[1]))
	matmul_int8_scalar(state.auxiliary[:], state.quantized[:], state.activation_scales[:], &weights.q_proj, token_count)
	rmsnorm_scalar(state.auxiliary[:], state.auxiliary[:], &weights.q_norm, head_dim, 1e-6, token_count * (query_width / head_dim))
	apply_rope_scalar(&weights.rope_cos, &weights.rope_sin, state.auxiliary[:], query_width / head_dim, head_dim, start_pos, token_count)

	if weights.k_proj.data != nil {
		new_keys := key_cache[(start_pos & cache_mask) * head_dim:]
		new_values := value_cache[(start_pos & cache_mask) * head_dim:]
		matmul_int8_scalar(new_keys, state.quantized[:], state.activation_scales[:], &weights.k_proj, token_count)
		matmul_int8_scalar(new_values, state.quantized[:], state.activation_scales[:], &weights.v_proj, token_count)
		rmsnorm_scalar(new_keys, new_keys, &weights.k_norm, head_dim, 1e-6, token_count)
		rmsnorm_scalar(new_values, new_values, nil, head_dim, 1e-6, token_count)
		apply_rope_scalar(&weights.rope_cos, &weights.rope_sin, new_keys, 1, head_dim, start_pos, token_count)
	}

	num_heads := query_width / head_dim
	for head in 0 ..< num_heads {
		for token in 0 ..< token_count {
			first_key := 0
			if !full_attention && (start_pos + token + 1 > SLIDING_WINDOW) {
				first_key = start_pos + token + 1 - SLIDING_WINDOW
			}
			num_keys := start_pos + token + 1 - first_key
			head_output := state.hidden[token * query_width + head * head_dim:]
			query := state.auxiliary[token * query_width + head * head_dim:]
			attention_scores_simd(scores, query, key_cache, first_key, num_keys, cache_mask, head_dim)
			softmax_scalar(scores[:num_keys])
			weighted_value_sum_simd(head_output, scores[:num_keys], value_cache, first_key, num_keys, cache_mask, head_dim)
		}
	}

	quantize_scalar(state.quantized[:], state.activation_scales[:], state.hidden[:], token_count, int(weights.o_proj.shape[1]))
	matmul_int8_scalar(state.hidden[:], state.quantized[:], state.activation_scales[:], &weights.o_proj, token_count)
}

forward_pass :: proc(
	model: ^Model,
	state: ^InferenceState,
	tokens: []i32,
	token_count: int,
	start_pos: int,
	scores_buf: []f32,
) {
	per_layer_width := int(model.weights.per_layer_projection_norm.shape[0])
	embedding_scalar(state.residual[:], &model.weights.embed, tokens, token_count, math.sqrt(f32(HIDDEN_SIZE)))

	quantize_scalar(state.quantized[:], state.activation_scales[:], state.residual[:], token_count, HIDDEN_SIZE)
	matmul_int8_scalar(state.per_layer_inputs[:], state.quantized[:], state.activation_scales[:], &model.weights.per_layer_model_projection, token_count)
	rmsnorm_scalar(state.per_layer_inputs[:], state.per_layer_inputs[:], &model.weights.per_layer_projection_norm, per_layer_width, 1e-6 * HIDDEN_SIZE, token_count * NUM_LAYERS)

	embedding_scalar(state.hidden[:], &model.weights.embed_per_layer, tokens, token_count, math.sqrt(f32(per_layer_width)))
	add_and_scale_scalar(state.per_layer_inputs[:], state.hidden[:], token_count * NUM_LAYERS * per_layer_width, 1.0 / math.sqrt(f32(2.0)))

	for layer in 0 ..< NUM_LAYERS {
		weights := &model.weights.layers[layer]

		rmsnorm_scalar(state.hidden[:], state.residual[:], &weights.input_layernorm, HIDDEN_SIZE, 1e-6, token_count)
		attention_forward(state, model.weights.layers[:], layer, start_pos, token_count, scores_buf)
		rmsnorm_scalar(state.hidden[:], state.hidden[:], &weights.post_attn_layernorm, HIDDEN_SIZE, 1e-6, token_count)
		add_and_scale_scalar(state.residual[:], state.hidden[:], token_count * HIDDEN_SIZE, 1.0)

		rmsnorm_scalar(state.hidden[:], state.residual[:], &weights.pre_ffn_layernorm, HIDDEN_SIZE, 1e-6, token_count)
		quantize_scalar(state.quantized[:], state.activation_scales[:], state.hidden[:], token_count, int(weights.gate_proj.shape[1]))
		matmul_int8_scalar(state.hidden[:], state.quantized[:], state.activation_scales[:], &weights.gate_proj, token_count)
		matmul_int8_scalar(state.auxiliary[:], state.quantized[:], state.activation_scales[:], &weights.up_proj, token_count)
		geglu_scalar(state.hidden[:], state.auxiliary[:], token_count, int(weights.gate_proj.shape[0]), int(weights.gate_proj.shape[0]), &model.weights.gelu_table)
		quantize_scalar(state.quantized[:], state.activation_scales[:], state.hidden[:], token_count, int(weights.down_proj.shape[1]))
		matmul_int8_scalar(state.hidden[:], state.quantized[:], state.activation_scales[:], &weights.down_proj, token_count)
		rmsnorm_scalar(state.hidden[:], state.hidden[:], &weights.post_ffn_layernorm, HIDDEN_SIZE, 1e-6, token_count)
		add_and_scale_scalar(state.residual[:], state.hidden[:], token_count * HIDDEN_SIZE, 1.0)

		quantize_scalar(state.quantized[:], state.activation_scales[:], state.residual[:], token_count, HIDDEN_SIZE)
		matmul_int8_scalar(state.hidden[:], state.quantized[:], state.activation_scales[:], &weights.per_layer_input_gate, token_count)
		geglu_scalar(state.hidden[:], state.per_layer_inputs[layer * per_layer_width:], token_count, per_layer_width, NUM_LAYERS * per_layer_width, &model.weights.gelu_table)
		quantize_scalar(state.quantized[:], state.activation_scales[:], state.hidden[:], token_count, int(weights.per_layer_projection.shape[1]))
		matmul_int8_scalar(state.hidden[:], state.quantized[:], state.activation_scales[:], &weights.per_layer_projection, token_count)
		rmsnorm_scalar(state.hidden[:], state.hidden[:], &weights.post_per_layer_input_norm, HIDDEN_SIZE, 1e-6, token_count)

		layer_scalar_val := (cast([^]f32)weights.layer_scalar.data)[0]
		add_and_scale_scalar(state.residual[:], state.hidden[:], token_count * HIDDEN_SIZE, layer_scalar_val)
	}
}

compute_logits :: proc(model: ^Model, state: ^InferenceState, token_idx: int) -> []f32 {
	rmsnorm_scalar(state.hidden[:], state.residual[token_idx * HIDDEN_SIZE:], &model.weights.norm, HIDDEN_SIZE, 1e-6, 1)
	quantize_scalar(state.quantized[:], state.activation_scales[:], state.hidden[:], 1, HIDDEN_SIZE)
	matmul_int8_scalar(state.hidden[:], state.quantized[:], state.activation_scales[:], &model.weights.embed, 1)

	for i in 0 ..< VOCAB_SIZE {
		state.hidden[i] = 30.0 * math.tanh(state.hidden[i] / 30.0)
	}
	return state.hidden[:VOCAB_SIZE]
}

// ----------------------------------------------------------------------------
// Generation & Sampling

TopCandidate :: struct {
	score: f32,
	token: i32,
}

sample_logits :: proc(logits: []f32, vocab_size: int, temperature: f32, rng: ^rand.Generator) -> i32 {
	if temperature <= 0.0 {
		best := 0
		for i in 1 ..< vocab_size {
			if logits[i] > logits[best] {
				best = i
			}
		}
		return i32(best)
	}

	top: [64]TopCandidate
	for i in 0 ..< 64 {
		top[i].score = -math.INF_F32
	}
	for token in 0 ..< vocab_size {
		if logits[token] <= top[63].score {
			continue
		}
		i := 63
		for i > 0 && logits[token] > top[i - 1].score {
			top[i] = top[i - 1]
			i -= 1
		}
		top[i].score = logits[token]
		top[i].token = i32(token)
	}

	sum: f32 = 0.0
	max_score := top[0].score / temperature
	for i in 0 ..< 64 {
		top[i].score = math.exp(top[i].score / temperature - max_score)
		sum += top[i].score
	}

	mass: f32 = 0.0
	count := 0
	for mass < 0.95 * sum {
		mass += top[count].score
		count += 1
	}

	rnd_val := rand.float32(rng^)
	threshold := rnd_val * mass
	for i in 0 ..< count {
		threshold -= top[i].score
		if threshold <= 0.0 {
			return top[i].token
		}
	}
	return top[count - 1].token
}

prefill_tokens_pass :: proc(
	model: ^Model,
	state: ^InferenceState,
	tokens: []i32,
	token_count: int,
	dump_logits: bool,
	scores_buf: []f32,
) {
	for position := 0; position < token_count; position += BATCH_SIZE {
		chunk := token_count - position
		if chunk > BATCH_SIZE {
			chunk = BATCH_SIZE
		}
		if global_use_vulkan {
			run_vulkan_forward_pass(model, state, tokens[position:], chunk, position, scores_buf)
		} else {
			forward_pass(model, state, tokens[position:], chunk, position, scores_buf)
		}
		if dump_logits {
			for i in 0 ..< chunk {
				lg := compute_logits(model, state, i)
				raw_bytes := mem.slice_to_bytes(lg)
				os.write(os.stdout, raw_bytes)
			}
		}
	}
}

generate_text :: proc(
	model: ^Model,
	state: ^InferenceState,
	prompt: string,
	max_new_tokens: int,
	temperature: f32,
	dump_logits: bool,
	scores_buf: []f32,
	rng: ^rand.Generator,
) {
	tokenizer := &model.tokenizer
	styled := !dump_logits && os.is_tty(os.stdout)

	if max_new_tokens < 0 {
		fmt.eprintfln("-n must be non-negative")
		os.exit(1)
	}

	segments: [3]string = {
		dump_logits ? "" : "<|turn>user\n",
		prompt,
		dump_logits ? "" : "<turn|>\n<|turn>model\n",
	}

	prompt_tokens := tokenize(tokenizer, segments, state.token_ids[:])
	if prompt_tokens < 0 {
		fmt.eprintfln("prompt exceeds the %d-token context limit", MAX_CONTEXT)
		os.exit(1)
	}

	if styled {
		fmt.print("\n\x1b[2;36m────────────────────────────────\x1b[0m\n")
	}

	prefill_tokens_pass(model, state, state.token_ids[:prompt_tokens], prompt_tokens, dump_logits, scores_buf)
	if dump_logits {
		return
	}

	end := prompt_tokens + max_new_tokens
	if end > MAX_CONTEXT || end < prompt_tokens {
		end = MAX_CONTEXT
	}

	for position := prompt_tokens; position < end; position += 1 {
		tok_offset := position == prompt_tokens ? (prompt_tokens - 1) % BATCH_SIZE : 0
		lg := compute_logits(model, state, tok_offset)
		next_token := sample_logits(lg, VOCAB_SIZE, temperature, rng)

		if next_token == 1 || next_token == 106 {
			break
		}

		txt := token_text(tokenizer, next_token)
		fmt.print(txt)

		next_tok_arr: [1]i32 = {next_token}
		if global_use_vulkan {
			run_vulkan_forward_pass(model, state, next_tok_arr[:], 1, position, scores_buf)
		} else {
			forward_pass(model, state, next_tok_arr[:], 1, position, scores_buf)
		}
	}
	fmt.println()
}

benchmark_run :: proc(
	model: ^Model,
	state: ^InferenceState,
	prefill_tokens: int,
	generated_tokens: int,
	scores_buf: []f32,
	rng: ^rand.Generator,
) {
	if prefill_tokens > 0 {
		for i in 0 ..< prefill_tokens {
			state.token_ids[i] = 2 + i32(i % 1000)
		}
		start := time.now()
		prefill_tokens_pass(model, state, state.token_ids[:prefill_tokens], prefill_tokens, false, scores_buf)
		elapsed := time.duration_seconds(time.since(start))
		fmt.printf("pp%d %.2f tok/s\n", prefill_tokens, f64(prefill_tokens) / elapsed)
	}
	if generated_tokens > 0 {
		token: i32 = 2
		start := time.now()
		for position in 0 ..< generated_tokens {
			tok_arr: [1]i32 = {token}
			forward_pass(model, state, tok_arr[:], 1, position, scores_buf)
			lg := compute_logits(model, state, 0)
			token = sample_logits(lg, VOCAB_SIZE, 0.0, rng)
		}
		elapsed := time.duration_seconds(time.since(start))
		fmt.printf("tg%d %.2f tok/s\n", generated_tokens, f64(generated_tokens) / elapsed)
	}
}

// ----------------------------------------------------------------------------
global_use_vulkan: bool

// Main CLI Entry

main :: proc() {
	model_path := "gemma4-E2B-int8.bin"
	prompt := "Why is the sky blue?"
	temperature: f32 = 1.0
	max_new_tokens := 1024
	benchmark_mode := false
	dump_logits := false
	prefill_tokens := 0
	generated_tokens := 256
	use_vulkan_flag := false

	args := os.args[1:]
	i := 0
	for i < len(args) {
		arg := args[i]
		if arg == "-m" && i + 1 < len(args) {
			i += 1
			model_path = args[i]
		} else if arg == "-t" && i + 1 < len(args) {
			i += 1
			temperature = f32(atof(args[i]))
		} else if arg == "-n" && i + 1 < len(args) {
			i += 1
			max_new_tokens = atoi(args[i])
		} else if arg == "--bench" {
			benchmark_mode = true
			if i + 1 < len(args) && args[i + 1][0] != '-' {
				i += 1
				prefill_tokens = atoi(args[i])
			}
			if i + 1 < len(args) && args[i + 1][0] != '-' {
				i += 1
				generated_tokens = atoi(args[i])
			}
		} else if arg == "--dump-logits" {
			dump_logits = true
		} else if arg == "-cpu" {
			use_vulkan_flag = false
		} else if arg == "-vk" {
			use_vulkan_flag = true
		} else {
			prompt = arg
		}
		i += 1
	}

	init_thread_pool()
	defer destroy_thread_pool()

	model, file_size, err := map_model_file(model_path)
	if err {
		os.exit(1)
	}
	defer unmap_model_file(model, file_size)

	state := new(InferenceState)
	defer free(state)

	scores_buf := make([]f32, MAX_CONTEXT + BATCH_SIZE)
	defer delete(scores_buf)

	seed := u64(time.to_unix_nanoseconds(time.now()))
	pcg_state := rand.PCG_Random_State{state = seed, inc = 1}
	rng_val := rand.pcg_random_generator(&pcg_state)

	if use_vulkan_flag {
		global_use_vulkan = true
		fmt.println("Vulkan acceleration requested (-vk)")
		if !init_vulkan_backend() {
			fmt.eprintfln("Warning: Vulkan initialization failed. Falling back to CPU execution.")
		}
	}
	defer destroy_vulkan_backend()

	if benchmark_mode {
		benchmark_run(model, state, prefill_tokens, generated_tokens, scores_buf, &rng_val)
	} else {
		generate_text(model, state, prompt, max_new_tokens, temperature, dump_logits, scores_buf, &rng_val)
	}
}

// Simple string to int / float helpers
atoi :: proc(s: string) -> int {
	res := 0
	sign := 1
	str := s
	if len(str) > 0 && str[0] == '-' {
		sign = -1
		str = str[1:]
	}
	for b in str {
		if b >= '0' && b <= '9' {
			res = res * 10 + int(b - '0')
		} else {
			break
		}
	}
	return res * sign
}

atof :: proc(s: string) -> f64 {
	val: f64 = 0.0
	sign: f64 = 1.0
	str := s
	if len(str) > 0 && str[0] == '-' {
		sign = -1.0
		str = str[1:]
	}
	dec := false
	div: f64 = 1.0
	for b in str {
		if b == '.' {
			dec = true
			continue
		}
		if b >= '0' && b <= '9' {
			if dec {
				div *= 10.0
				val = val + f64(b - '0') / div
			} else {
				val = val * 10.0 + f64(b - '0')
			}
		} else {
			break
		}
	}
	return val * sign
}
