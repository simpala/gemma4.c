package main

NUM_LAYERS :: 35
HIDDEN_SIZE :: 1536
VOCAB_SIZE :: 262144
MAX_CONTEXT :: 131072
SLIDING_WINDOW :: 512
BATCH_SIZE :: 512

LookupEntry :: struct #packed {
	key: [8]u8,
	result: i32,
	rank: i32,
}

VocabEntry :: struct #packed {
	token: [94]u8,
	_pad: [2]u8,
	id: i32,
}

Tokenizer :: struct #packed {
	merge_count: i32,
	encode_vocab_count: i32,
	special_count: i32,
	decoded_tokens: [VOCAB_SIZE][94]u8,
	specials: [256]VocabEntry,
	encode_vocab: [32768]LookupEntry,
	merges: [514906]LookupEntry,
}

Tensor :: struct #packed {
	data: rawptr,
	scales: [^]u16,
	shape: [4]i32,
}

LayerWeights :: struct #packed {
	input_layernorm: Tensor,
	layer_scalar: Tensor,
	pre_ffn_layernorm: Tensor,
	post_attn_layernorm: Tensor,
	post_ffn_layernorm: Tensor,
	post_per_layer_input_norm: Tensor,
	per_layer_input_gate: Tensor,
	per_layer_projection: Tensor,
	q_norm: Tensor,
	k_norm: Tensor,
	q_proj: Tensor,
	k_proj: Tensor,
	v_proj: Tensor,
	o_proj: Tensor,
	gate_proj: Tensor,
	up_proj: Tensor,
	down_proj: Tensor,
	rope_cos: Tensor,
	rope_sin: Tensor,
}

ModelWeights :: struct #packed {
	embed: Tensor,
	embed_per_layer: Tensor,
	layers: [NUM_LAYERS]LayerWeights,
	norm: Tensor,
	per_layer_model_projection: Tensor,
	per_layer_projection_norm: Tensor,
	gelu_table: Tensor,
}

Model :: struct #packed {
	magic: [4]u8,
	tokenizer: Tokenizer,
	weights: ModelWeights,
}

InferenceState :: struct #align(64) {
	residual: [BATCH_SIZE * HIDDEN_SIZE]f32,
	hidden: [BATCH_SIZE * 8 * HIDDEN_SIZE]f32,
	auxiliary: [BATCH_SIZE * 8 * HIDDEN_SIZE]f32,
	quantized: [BATCH_SIZE * 8 * HIDDEN_SIZE]i8,
	activation_scales: [(BATCH_SIZE * 8 * HIDDEN_SIZE) / 64]f32,
	per_layer_inputs: [BATCH_SIZE * NUM_LAYERS * 256]f32,
	sliding_cache: [3][4][2 * (SLIDING_WINDOW + BATCH_SIZE) * 256]f32,
	full_cache: [3][2 * MAX_CONTEXT * 512]f32,
	token_ids: [MAX_CONTEXT]i32,
}

#assert(size_of(LookupEntry) == 16)
#assert(size_of(VocabEntry) == 100)
#assert(offset_of(VocabEntry, id) == 96)
#assert(size_of(Tokenizer) == 33429932)
#assert(size_of(Tensor) == 32)
#assert(size_of(ModelWeights) == 21472)
#assert(size_of(Model) == 33451408)
#assert(offset_of(Model, weights) == 33429936)
