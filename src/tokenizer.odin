package main

import "core:mem"

lookup_binary_search :: proc(key: ^[8]u8, entries: []LookupEntry) -> ^LookupEntry {
	low := 0
	high := len(entries) - 1
	for low <= high {
		mid := low + (high - low) / 2
		cmp := mem.compare(key[:], entries[mid].key[:])
		if cmp == 0 {
			return &entries[mid]
		} else if cmp < 0 {
			high = mid - 1
		} else {
			low = mid + 1
		}
	}
	return nil
}

apply_bpe_merges :: proc(tokenizer: ^Tokenizer, tokens: []i32) -> int {
	count := len(tokens)
	merges := tokenizer.merges[:tokenizer.merge_count]
	for {
		best_merge: ^LookupEntry = nil
		position := -1
		for i in 0 ..< count - 1 {
			pair_key: [8]u8
			mem.copy(&pair_key[0], &tokens[i], 8)
			merge := lookup_binary_search(&pair_key, merges)
			if merge != nil && (best_merge == nil || merge.rank < best_merge.rank) {
				best_merge = merge
				position = i
			}
		}
		if best_merge == nil {
			return count
		}

		tokens[position] = best_merge.result
		for i in position + 1 ..< count - 1 {
			tokens[i] = tokens[i + 1]
		}
		count -= 1
	}
}

tokenize :: proc(tokenizer: ^Tokenizer, segments: [3]string, tokens: []i32) -> int {
	capacity := len(tokens)
	count := 1

	for segment_idx in 0 ..< 3 {
		cursor := segments[segment_idx]
		for len(cursor) > 0 {
			if count >= capacity {
				return -1
			}

			special := -1
			if cursor[0] == '<' {
				for i in 0 ..< int(tokenizer.special_count) {
					spec_tok_bytes := tokenizer.specials[i].token[:]
					spec_len := 0
					for spec_len < len(spec_tok_bytes) && spec_tok_bytes[spec_len] != 0 {
						spec_len += 1
					}
					if spec_len > 0 && len(cursor) >= spec_len {
						if cursor[:spec_len] == string(spec_tok_bytes[:spec_len]) {
							special = int(tokenizer.specials[i].id)
							cursor = cursor[spec_len:]
							break
						}
					}
				}
			}

			if special >= 0 {
				tokens[count] = i32(special)
				count += 1
				continue
			}

			piece: [8]u8
			if cursor[0] == ' ' {
				piece[0] = 0xE2
				piece[1] = 0x96
				piece[2] = 0x81
				cursor = cursor[1:]
			} else {
				piece[0] = cursor[0]
				cursor = cursor[1:]
				if (piece[0] & 0xC0) == 0xC0 {
					for i in 1 ..< 4 {
						if len(cursor) > 0 && (cursor[0] & 0xC0) == 0x80 {
							piece[i] = cursor[0]
							cursor = cursor[1:]
						} else {
							break
						}
					}
				}
			}

			encode_vocab := tokenizer.encode_vocab[:tokenizer.encode_vocab_count]
			entry := lookup_binary_search(&piece, encode_vocab)
			if entry != nil {
				tokens[count] = entry.result
				count += 1
				continue
			}

			for b in piece {
				if b == 0 {
					break
				}
				if count >= capacity {
					return -1
				}
				tokens[count] = 238 + i32(b)
				count += 1
			}
		}
	}

	merged_len := apply_bpe_merges(tokenizer, tokens[1:count])
	count = 1 + merged_len
	tokens[0] = 2 // <bos>
	return count
}

token_text :: proc(tokenizer: ^Tokenizer, token: i32) -> string {
	if token >= 0 && token < VOCAB_SIZE {
		str_bytes := tokenizer.decoded_tokens[token][:]
		str_len := 0
		for str_len < len(str_bytes) && str_bytes[str_len] != 0 {
			str_len += 1
		}
		return string(str_bytes[:str_len])
	}
	return ""
}
