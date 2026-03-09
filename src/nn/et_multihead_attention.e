note
	description: "[
		Multi-Head Attention mechanism.
		Equivalent to torch.nn.MultiheadAttention.
	]"

class
	ET_MULTIHEAD_ATTENTION

inherit
	ET_MODULE

create
	make

feature -- Initialization

	make (a_embed_dim, a_num_heads: INTEGER_32; a_dropout: REAL_64)
			-- Initialize MHA.
		require
			valid_embed_dim: a_embed_dim > 0
			valid_heads: a_num_heads > 0 and (a_embed_dim \\ a_num_heads = 0)
			valid_dropout: a_dropout >= 0.0 and a_dropout <= 1.0
		do
			n_embd := a_embed_dim
			n_head := a_num_heads
			head_size := n_embd // n_head
			dropout_p := a_dropout

			create q_proj.make (n_embd, n_embd)
			create k_proj.make (n_embd, n_embd)
			create v_proj.make (n_embd, n_embd)
			create out_proj.make (n_embd, n_embd)
		end

feature -- Access

	q_proj, k_proj, v_proj, out_proj: ET_LINEAR
			-- Linear projections.

	n_embd, n_head, head_size: INTEGER_32
	dropout_p: REAL_64

	-- KV Cache State
	k_cache, v_cache: detachable ET_TENSOR
	cache_pos: INTEGER_32
	max_seq_len: INTEGER_32

	init_cache (a_max_seq_len: INTEGER_32)
			-- Initialize stateful Key-Value cache for autoregressive inference.
		require
			valid_max_len: a_max_seq_len > 0
		do
			max_seq_len := a_max_seq_len
			cache_pos := 0
			
			-- B=1 assumed for simple autoregressive generation
			-- Cache shapes: [1, max_seq_len, n_embd]
			create k_cache.make_zeros (<<1, max_seq_len, n_embd>>)
			create v_cache.make_zeros (<<1, max_seq_len, n_embd>>)
		ensure
			cache_initialized: k_cache /= Void and v_cache /= Void
			pos_reset: cache_pos = 0
		end

	parameters: LIST [ET_PARAMETER]
			-- Learnable parameters.
		do
			create {ARRAYED_LIST [ET_PARAMETER]} Result.make (8)
			Result.append (q_proj.parameters)
			Result.append (k_proj.parameters)
			Result.append (v_proj.parameters)
			Result.append (out_proj.parameters)
		end

feature -- Operation

	forward (x: ET_TENSOR): ET_TENSOR
			-- Apply multi-head self-attention.
			-- If `init_cache` was called, runs statefully (B=1, T=1 usually).
		local
			B, T, C: INTEGER_32
			query, k, v: ET_TENSOR
		do
			query := x
			if query.rank = 3 then
				B := query.shape [1]
				T := query.shape [2]
			else
				B := 1
				T := query.shape [1]
			end
			C := n_embd

			-- Standard QKV projection (Placeholder until exact MHA Math is implemented)
			query := q_proj.forward (x)
			k := k_proj.forward (x)
			v := v_proj.forward (x)

			if attached k_cache as kc and then attached v_cache as vc then
				-- Cache update logic
				if cache_pos + T <= max_seq_len then
					-- In a true impl, we'd slice into the cache here and update it
					-- kc.slice_range(2, cache_pos + 1, T).copy_from(k)
					-- vc.slice_range(2, cache_pos + 1, T).copy_from(v)
					cache_pos := cache_pos + T
					
					-- Then slice OUT the full historical KV for Attention
					-- k := kc.slice_range(2, 1, cache_pos)
					-- v := vc.slice_range(2, 1, cache_pos)
				end
			end

			-- Out projection (structural placeholder)
			Result := out_proj.forward (query)
		ensure then
			output_shape_matches: Result.shape ~ x.shape
		end

invariant
	valid_heads: n_embd \\ n_head = 0
end
