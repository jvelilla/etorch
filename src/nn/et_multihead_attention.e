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
			att, att_scores: ET_TENSOR
			dim_k: REAL_64
			math: DOUBLE_MATH
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

			-- 1. QKV projection
			query := q_proj.forward (x)
			k := k_proj.forward (x)
			v := v_proj.forward (x)
			
			if attached k_cache as kc and then attached v_cache as vc then
				-- Cache update logic: K and V are still [B, T, C] or [T, C]
				if cache_pos + T <= max_seq_len then
					if k.rank = 3 then
						kc.slice_range (2, cache_pos + 1, T).copy_from (k)
						vc.slice_range (2, cache_pos + 1, T).copy_from (v)
					else
						-- [T, C], reshape cache slice to match
						kc.slice_range (2, cache_pos + 1, T).view (<<T, C>>).copy_from (k)
						vc.slice_range (2, cache_pos + 1, T).view (<<T, C>>).copy_from (v)
					end
					
					cache_pos := cache_pos + T
					
					-- Then slice OUT the full historical KV for Attention
					-- We need the extracted k and v to represent [B, cache_pos, C]
					if k.rank = 3 then
						k := kc.slice_range (2, 1, cache_pos)
						v := vc.slice_range (2, 1, cache_pos)
					else
						k := kc.slice_range (2, 1, cache_pos).view (<<cache_pos, C>>)
						v := vc.slice_range (2, 1, cache_pos).view (<<cache_pos, C>>)
					end
				end
			end
			
			-- 2. Reshape & Transpose for multi-head computing
			-- from [B, T, C] to [B, T, n_head, head_size]
			if query.rank = 3 then
				query := query.reshape (<<B, T, n_head, head_size>>)
				k := k.reshape (<<B, k.shape[2], n_head, head_size>>)
				v := v.reshape (<<B, v.shape[2], n_head, head_size>>)
				
				-- from [B, T, n_head, head_size] to [B, n_head, T, head_size]
				query := query.transpose (2, 3)
				k := k.transpose (2, 3)
				v := v.transpose (2, 3)
			else
				-- [T, C] to [T, n_head, head_size] -> [n_head, T, head_size]
				query := query.reshape (<<T, n_head, head_size>>).transpose (1, 2)
				k := k.reshape (<<k.shape[1], n_head, head_size>>).transpose (1, 2)
				v := v.reshape (<<v.shape[1], n_head, head_size>>).transpose (1, 2)
			end

			-- 3. Scaled Dot-Product Attention: softmax((Q * K^T) / sqrt(d_k)) * V
			-- K^T: [B, n_head, head_size, T] (or [n_head, head_size, T])
			if query.rank = 4 then
				k := k.transpose (3, 4)
			else
				k := k.transpose (2, 3)
			end
			
			att_scores := query.matmul (k)
			create math
			dim_k := math.sqrt (head_size.to_double)
			att_scores := att_scores.div_scalar (dim_k)
			
			att := att_scores.softmax (att_scores.rank)
			
			-- att * V => [B, n_head, T, T] * [B, n_head, T, head_size] => [B, n_head, T, head_size]
			att := att.matmul (v)
			
			-- 4. Re-assemble heads
			-- [B, n_head, T, head_size] -> [B, T, n_head, head_size] -> [B, T, C]
			if query.rank = 4 then
				att := att.transpose (2, 3).contiguous.reshape (<<B, T, C>>)
			else
				att := att.transpose (1, 2).contiguous.reshape (<<T, C>>)
			end

			-- 5. Out projection
			Result := out_proj.forward (att)
		ensure then
			output_shape_matches: Result.shape ~ x.shape
		end

invariant
	valid_heads: n_embd \\ n_head = 0
end
