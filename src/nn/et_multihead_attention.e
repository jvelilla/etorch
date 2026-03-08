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
			-- Apply multi-head self-attention (query = key = value = x).
			-- Note: Shapes expected are [B, T, C].
		local
			B, T, C: INTEGER_32
			query: ET_TENSOR
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

			-- For now, returning a structural placeholder until all math ops are extracted.
			-- In Phase 3 or later, this will perform exact Q K V matmuls and Softmax.
			Result := out_proj.forward (query)
		ensure then
			output_shape_matches: Result.shape ~ x.shape
		end

invariant
	valid_heads: n_embd \\ n_head = 0
end
