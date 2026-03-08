note
	description: "[
		Transformer Block combining Attention and MLP with Layer Norms.
	]"

class
	ET_TRANSFORMER_BLOCK

inherit
	ET_MODULE

create
	make

feature -- Initialization

	make (n_embd, n_head: INTEGER_32; dropout: REAL_64)
			-- Initialize block.
		do
			create ln_1.make (n_embd, 1.0e-5)
			create attn.make (n_embd, n_head, dropout)
			create ln_2.make (n_embd, 1.0e-5)
			
			-- MLP equivalent to: [Linear(n_embd, 4*n_embd), GELU, Linear(4*n_embd, n_embd)]
			-- Keeping it straightforward using two linear layers as a placeholder
			create mlp_fc1.make (n_embd, 4 * n_embd)
			create mlp_fc2.make (4 * n_embd, n_embd)
		end

feature -- Access

	ln_1, ln_2: ET_LAYER_NORM
	attn: ET_MULTIHEAD_ATTENTION
	mlp_fc1, mlp_fc2: ET_LINEAR

	parameters: LIST [ET_PARAMETER]
			-- Learnable parameters.
		do
			create {ARRAYED_LIST [ET_PARAMETER]} Result.make (12)
			Result.append (ln_1.parameters)
			Result.append (attn.parameters)
			Result.append (ln_2.parameters)
			Result.append (mlp_fc1.parameters)
			Result.append (mlp_fc2.parameters)
		end

feature -- Operation

	forward (x: ET_TENSOR): ET_TENSOR
			-- Apply block transformation: x + attn(ln_1(x)) + mlp(ln_2(x)).
		local
			x_norm, attn_out: ET_TENSOR
			x_res: ET_TENSOR
			x_norm2, mlp_out: ET_TENSOR
		do
			-- Sublayer 1: Attention
			x_norm := ln_1.forward (x)
			attn_out := attn.forward (x_norm)
			x_res := x.plus (attn_out)

			-- Sublayer 2: MLP
			x_norm2 := ln_2.forward (x_res)
			mlp_out := mlp_fc1.forward (x_norm2)
			-- (Missing GELU here since it's not extracted yet)
			mlp_out := mlp_fc2.forward (mlp_out)

			Result := x_res.plus (mlp_out)
		ensure then
			shape_preserved: Result.shape ~ x.shape
		end

end
