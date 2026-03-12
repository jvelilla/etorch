note
	description: "[
		MicroGPT Model.
		A simple multi-layer Transformer encoder/decoder stack.
	]"

class
	ET_MICROGPT

inherit
	ET_MODULE

create
	make

feature -- Initialization

	make (a_n_layer, a_n_embd, a_n_head: INTEGER_32; a_dropout: REAL_64)
			-- Initialize MicroGPT with `a_n_layer` transformer blocks.
		require
			valid_layers: a_n_layer > 0
			valid_embd: a_n_embd > 0
			valid_heads: a_n_head > 0 and (a_n_embd \\ a_n_head = 0)
		local
			i: INTEGER_32
			block: ET_TRANSFORMER_BLOCK
		do
			n_layer := a_n_layer
			n_embd := a_n_embd
			n_head := a_n_head

			create blocks.make (n_layer)
			from i := 1 until i > n_layer loop
				create block.make (n_embd, n_head, a_dropout)
				blocks.extend (block)
				i := i + 1
			end
			
			create ln_f.make (n_embd, 1.0e-5)
		end

feature -- Access

	blocks: ARRAYED_LIST [ET_TRANSFORMER_BLOCK]
	ln_f: ET_LAYER_NORM
	
	n_layer, n_embd, n_head: INTEGER_32

	parameters: LIST [ET_PARAMETER]
			-- Learnable parameters.
		local
			i: INTEGER_32
		do
			create {ARRAYED_LIST [ET_PARAMETER]} Result.make (0)
			from i := 1 until i > n_layer loop
				Result.append (blocks [i].parameters)
				i := i + 1
			end
			Result.append (ln_f.parameters)
		end

feature -- Operation

	forward (x: ET_TENSOR): ET_TENSOR
			-- Apply MicroGPT multi-layer transformation.
		local
			curr_x: ET_TENSOR
			i: INTEGER_32
		do
			curr_x := x
			from i := 1 until i > n_layer loop
				curr_x := blocks [i].forward (curr_x)
				i := i + 1
			end
			Result := ln_f.forward (curr_x)
		ensure then
			shape_preserved: Result.shape ~ x.shape
		end

end
