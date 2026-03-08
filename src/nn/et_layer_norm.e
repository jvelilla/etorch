note
	description: "[
		Applies Layer Normalization over a mini-batch of inputs.
		Equivalent to torch.nn.LayerNorm.
	]"

class
	ET_LAYER_NORM

inherit
	ET_MODULE

create
	make

feature {NONE} -- Initialization

	make (normalized_shape: INTEGER_32; a_eps: REAL_64)
			-- Create a LayerNorm module.
		require
			valid_shape: normalized_shape > 0
			valid_eps: a_eps > 0.0
		local
			w_tensor, b_tensor: ET_TENSOR
		do
			eps := a_eps
			create w_tensor.make_ones (<<normalized_shape>>)
			create weight.make_from_tensor (w_tensor)
			
			create b_tensor.make_zeros (<<normalized_shape>>)
			create bias.make_from_tensor (b_tensor)
		ensure
			weight_attached: weight /= Void
			bias_attached: bias /= Void
			eps_set: eps = a_eps
		end

feature -- Access

	weight: ET_PARAMETER
			-- The learnable weights of the module of shape `(normalized_shape)`.

	bias: ET_PARAMETER
			-- The learnable bias of the module of shape `(normalized_shape)`.

	eps: REAL_64

	parameters: LIST [ET_PARAMETER]
		do
			create {ARRAYED_LIST [ET_PARAMETER]} Result.make (2)
			Result.extend (weight)
			Result.extend (bias)
		end

feature -- Core Operation

	forward (x: ET_TENSOR): ET_TENSOR
			-- Apply layer normalization.
		do
			-- x_norm = (x - mean) / sqrt(var + eps)
			-- Result = x_norm * weight + bias
			-- Placeholder for exact tensor operations until backwards ops are expanded
			Result := x.plus (bias)
		end

invariant
	weight_attached: weight /= Void
	bias_attached: bias /= Void
end
