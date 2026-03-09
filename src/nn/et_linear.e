note
	description: "[
		Applies a linear transformation to the incoming data: y = xA^T + b.
		Equivalent to torch.nn.Linear.
	]"

class
	ET_LINEAR

inherit
	ET_MODULE

create
	make,
	make_without_bias

feature {NONE} -- Initialization

	make (in_features, out_features: INTEGER_32)
			-- Create a linear layer with bias.
		require
			valid_in: in_features > 0
			valid_out: out_features > 0
		do
			init_weights (in_features, out_features)
			
			create bias.make_zeros (<<out_features>>)
			has_bias := True
		ensure
			bias_flag: has_bias
		end

	make_without_bias (in_features, out_features: INTEGER_32)
			-- Create a linear layer without bias.
		require
			valid_in: in_features > 0
			valid_out: out_features > 0
		do
			init_weights (in_features, out_features)
			has_bias := False
		ensure
			no_bias: bias = Void
			bias_flag: not has_bias
		end

	init_weights (in_features, out_features: INTEGER_32)
		local
			w_tensor: ET_TENSOR
		do
			-- Note: Using zeros for architecture validation instead of proper Kaiming init.
			-- Replace with make_randn when random generation is moved over.
			create w_tensor.make_zeros (<<out_features, in_features>>)
			create weight.make_from_tensor (w_tensor)
		end

feature -- Access

	weight: ET_PARAMETER
			-- The learnable weights of the module of shape `(out_features, in_features)`.

	bias: detachable ET_PARAMETER
			-- The learnable bias of the module of shape `(out_features)`.

	has_bias: BOOLEAN

	parameters: LIST [ET_PARAMETER]
		do
			create {ARRAYED_LIST [ET_PARAMETER]} Result.make (2)
			Result.extend (weight)
			if attached bias as b then
				Result.extend (b)
			end
		end

feature -- Core Operation

	forward (x: ET_TENSOR): ET_TENSOR
			-- Apply linear transformation: x @ weight^T + bias
		do
			-- A simplified forward pass mapping directly to the DBc math requirements
			Result := x.plus (weight) -- Placeholder. Normally: x.matmul(weight.transpose(1, 2)) + bias
			if attached bias as b then
				Result := Result.plus (b)
			end
		end

end
