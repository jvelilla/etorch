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
			l_math: DOUBLE_MATH
		do
			create l_math
			create w_tensor.make_randn (<<out_features, in_features>>)
			w_tensor := w_tensor.mul_scalar (1.0 / l_math.sqrt (in_features.to_double))
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
		local
			l_x_2d, l_res_2d: ET_TENSOR
			l_w_t: ET_TENSOR
		do
			if x.rank = 3 then
				l_x_2d := x.reshape (<<x.shape [1] * x.shape [2], x.shape [3]>>)
			else
				l_x_2d := x
			end
			
			l_w_t := weight.transpose (1, 2)
			l_res_2d := l_x_2d.matmul (l_w_t)
			
			if attached bias as b then
				l_res_2d := l_res_2d.plus (b)
			end
			
			if x.rank = 3 then
				Result := l_res_2d.reshape (<<x.shape [1], x.shape [2], weight.shape [1]>>)
			else
				Result := l_res_2d
			end
		end

end
