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
				l_res_2d := add_bias (l_res_2d, b)
			end
			
			if x.rank = 3 then
				Result := l_res_2d.reshape (<<x.shape [1], x.shape [2], weight.shape [1]>>)
			else
				Result := l_res_2d
			end
		end

feature {NONE} -- Helpers

	add_bias (res_2d: ET_TENSOR; b: ET_TENSOR): ET_TENSOR
			-- Adds 1D bias to 2D tensor (broadcasting over rows).
			-- Dispatches on float64 and float32 storage types correctly.
		local
			i, res_size, b_size: INTEGER_32
			l_store_f64: ET_STORAGE_REAL_64
			l_store_f32: ET_STORAGE_REAL_32
			l_strides: ARRAY [INTEGER_32]
		do
			res_size := res_2d.numel
			b_size := b.numel
			l_strides := res_2d.strides.deep_twin

			if res_2d.dtype.is_float32 then
				create l_store_f32.make (res_size)
				from i := 1 until i > res_size loop
					l_store_f32.put_real_32 (
						res_2d.storage.item_as_real_32 (res_2d.offset + i) +
						b.storage.item_as_real_32 (b.offset + ((i - 1) \\ b_size) + 1),
						i)
					i := i + 1
				end
				create Result.make_from_storage (l_store_f32, res_2d.shape.deep_twin, l_strides, 0)
				Result.set_dtype (res_2d.dtype)
			else
				-- Default: float64
				create l_store_f64.make (res_size)
				from i := 1 until i > res_size loop
					l_store_f64.put_real_64 (
						res_2d.storage.item_as_real_64 (res_2d.offset + i) +
						b.storage.item_as_real_64 (b.offset + ((i - 1) \\ b_size) + 1),
						i)
					i := i + 1
				end
				create Result.make_from_storage (l_store_f64, res_2d.shape.deep_twin, l_strides, 0)
				Result.set_dtype (res_2d.dtype)
			end
		end

end
