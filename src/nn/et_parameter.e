note
	description: "[
		A kind of Tensor that is to be considered a module parameter.
		Parameters are Tensors that have `requires_grad=True` by default
		and are registered in the `parameters` list of `ET_MODULE` instances.
	]"

class
	ET_PARAMETER

inherit
	ET_TENSOR
		redefine
			make_zeros,
			make_ones,
			out
		end

create
	make_from_tensor,
	make_zeros,
	make_ones

feature {NONE} -- Initialization

	make_from_tensor (a_tensor: ET_TENSOR)
			-- Adopt an existing tensor as a parameter.
		do
			make_from_storage (a_tensor.storage, a_tensor.shape, a_tensor.strides, a_tensor.offset)
			requires_grad := True
			if attached ensure_grad_node as n then end
		ensure
			requires_grad: requires_grad
			has_node: grad_node /= Void
		end

	make_zeros (a_shape: ARRAY [INTEGER_32])
			-- Create a zero-initialized parameter.
		do
			Precursor (a_shape)
			requires_grad := True
		ensure then
			requires_grad: requires_grad
		end

	make_ones (a_shape: ARRAY [INTEGER_32])
			-- Create a one-initialized parameter.
		do
			Precursor (a_shape)
			requires_grad := True
		ensure then
			requires_grad: requires_grad
		end

feature -- Output

	out: STRING
		do
			Result := "Parameter containing:%N" + Precursor
		end

end
