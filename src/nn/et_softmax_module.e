note
	description: "[
		Softmax activation module.
		Equivalent to torch.nn.Softmax.
		Applies softmax over the specified dimension.
		Output values are in (0, 1) and sum to 1 along dim.
	]"

class
	ET_SOFTMAX_MODULE

inherit
	ET_MODULE

create
	make

feature {NONE} -- Initialization

	make (a_dim: INTEGER_32)
			-- Create a Softmax module for the given dimension.
		require
			valid_dim: a_dim >= 1
		do
			dim := a_dim
		ensure
			dim_set: dim = a_dim
		end

feature -- Access

	dim: INTEGER_32
			-- Dimension along which softmax is applied.

	parameters: LIST [ET_PARAMETER]
			-- Softmax has no learnable parameters.
		do
			create {ARRAYED_LIST [ET_PARAMETER]} Result.make (0)
		end

feature -- Core Operation

	forward (x: ET_TENSOR): ET_TENSOR
			-- Apply Softmax activation along `dim`.
		require else
			valid_dim_for_input: dim <= x.rank
		do
			Result := x.softmax (dim)
		end

end
