note
	description: "[
		Tanh activation module.
		Equivalent to torch.nn.Tanh.
		Applies tanh(x) element-wise; output is in (-1, 1).
	]"

class
	ET_TANH

inherit
	ET_MODULE

create
	make

feature {NONE} -- Initialization

	make
			-- Create a Tanh module.
		do
		end

feature -- Access

	parameters: LIST [ET_PARAMETER]
			-- Tanh has no learnable parameters.
		do
			create {ARRAYED_LIST [ET_PARAMETER]} Result.make (0)
		end

feature -- Core Operation

	forward (x: ET_TENSOR): ET_TENSOR
			-- Apply Tanh activation to x.
		do
			Result := x.tanh
		end

end
