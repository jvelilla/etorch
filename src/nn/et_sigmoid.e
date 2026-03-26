note
	description: "[
		Sigmoid activation module.
		Equivalent to torch.nn.Sigmoid.
		Applies σ(x) = 1 / (1 + exp(-x)) element-wise.
		Commonly used for binary classification outputs.
	]"

class
	ET_SIGMOID

inherit
	ET_MODULE

create
	make

feature {NONE} -- Initialization

	make
			-- Create a Sigmoid module.
		do
		end

feature -- Access

	parameters: LIST [ET_PARAMETER]
			-- Sigmoid has no learnable parameters.
		do
			create {ARRAYED_LIST [ET_PARAMETER]} Result.make (0)
		end

feature -- Core Operation

	forward (x: ET_TENSOR): ET_TENSOR
			-- Apply Sigmoid activation to x.
		do
			Result := x.sigmoid
		end

end
