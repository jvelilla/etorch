note
	description: "[
		ReLU (Rectified Linear Unit) activation module.
		Equivalent to torch.nn.ReLU.
		Applies relu(x) = max(0, x) element-wise.
	]"

class
	ET_RELU

inherit
	ET_MODULE

create
	make

feature {NONE} -- Initialization

	make
			-- Create a ReLU module.
		do
		end

feature -- Access

	parameters: LIST [ET_PARAMETER]
			-- ReLU has no learnable parameters.
		do
			create {ARRAYED_LIST [ET_PARAMETER]} Result.make (0)
		end

feature -- Core Operation

	forward (x: ET_TENSOR): ET_TENSOR
			-- Apply ReLU activation to x.
		do
			Result := x.relu
		end

end
