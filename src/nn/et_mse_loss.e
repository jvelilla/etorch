note
	description: "[
		Mean Squared Error Loss.
		Equivalent to torch.nn.MSELoss.
	]"

class
	ET_MSE_LOSS

feature -- Core Operation

	forward (input, target: ET_TENSOR): ET_TENSOR
			-- Calculate the mean squared error.
		require
			same_shape: input.shape ~ target.shape or else input.is_broadcastable (target.shape)
		local
			diff, sq: ET_TENSOR
		do
			diff := input.minus (target)
			sq := diff.mul (diff)
			Result := sq.mean_all
		end

end
