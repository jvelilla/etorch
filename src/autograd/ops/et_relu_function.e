note
	description: "[
		Autograd function for ReLU activation.
		Equivalent to PyTorch's `ReluBackward`.
	]"

class
	ET_RELU_FUNCTION

inherit
	ET_FUNCTION

feature -- State

	saved_mask: detachable ET_TENSOR
			-- Boolean mask: True where input was > 0

feature -- Operational

	forward (inputs: ARRAY [ET_VALUE]): ET_VALUE
			-- Computes relu(x) = max(0, x).
		local
			t1, t_res: ET_TENSOR
		do
			t1 := inputs [1].data
			t_res := t1.relu_internal
				-- Save a mask: 1.0 where x > 0, else 0.0
			saved_mask := t1.relu_mask_internal
			create Result.make_with_parents (t_res, inputs, Current)
		end

	backward (grad_output: ET_TENSOR): ARRAY [ET_TENSOR]
			-- Computes dL/dx = grad_output * mask  (mask = 1 where x > 0, else 0).
		require else
			mask_saved: saved_mask /= Void
		do
			create Result.make_empty
			if attached saved_mask as mask then
				Result.force (grad_output.mul (mask), 1)
			end
		end

end
