note
	description: "[
		Autograd function for Sigmoid activation.
		Equivalent to PyTorch's `SigmoidBackward`.
		Forward:  σ(x) = 1 / (1 + exp(-x))
		Backward: dL/dx = grad * σ(x) * (1 - σ(x))
	]"

class
	ET_SIGMOID_FUNCTION

inherit
	ET_FUNCTION

feature -- State

	saved_y: detachable ET_TENSOR
			-- Cache forward output σ(x) for use in backward

feature -- Operational

	forward (inputs: ARRAY [ET_VALUE]): ET_VALUE
			-- Computes sigmoid(x).
		local
			t1, t_res: ET_TENSOR
		do
			t1 := inputs [1].data
			t_res := t1.sigmoid_internal
			saved_y := t_res
			create Result.make_with_parents (t_res, inputs, Current)
		end

	backward (grad_output: ET_TENSOR): ARRAY [ET_TENSOR]
			-- Computes dL/dx = grad * σ(x) * (1 - σ(x)).
		require else
			output_saved: saved_y /= Void
		local
			one_minus_y: ET_TENSOR
		do
			create Result.make_empty
			if attached saved_y as y then
					-- (1 - y)
				one_minus_y := y.mul_scalar (-1.0).plus_scalar (1.0)
					-- grad * y * (1 - y)
				Result.force (grad_output.mul (y).mul (one_minus_y), 1)
			end
		end

end
