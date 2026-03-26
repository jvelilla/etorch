note
	description: "[
		Autograd function for Tanh activation.
		Equivalent to PyTorch's `TanhBackward`.
		Forward:  tanh(x)
		Backward: dL/dx = grad * (1 - tanh(x)^2)
	]"

class
	ET_TANH_FUNCTION

inherit
	ET_FUNCTION

feature -- State

	saved_y: detachable ET_TENSOR
			-- Cache forward output tanh(x) for use in backward

feature -- Operational

	forward (inputs: ARRAY [ET_VALUE]): ET_VALUE
			-- Computes tanh(x).
		local
			t1, t_res: ET_TENSOR
		do
			t1 := inputs [1].data
			t_res := t1.tanh_val
			saved_y := t_res
			create Result.make_with_parents (t_res, inputs, Current)
		end

	backward (grad_output: ET_TENSOR): ARRAY [ET_TENSOR]
			-- Computes dL/dx = grad * (1 - tanh(x)^2).
		require else
			output_saved: saved_y /= Void
		local
			tanh_sq, one_minus_tanh_sq: ET_TENSOR
		do
			create Result.make_empty
			if attached saved_y as y then
					-- tanh^2
				tanh_sq := y.mul (y)
					-- 1 - tanh^2
				one_minus_tanh_sq := tanh_sq.mul_scalar (-1.0).plus_scalar (1.0)
					-- grad * (1 - tanh^2)
				Result.force (grad_output.mul (one_minus_tanh_sq), 1)
			end
		end

end
