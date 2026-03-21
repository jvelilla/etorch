note
	description: "[
		Autograd function for softmax activation.
		Equivalent to PyTorch's `SoftmaxBackward`.
	]"

class
	ET_SOFTMAX_FUNCTION

inherit
	ET_FUNCTION

feature -- State

	saved_y: detachable ET_TENSOR
			-- Cache the forward output `y` (the probabilities)
			
	saved_dim: INTEGER_32
			-- Cache the dimension softmax was applied on

feature -- Operational

	forward_with_dim (inputs: ARRAY [ET_VALUE]; a_dim: INTEGER_32): ET_VALUE
			-- Computes softmax (x, dim).
		local
			t1, t_res: ET_TENSOR
		do
			t1 := inputs [1].data
			saved_dim := a_dim
			
			t_res := t1.softmax_internal (a_dim)
			saved_y := t_res
			
			create Result.make_with_parents (t_res, inputs, Current)
		end
		
	forward (inputs: ARRAY [ET_VALUE]): ET_VALUE
			-- Default forward, falls back to dim = rank
		do
			Result := forward_with_dim (inputs, inputs[1].data.rank)
		end

	backward (grad_output: ET_TENSOR): ARRAY [ET_TENSOR]
			-- Computes dL/dx = y * (grad_y - (y * grad_y).sum_dim(dim, keep_dim=True)).
		require else
			output_saved: saved_y /= Void
		local
			y_mul_grad_y: ET_TENSOR
			sum_y_mul_grad_y: ET_TENSOR
		do
			create Result.make_empty
			if attached saved_y as y then
				y_mul_grad_y := y.mul (grad_output)
				sum_y_mul_grad_y := y_mul_grad_y.sum_dim (saved_dim, True)
				
				-- grad_x = y * (grad_y - sum_y_mul_grad_y)
				Result.force (y.mul (grad_output.minus (sum_y_mul_grad_y)), 1)
			end
		end

end
