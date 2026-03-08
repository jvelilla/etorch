note
	description: "[
		Autograd function for element-wise multiplication.
		Equivalent to PyTorch's `MulBackward`.
	]"

class
	ET_MUL_FUNCTION

inherit
	ET_FUNCTION

feature -- State

	saved_a, saved_b: detachable ET_TENSOR
			-- Cache inputs for backward pass

feature -- Operational

	forward (inputs: ARRAY [ET_VALUE]): ET_VALUE
			-- Computes a * b.
		local
			t1, t2, t_res: ET_TENSOR
		do
			t1 := inputs [1].data
			t2 := inputs [2].data
			
			saved_a := t1
			saved_b := t2
			
			-- For Phase 2, relying on a naive simulated multiplication for the graph
			-- A true numeric mul feature will be required in ET_TENSOR later
			t_res := t1.mul (t2)
			create Result.make_with_parents (t_res, inputs, Current)
		end

	backward (grad_output: ET_TENSOR): ARRAY [ET_TENSOR]
			-- Computes dL/da = grad_output * b, and dL/db = grad_output * a.
		do
			create Result.make_empty
			if attached saved_a as a and attached saved_b as b then
				Result.force (grad_output.mul (b), 1)
				Result.force (grad_output.mul (a), 2)
			end
		end

end
