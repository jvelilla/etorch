note
	description: "[
		Autograd function for element-wise addition.
		Equivalent to PyTorch's `AddBackward`.
	]"

class
	ET_ADD_FUNCTION

inherit
	ET_FUNCTION

feature -- Operational

	forward (inputs: ARRAY [ET_VALUE]): ET_VALUE
			-- Computes a + b.
		local
			t1, t2, t_res: ET_TENSOR
		do
			t1 := inputs [1].data
			t2 := inputs [2].data
			t_res := t1.plus (t2)
			create Result.make_with_parents (t_res, inputs, Current)
		end

	backward (grad_output: ET_TENSOR): ARRAY [ET_TENSOR]
			-- Computes dL/da and dL/db.
			-- grad_a = grad_output
			-- grad_b = grad_output (assuming no broadcasting adjustments for now)
		do
			create Result.make_empty
			Result.force (grad_output, 1)
			Result.force (grad_output, 2)
		end

end
