note
	description: "[
		Autograd function for element-wise addition.
		Equivalent to PyTorch's `AddBackward`.
	]"

class
	ET_ADD_FUNCTION

inherit
	ET_FUNCTION

feature -- State

	saved_a_shape, saved_b_shape: detachable ARRAY [INTEGER_32]
			-- Cache shapes for broadcasting un-reduction in backward pass

feature -- Operational

	forward (inputs: ARRAY [ET_VALUE]): ET_VALUE
			-- Computes a + b.
		local
			t1, t2, t_res: ET_TENSOR
		do
			t1 := inputs [1].data
			t2 := inputs [2].data
			saved_a_shape := t1.shape.deep_twin
			saved_b_shape := t2.shape.deep_twin
			
			t_res := t1.plus_internal (t2)
			create Result.make_with_parents (t_res, inputs, Current)
		end

	backward (grad_output: ET_TENSOR): ARRAY [ET_TENSOR]
			-- Computes dL/da and dL/db.
		require else
			shapes_saved: saved_a_shape /= Void and saved_b_shape /= Void
		do
			create Result.make_empty
			if attached saved_a_shape as shape_a and attached saved_b_shape as shape_b then
				Result.force (grad_output.sum_to_size (shape_a), 1)
				Result.force (grad_output.sum_to_size (shape_b), 2)
			end
		end

end
