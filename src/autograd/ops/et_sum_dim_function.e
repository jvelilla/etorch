note
	description: "[
		Autograd function for sum over a specific dimension.
		The backward pass broadcasts the incoming gradient along the reduced dimension.
	]"

class
	ET_SUM_DIM_FUNCTION

inherit
	ET_FUNCTION
		redefine
			default_create
		end

feature {NONE} -- Initialization

	default_create
		do
			create saved_shape.make_empty
		end

feature -- Forward

	forward_with_params (inputs: ARRAY [ET_VALUE]; a_dim: INTEGER_32; a_keep_dim: BOOLEAN): ET_VALUE
			-- Computes sum_dim and tracks the graph.
		local
			t1, t_res: ET_TENSOR
		do
			t1 := inputs [1].data
			saved_shape := t1.shape.deep_twin
			saved_dim := a_dim
			saved_keep_dim := a_keep_dim
			
			t_res := t1.sum_dim_internal (a_dim, a_keep_dim)
			create Result.make_with_parents (t_res, inputs, Current)
		end

	forward (inputs: ARRAY [ET_VALUE]): ET_VALUE
		do
			(create {EXCEPTIONS}).raise ("Use forward_with_params for ET_SUM_DIM_FUNCTION")
			create Result.make (inputs[1].data)
		end

feature -- Backward

	backward (grad_output: ET_TENSOR): ARRAY [ET_TENSOR]
			-- Backpropagate gradient.
		local
			l_grad: ET_TENSOR
			br_shape: ARRAY [INTEGER_32]
			i: INTEGER_32
			l_grad_unsqueezed: ET_TENSOR
		do
			if not saved_keep_dim then
				-- We must unsqueeze the grad_output to restore the dimension that was summed over
				create br_shape.make_empty
				from i := 1 until i > saved_shape.count loop
					if i = saved_dim then
						br_shape.force (1, br_shape.count + 1)
					else
						if i < saved_dim then
							br_shape.force (grad_output.shape [i], br_shape.count + 1)
						else
							br_shape.force (grad_output.shape [i - 1], br_shape.count + 1)
						end
					end
					i := i + 1
				end
				l_grad_unsqueezed := grad_output.reshape (br_shape)
			else
				l_grad_unsqueezed := grad_output
			end
			
			-- The sum gradient is just broadcasting the unsqueezed gradient to the original shape
			create l_grad.make_zeros_with_dtype (saved_shape, l_grad_unsqueezed.dtype)
			l_grad := l_grad_unsqueezed.plus (l_grad)
			
			create Result.make_filled (l_grad, 1, 1)
		end

feature {NONE} -- State

	saved_shape: ARRAY [INTEGER_32]
	saved_dim: INTEGER_32
	saved_keep_dim: BOOLEAN

end
