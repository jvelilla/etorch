note
	description: "[
		Autograd function for global mean operation (mean_all).
		The backward pass of a global mean distributes the incoming scalar gradient
		equally to all elements: grad_in = grad_out / N.
	]"

class
	ET_MEAN_ALL_FUNCTION

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

	forward (inputs: ARRAY [ET_VALUE]): ET_VALUE
			-- Computes mean_all and tracks the graph.
		local
			t1, t_res: ET_TENSOR
		do
			t1 := inputs [1].data
			saved_shape := t1.shape.deep_twin
			
			t_res := t1.mean_all_internal
			create Result.make_with_parents (t_res, inputs, Current)
		end

feature -- Backward

	backward (grad_output: ET_TENSOR): ARRAY [ET_TENSOR]
			-- Backpropagate gradient.
		local
			l_grad: ET_TENSOR
			N: INTEGER_32
			g_val: REAL_64
		do
			-- grad_output is a scalar tensor.
			-- We create a gradient tensor of saved_shape filled with (grad_output / N)
			N := 1
			across saved_shape as dim loop N := N * dim.item end
			
			if N = 0 then N := 1 end
			
			g_val := grad_output.item_as_real_64 (<<1>>) / N.to_double
			
			create l_grad.make_ones (saved_shape)
			l_grad := l_grad.mul_scalar (g_val)
			l_grad.set_dtype (grad_output.dtype)
			
			create Result.make_filled (l_grad, 1, 1)
		end

feature {NONE} -- State

	saved_shape: ARRAY [INTEGER_32]
			-- Shape of the original tensor to reconstruct gradient.

end
