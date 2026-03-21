note
	description: "[
		Autograd function for matrix multiplication.
		Equivalent to PyTorch's `MmBackward` or `BmmBackward`.
	]"

class
	ET_MATMUL_FUNCTION

inherit
	ET_FUNCTION

feature -- State

	saved_a, saved_b: detachable ET_TENSOR
			-- Cache inputs for backward pass

feature -- Operational

	forward (inputs: ARRAY [ET_VALUE]): ET_VALUE
			-- Computes a \@ b.
		local
			t1, t2, t_res: ET_TENSOR
		do
			t1 := inputs [1].data
			t2 := inputs [2].data
			
			saved_a := t1
			saved_b := t2
			
			t_res := t1.matmul_internal (t2)
			create Result.make_with_parents (t_res, inputs, Current)
		end

	backward (grad_output: ET_TENSOR): ARRAY [ET_TENSOR]
			-- Computes dL/da and dL/db cleanly, handling 1D vectors and batched ND tensors.
		require else
			inputs_saved: saved_a /= Void and saved_b /= Void
		local
			g_a, g_b: ET_TENSOR
			a_2d_shape, b_2d_shape, g_out_shape: ARRAY [INTEGER_32]
			i: INTEGER_32
			a_rank, b_rank: INTEGER_32
		do
			create Result.make_empty
			if attached saved_a as a and attached saved_b as b then
				a_rank := a.rank
				b_rank := b.rank
				
				if a_rank = 1 and b_rank = 1 then
					-- Dot product: grad_output is scalar
					g_a := grad_output.mul (b)
					g_b := grad_output.mul (a)
				elseif a_rank = 1 and b_rank >= 2 then
					-- Vector x Matrix
					g_a := grad_output.matmul (b.transpose (b_rank - 1, b_rank))
					
					create a_2d_shape.make_empty
					a_2d_shape.force (a.shape [1], 1)
					a_2d_shape.force (1, 2)
					
					create g_out_shape.make_empty
					from i := 1 until i = grad_output.rank loop
						g_out_shape.force (grad_output.shape[i], g_out_shape.count + 1)
						i := i + 1
					end
					g_out_shape.force (1, g_out_shape.count + 1)
					g_out_shape.force (grad_output.shape[grad_output.rank], g_out_shape.count + 1)
					
					g_b := a.reshape(a_2d_shape).matmul (grad_output.reshape(g_out_shape))
				elseif a_rank >= 2 and b_rank = 1 then
					-- Matrix x Vector
					g_b := a.transpose (a_rank - 1, a_rank).matmul (grad_output)
					
					create b_2d_shape.make_empty
					b_2d_shape.force (1, 1)
					b_2d_shape.force (b.shape [1], 2)
					
					create g_out_shape.make_empty
					from i := 1 until i > grad_output.rank loop
						g_out_shape.force (grad_output.shape[i], g_out_shape.count + 1)
						i := i + 1
					end
					g_out_shape.force (1, g_out_shape.count + 1)
					
					g_a := grad_output.reshape(g_out_shape).matmul (b.reshape(b_2d_shape))
				else
					-- Matrix x Matrix (Standard)
					g_a := grad_output.matmul (b.transpose (b_rank - 1, b_rank))
					g_b := a.transpose (a_rank - 1, a_rank).matmul (grad_output)
				end
				
				Result.force (g_a.sum_to_size (a.shape), 1)
				Result.force (g_b.sum_to_size (b.shape), 2)
			end
		end

end
