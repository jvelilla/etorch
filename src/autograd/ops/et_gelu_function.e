note
	description: "[
		Autograd function for GELU activation.
		Equivalent to PyTorch's `GeluBackward`.
	]"

class
	ET_GELU_FUNCTION

inherit
	ET_FUNCTION

feature -- State

	saved_x: detachable ET_TENSOR
			-- Cache the input `x`

feature -- Operational

	forward (inputs: ARRAY [ET_VALUE]): ET_VALUE
			-- Computes gelu(x).
		local
			t1, t_res: ET_TENSOR
		do
			t1 := inputs [1].data
			saved_x := t1
			
			t_res := t1.gelu_internal
			create Result.make_with_parents (t_res, inputs, Current)
		end

	backward (grad_output: ET_TENSOR): ARRAY [ET_TENSOR]
			-- Computes GELU derivative.
			-- D GELU(x) = 0.5 * [1 + tanh(k * (x + c * x^3))] 
			--           + 0.5 * x * sech^2(k * (x + c * x^3)) * k * (1 + 3 * c * x^2)
			-- where k = sqrt(2/pi) = 0.79788456, c = 0.044715
			-- Using our tensor ops to compute this element-wise.
		require else
			input_saved: saved_x /= Void
		local
			x, x_sq, x_cb, inner, tanh_inner, sech_sq_inner, left_term, right_term, scale, d_gelu: ET_TENSOR
			k, c: REAL_64
			pi: REAL_64
			l_math: DOUBLE_MATH
		do
			create l_math
			pi := 3.14159265358979323846
			k := l_math.sqrt (2.0 / pi)
			c := 0.044715
			
			create Result.make_empty
			if attached saved_x as local_x then
				x := local_x
				
				-- inner = k * (x + c * x^3)
				x_sq := x.mul (x)
				x_cb := x_sq.mul (x)
				inner := x.plus (x_cb.mul_scalar (c)).mul_scalar (k)
				
				-- tanh(inner)
				tanh_inner := inner.tanh_val
				
				-- left_term = 0.5 * (1 + tanh(inner))
				left_term := tanh_inner.plus_scalar (1.0).mul_scalar (0.5)
				
				-- sech^2 = 1 - tanh^2
				sech_sq_inner := tanh_inner.mul (tanh_inner).mul_scalar (-1.0).plus_scalar (1.0)
				
				-- right_term = 0.5 * x * sech^2(inner) * k * (1 + 3 * c * x^2)
				scale := x_sq.mul_scalar (3.0 * c).plus_scalar (1.0).mul_scalar (k).mul_scalar (0.5)
				right_term := x.mul (sech_sq_inner).mul (scale)
				
				d_gelu := left_term.plus (right_term)
				
				Result.force (grad_output.mul (d_gelu), 1)
			end
		end

end
