note
	description: "[
		Adam Optimizer.
		Implements the Adam algorithm for eTorch.
	]"

class
	ET_ADAM

inherit
	ET_OPTIMIZER

create
	make

feature {NONE} -- Initialization

	make (a_params: LIST [ET_TENSOR]; a_lr: REAL_64)
			-- Initialize Adam with default beta and eps parameters.
		require
			has_params: not a_params.is_empty
			valid_lr: a_lr > 0.0
		local
			p: ET_TENSOR
			i: INTEGER_32
			m_t, v_t: ET_TENSOR
		do
			params := a_params
			lr := a_lr
			beta1 := 0.9
			beta2 := 0.999
			eps := 1.0e-8
			t := 0
			
			-- Initialize moments based on parameters sizes
			create m_moments.make_empty
			create v_moments.make_empty
			
			from i := 1 until i > params.count loop
				p := params [i]
				create m_t.make_zeros (p.shape)
				create v_t.make_zeros (p.shape)
				m_moments.force (m_t, i)
				v_moments.force (v_t, i)
				i := i + 1
			end
		ensure
			params_set: params = a_params
			lr_set: lr = a_lr
			moments_initialized: m_moments.count = params.count
		end

feature -- Access

	params: LIST [ET_TENSOR]
	lr, beta1, beta2, eps: REAL_64
	t: INTEGER_32
	m_moments: ARRAY [ET_TENSOR]
	v_moments: ARRAY [ET_TENSOR]

feature -- Operations

	step
			-- Perform a single optimization step using in-place tensor operations.
		local
			pi: INTEGER_32
			p, g, m, v: ET_TENSOR
			m_grad, g_sq, v_grad, m_hat, v_hat, v_hat_sqrt, v_denom, update: ET_TENSOR
			l_math: DOUBLE_MATH
			m_hat_scale, v_hat_scale: REAL_64
		do
			t := t + 1
			create l_math
			
			from pi := 1 until pi > params.count loop
				p := params [pi]
				if attached p.grad as ag then
					g := ag
					m := m_moments [pi]
					v := v_moments [pi]
					
					-- m = beta1 * m + (1 - beta1) * grad
					m.mul_scalar_in_place (beta1)
					m_grad := g.mul_scalar (1.0 - beta1)
					m.plus_in_place (m_grad)
					
					-- v = beta2 * v + (1 - beta2) * (grad * grad)
					v.mul_scalar_in_place (beta2)
					g_sq := g.mul (g)
					v_grad := g_sq.mul_scalar (1.0 - beta2)
					v.plus_in_place (v_grad)
					
					-- bias correction
					m_hat_scale := 1.0 / (1.0 - l_math.exp (t.to_double * l_math.log (beta1)))
					v_hat_scale := 1.0 / (1.0 - l_math.exp (t.to_double * l_math.log (beta2)))
					
					m_hat := m.mul_scalar (m_hat_scale)
					v_hat := v.mul_scalar (v_hat_scale)
					
					-- update parameter
					v_hat_sqrt := v_hat.sqrt_val
					v_denom := v_hat_sqrt.plus_scalar (eps)
					update := m_hat.div (v_denom)
					update.mul_scalar_in_place (-lr)
					
					p.set_requires_grad (False)
					p.plus_in_place (update)
					p.set_requires_grad (True)
				end
				pi := pi + 1
			end
		end

	zero_grad
			-- Zero gradients of all parameters (set to Void to free memory).
		local
			p: ET_TENSOR
		do
			across params as param loop
				p := param
				p.set_grad (Void)
				if attached p.grad_node as n then
					n.set_grad_void
				end
			end
		end

	set_lr (new_lr: REAL_64)
			-- Update learning rate
		do
			lr := new_lr
		end

end
