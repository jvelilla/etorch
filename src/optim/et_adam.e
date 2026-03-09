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
			zero_arr1, zero_arr2: ARRAY [REAL_64]
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
				create zero_arr1.make_filled (0.0, 1, p.numel)
				create zero_arr2.make_filled (0.0, 1, p.numel)
				m_moments.force (zero_arr1, i)
				v_moments.force (zero_arr2, i)
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
	m_moments: ARRAY [ARRAY [REAL_64]]
	v_moments: ARRAY [ARRAY [REAL_64]]

feature -- Operations

	step
			-- Perform a single optimization step.
		local
			pi: INTEGER_32
			p: ET_TENSOR
			m_arr, v_arr: ARRAY [REAL_64]
			i: INTEGER_32
			g_val, m_val, v_val, m_hat, v_hat, p_val: REAL_64
			l_math: DOUBLE_MATH
		do
			t := t + 1
			create l_math
			
			from pi := 1 until pi > params.count loop
				p := params [pi]
				if attached p.grad as g then
					if attached {ET_STORAGE_REAL_64} p.storage as ps and then
					   attached {ET_STORAGE_REAL_64} g.storage as gs then
						
						m_arr := m_moments [pi]
						v_arr := v_moments [pi]
						
						from i := 1 until i > ps.count loop
							g_val := gs.item_as_real_64 (i)
							
							-- m = beta1 * m + (1 - beta1) * grad
							m_val := beta1 * m_arr [i] + (1.0 - beta1) * g_val
							m_arr [i] := m_val
							
							-- v = beta2 * v + (1 - beta2) * (grad * grad)
							v_val := beta2 * v_arr [i] + (1.0 - beta2) * (g_val * g_val)
							v_arr [i] := v_val
							
							-- bias correction
							m_hat := m_val / (1.0 - l_math.exp (t.to_double * l_math.log (beta1)))
							v_hat := v_val / (1.0 - l_math.exp (t.to_double * l_math.log (beta2)))
							
							-- update parameter
							p_val := ps.item_as_real_64 (i) - lr * m_hat / (l_math.sqrt (v_hat) + eps)
							ps.put_real_64 (p_val, i)
							
							i := i + 1
						end
					end
				end
				pi := pi + 1
			end
		end

	zero_grad
			-- Zero gradients of all parameters.
		local
			p: ET_TENSOR
			i: INTEGER_32
		do
			across params as param loop
				p := param
				if attached p.grad as g then
					if attached {ET_STORAGE_REAL_64} g.storage as gs then
						from i := 1 until i > gs.count loop
							gs.put_real_64 (0.0, i)
							i := i + 1
						end
					end
				end
			end
		end

	set_lr (new_lr: REAL_64)
			-- Update learning rate
		do
			lr := new_lr
		end

end
