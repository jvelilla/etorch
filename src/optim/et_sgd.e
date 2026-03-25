note
	description: "[
		Stochastic Gradient Descent Optimizer.
		Provides the simplest optimization step using param = param - lr * grad.
	]"

class
	ET_SGD

inherit
	ET_OPTIMIZER

create
	make

feature {NONE} -- Initialization

	make (a_params: LIST [ET_TENSOR]; a_lr: REAL_64)
			-- Initialize SGD with parameters and learning rate.
		require
			has_params: not a_params.is_empty
			valid_lr: a_lr > 0.0
		do
			params := a_params
			lr := a_lr
		ensure
			params_set: params = a_params
			lr_set: lr = a_lr
		end

feature -- Access

	params: LIST [ET_TENSOR]
	lr: REAL_64

feature -- Operations

	step
			-- Perform a single optimization step using in-place tensor operations.
		local
			pi: INTEGER_32
			p, g, update: ET_TENSOR
		do
			from pi := 1 until pi > params.count loop
				p := params [pi]
				if attached p.grad as ag then
					g := ag
					update := g.mul_scalar (-lr)
					
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
