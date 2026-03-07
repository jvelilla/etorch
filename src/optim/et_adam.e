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
			valid_params: a_params /= Void and then not a_params.is_empty
			valid_lr: a_lr > 0.0
		do
			params := a_params
			lr := a_lr
			beta1 := 0.9
			beta2 := 0.999
			eps := 1.0e-8
			t := 0
			
			-- Initialize moments based on parameters sizes
			-- In a real implementation this creates parallel list of zero-tensors
		ensure
			params_set: params = a_params
			lr_set: lr = a_lr
		end

feature -- Access

	params: LIST [ET_TENSOR]
	lr, beta1, beta2, eps: REAL_64
	t: INTEGER_32

feature -- Operations

	step
			-- Perform a single optimization step.
		do
			t := t + 1
			-- Adam implementation goes here utilizing the Tensor overloaded math operators.
			-- (Extracted shell to verify architecture compilation first)
		end

	zero_grad
			-- Zero gradients of all parameters.
		do
			across params as p loop
				-- In PyTorch setting grad to None is preferred over zeros for memory, but zero works too
				if attached p.grad as g then
					-- g.zero_() equivalent 
				end
			end
		end

	set_lr (new_lr: REAL_64)
			-- Update learning rate
		do
			lr := new_lr
		end

end
