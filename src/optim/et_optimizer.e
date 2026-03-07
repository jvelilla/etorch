note
	description: "[
		Base class for all Optimizers in eTorch.
		Equivalent to torch.optim.Optimizer.
	]"

deferred class
	ET_OPTIMIZER

feature -- Operations

	step
			-- Perform a single optimization step (parameter update).
		deferred
		end

	zero_grad
			-- Clear the gradients of all optimized parameters.
		deferred
		end

	set_lr (new_lr: REAL_64)
			-- Update the learning rate.
		deferred
		end

end
