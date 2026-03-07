note
    description: "[
        Main entry point and shared configuration for the ET_TORCH tensor framework.
        Provides a process-wide `grad_enabled` flag (matching PyTorch's torch.no_grad())
		and factory functions mirroring the torch.* namespace.

        Usage:
            {ET_TORCH}.no_grad
            tensor := {ET_TORCH}.tensor (...)
    ]"

class
    ET_TORCH

feature -- Factories (PyTorch style)

	tensor (a_shape: ARRAY [INTEGER_32]): ET_TENSOR
			-- Creates a zero-initialized tensor of float64 (shortcut for now).
			-- In a full implementation, this takes actual data arrays.
		do
			create Result.make_zeros (a_shape)
		end

	zeros (a_shape: ARRAY [INTEGER_32]): ET_TENSOR
		do
			create Result.make_zeros (a_shape)
		end

	ones (a_shape: ARRAY [INTEGER_32]): ET_TENSOR
		do
			create Result.make_ones (a_shape)
		end

feature -- Gradient Mode (instance-free)

    grad_enabled: CELL [BOOLEAN]
        do  
            create Result.put (True)
        ensure
            class
        end

    no_grad
            -- Disable gradient computation (for inference).
            -- Equivalent to PyTorch's `torch.no_grad()`.
        do
            grad_enabled.put (False)
        ensure
            class
        end

    enable_grad
            -- Re-enable gradient computation (for training).
        do
            grad_enabled.put (True)
        ensure
            class
        end

    is_grad_enabled: BOOLEAN
            -- Is gradient computation currently enabled?
        do
            Result := grad_enabled.item
        ensure
            class
        end

end
