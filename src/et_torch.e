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
		ensure
			class
		end

	zeros (a_shape: ARRAY [INTEGER_32]): ET_TENSOR
		do
			create Result.make_zeros (a_shape)
		ensure
			class
		end

	ones (a_shape: ARRAY [INTEGER_32]): ET_TENSOR
		do
			create Result.make_ones (a_shape)
		ensure
			class
		end

feature -- Gradient Mode (instance-free)

    grad_enabled: CELL [BOOLEAN]
        once
            create Result.put (True)
        ensure
            instance_free: class
        end

    no_grad
            -- Disable gradient computation (for inference).
            -- Equivalent to PyTorch's `torch.no_grad()`.
        do
            grad_enabled.put (False)
        ensure
            instance_free: class
        end

    enable_grad
            -- Re-enable gradient computation (for training).
        do
            grad_enabled.put (True)
        ensure
            instance_free: class
        end

    is_grad_enabled: BOOLEAN
            -- Is gradient computation currently enabled?
        do
            Result := grad_enabled.item
        ensure
            instance_free: class
        end

    with_no_grad (action: PROCEDURE)
            -- Execute a block without tracking gradients. Ensures grad mode is restored.
        local
            was_enabled: BOOLEAN
        do
            was_enabled := is_grad_enabled
            no_grad
            action.call (Void)
            if was_enabled then
                enable_grad
            end
        ensure
            instance_free: class
        rescue
            if was_enabled then
                enable_grad
            end
        end

feature -- Serialization

    save (state: HASH_TABLE [ET_TENSOR, STRING]; file_path: STRING)
            -- Save the state_dict to the specified file path. (Like `torch.save`)
        local
            sl: ET_SAVE_LOAD
        do
            create sl
            sl.save (state, file_path)
        ensure
            class
        end

    load (file_path: STRING): detachable HASH_TABLE [ET_TENSOR, STRING]
            -- Load a state_dict from the specified file path. (Like `torch.load`)
        local
            sl: ET_SAVE_LOAD
        do
            create sl
            Result := sl.load (file_path)
        ensure
            class
        end

end
