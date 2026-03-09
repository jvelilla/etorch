note
	description: "[
		Utilities to serialize and deserialize a module's state_dict.
		Analogous to torch.save and torch.load.
	]"

class
	ET_SAVE_LOAD

feature -- Serialization

	save (state: HASH_TABLE [ET_TENSOR, STRING]; file_path: STRING)
			-- Save the state_dict to the specified file path.
		require
			valid_path: not file_path.is_empty
		local
			file: RAW_FILE
		do
			create file.make_with_name (file_path)
			file.open_write
			file.independent_store (state)
			file.close
		end

	load (file_path: STRING): detachable HASH_TABLE [ET_TENSOR, STRING]
			-- Load a state_dict from the specified file path.
		require
			valid_path: not file_path.is_empty
		local
			file: RAW_FILE
		do
			create file.make_with_name (file_path)
			if file.exists and file.is_readable then
				file.open_read
				if attached {HASH_TABLE [ET_TENSOR, STRING]} file.retrieved as loaded_state then
					Result := loaded_state
				end
				file.close
			end
		end

end
