note
    description: "Loader for Safetensors format (Zero-Copy using MANAGED_POINTER)."

class
    ET_SAFE_TENSORS_LOADER

create
    make

feature -- Initialization

    make
        do
        end

feature -- Access

    load_tensor (a_path: PATH; key: READABLE_STRING_GENERAL): detachable ET_TENSOR
        local
            file: RAW_FILE
            header_size: INTEGER_64
            header_json: STRING_8
            data_offset: INTEGER
            tensor_data_start: INTEGER
            
            parser: JSON_PARSER
            mmap: ET_MEMORY_MAP
        do
            create file.make_with_path (a_path)
            if file.exists then
                file.open_read

                if file.count > 8 then
                    file.read_integer_64
                    header_size := file.last_integer_64
                    
                    if header_size > 0 then
                        -- Read header
                        file.read_stream (header_size.to_integer_32)
                        header_json := file.last_string
                        
                        -- Parse JSON header
                        create parser.make_with_string (header_json)
                        parser.parse_content
                        
                        if parser.is_valid and then attached {JSON_OBJECT} parser.parsed_json_value as j_obj then
                             if attached {JSON_OBJECT} j_obj.item (key) as info then
                                 -- Calculate absolute offset
                                 data_offset := 8 + header_size.to_integer_32
                                 
                                 if attached {JSON_ARRAY} info.item ("data_offsets") as j_offsets and then j_offsets.count >= 2 and then
                                    attached {JSON_NUMBER} j_offsets.i_th (1) as start_off and then
                                    attached {JSON_NUMBER} j_offsets.i_th (2) as end_off 
                                 then
                                     tensor_data_start := data_offset + start_off.integer_64_item.to_integer_32
                                     
                                     if attached {JSON_ARRAY} info.item ("shape") as j_shape then
                                         -- Create memory map of the file
                                         create mmap.make_with_path (a_path)
                                         
                                         if mmap.is_mapped then
                                         	-- Create Tensor directly from memory mapped pointer
                                         	Result := create_tensor_from_mmap (mmap, tensor_data_start, json_array_to_integer_array (j_shape), (end_off.integer_64_item - start_off.integer_64_item).to_integer_32)
                                         	mmap.unmap -- Data is copied to F64 storage, safe to unmap
                                         else
                                         	print ("Could not memory map file%N")
                                         end
                                     end
                                 end
                             end
                        end
                    end
                end
                
                file.close
            end
        end

feature {NONE} -- Implementation

    create_tensor_from_mmap (mmap: ET_MEMORY_MAP; offset_in_file: INTEGER; shape: ARRAY [INTEGER]; byte_len: INTEGER): ET_TENSOR
        local
            l_store: ET_STORAGE_REAL_64
            l_strides: ARRAY [INTEGER_32]
            i, l_count: INTEGER_32
            mp: MANAGED_POINTER
        do
            -- Wrap mmapped pointer in MANAGED_POINTER
            create mp.share_from_pointer (mmap.item + offset_in_file, byte_len)
            
            -- Convert F32 bytes to Real64 storage
            -- Note: Production impl would use zero-copy directly if natively float64
            l_count := byte_len // 4
            create l_store.make (l_count)
            from i := 1 until i > l_count loop
                l_store.put_real_64 (mp.read_real_32_le ((i - 1) * 4).to_double, i)
                i := i + 1
            end
            
            l_strides := default_strides (shape)
            create Result.make_from_storage (l_store, shape, l_strides, 0)
        end

    default_strides (shape: ARRAY [INTEGER]): ARRAY [INTEGER]
        local
            i, acc: INTEGER
        do
            create Result.make_empty
            if not shape.is_empty then
                create Result.make_filled (1, 1, shape.count)
                acc := 1
                from i := shape.count until i < 1 loop
                    Result [i] := acc
                    acc := acc * shape [i]
                    i := i - 1
                end
            end
        end

    json_array_to_integer_array (j_array: JSON_ARRAY): ARRAY [INTEGER]
        local
            i: INTEGER
        do
            create Result.make_filled (0, 1, j_array.count)
            from i := 1 until i > j_array.count loop
                if attached {JSON_NUMBER} j_array.i_th (i) as j_num then
                    Result [i] := j_num.integer_64_item.to_integer_32
                end
                i := i + 1
            end
        end

end
