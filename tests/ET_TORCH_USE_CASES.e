note
    description: "PyTorch Use Cases implemented in Eiffel for eTorch."

class
    ET_TORCH_USE_CASES

create
	make

feature -- Initialization

	make
		do
			print ("eTorch Compilation Successful! Tests will be run here.%N")
		end

feature -- Tests

    from_existing_data
            -- 1.1 From Existing Data Structures
        local
            x: ET_TENSOR
            l_data: ARRAY [INTEGER_32]
            l_shape: ARRAY [INTEGER_32]
        do
            print ("%N[1.1] From Existing Data Structures ( List -> Tensor)%N")
            
            -- Simplified for compile check - normally we parse the array
            l_data := <<1, 2, 3>>
            l_shape := <<3>>
            create x.make_zeros (l_shape)

            print ("TENSOR NUMEL: " + x.numel.out + "%N")
        end

end
