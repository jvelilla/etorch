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
			from_existing_data
			test_matmul
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

	test_matmul
		local
			a, b, c: ET_TENSOR
			a_store, b_store: ET_STORAGE_REAL_64
			l_strides: ARRAY [INTEGER_32]
		do
			print ("%N[1.2] Matrix Multiplication (OpenBLAS cblas_dgemm)%N")
			
			create a_store.make (4)
			a_store.put_real_64 (1.0, 1)
			a_store.put_real_64 (2.0, 2)
			a_store.put_real_64 (3.0, 3)
			a_store.put_real_64 (4.0, 4)
			
			create b_store.make (4)
			b_store.put_real_64 (2.0, 1)
			b_store.put_real_64 (0.0, 2)
			b_store.put_real_64 (1.0, 3)
			b_store.put_real_64 (2.0, 4)
			
			l_strides := <<2, 1>>
			
			create a.make_from_storage (a_store, <<2, 2>>, l_strides, 0)
			create b.make_from_storage (b_store, <<2, 2>>, l_strides, 0)
			
			c := a.matmul (b)
			print ("MATMUL Result shape: [" + c.shape [1].out + ", " + c.shape [2].out + "]%N")
			
			print ("C[1,1]: " + c.storage.item_as_real_64 (1).out + " (Expected 4)%N")
			print ("C[1,2]: " + c.storage.item_as_real_64 (2).out + " (Expected 4)%N")
			print ("C[2,1]: " + c.storage.item_as_real_64 (3).out + " (Expected 10)%N")
			print ("C[2,2]: " + c.storage.item_as_real_64 (4).out + " (Expected 8)%N")
		end

end
