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
			test_adam
			test_save_load
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

	test_adam
		local
			p: ET_TENSOR
			g: ET_TENSOR
			p_store, g_store: ET_STORAGE_REAL_64
			optim: ET_ADAM
			params: ARRAYED_LIST [ET_TENSOR]
		do
			print ("%N[1.3] Adam Optimization Step%N")
			
			create p.make_zeros (<<2>>)
			if attached {ET_STORAGE_REAL_64} p.storage as ps then
				ps.put_real_64 (1.0, 1) -- value: 1.0
				ps.put_real_64 (2.0, 2) -- value: 2.0
			end
			p.set_requires_grad (True)
			
			create g.make_zeros (<<2>>)
			if attached {ET_STORAGE_REAL_64} g.storage as gs then
				gs.put_real_64 (0.1, 1) -- grad: 0.1
				gs.put_real_64 (0.2, 2) -- grad: 0.2
			end
			p.set_grad (g)
			
			create params.make (1)
			params.extend (p)
			
			create optim.make (params, 0.1) -- lr = 0.1
			optim.step
			
			print ("After Adam Step (with LR=0.1):%N")
			if attached {ET_STORAGE_REAL_64} p.storage as ps then
				print ("P[1]: " + ps.item_as_real_64 (1).out + "%N")
				print ("P[2]: " + ps.item_as_real_64 (2).out + "%N")
			end
		end

	test_save_load
		local
			p: ET_TENSOR
			state: HASH_TABLE [ET_TENSOR, STRING]
			sl: ET_SAVE_LOAD
			loaded_state: detachable HASH_TABLE [ET_TENSOR, STRING]
			loaded_p: detachable ET_TENSOR
		do
			print ("%N[1.4] Model Serialization%N")
			
			create p.make_zeros (<<2>>)
			if attached {ET_STORAGE_REAL_64} p.storage as ps then
				ps.put_real_64 (5.5, 1)
				ps.put_real_64 (7.7, 2)
			end
			
			create state.make (1)
			state.put (p, "layer.weight")
			
			create sl
			sl.save (state, "test_model.pt")
			
			loaded_state := sl.load ("test_model.pt")
			if attached loaded_state as ls and then attached ls.item ("layer.weight") as lp then
				print ("Loaded state value P[1]: ")
				if attached {ET_STORAGE_REAL_64} lp.storage as lps then
					print (lps.item_as_real_64 (1).out + " (Expected 5.5)%N")
				end
			else
				print ("Failed to load state!%N")
			end
		end

end
