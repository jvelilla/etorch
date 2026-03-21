note
	description: "Tests for Utilities like Save/Load."

class
	ET_TEST_UTILS

inherit
	EQA_TEST_SET

feature -- Tests

	test_model_serialization
		local
			p: ET_TENSOR
			state: HASH_TABLE [ET_TENSOR, STRING]
			sl: ET_SAVE_LOAD
			loaded_state: detachable HASH_TABLE [ET_TENSOR, STRING]
			tol: REAL_64
		do
			tol := 1.0e-5
			
			create p.make_zeros_with_dtype (<<2>>, create {ET_DTYPE_FLOAT64})
			if attached {ET_STORAGE_REAL_64} p.storage as ps then
				ps.put_real_64 (5.5, 1)
				ps.put_real_64 (7.7, 2)
			end
			
			create state.make (1)
			state.put (p, "layer.weight")
			
			create sl
			sl.save (state, "test_model_autotest.pt")
			
			loaded_state := sl.load ("test_model_autotest.pt")
			assert ("State loaded from disk", loaded_state /= Void)
			
			if attached loaded_state as ls then
				assert ("Key exists in loaded dict", ls.has ("layer.weight"))
				if attached ls.item ("layer.weight") as lp then
					if attached {ET_STORAGE_REAL_64} lp.storage as lps then
						assert_approx_32 (lps.item_as_real_64 (1), 5.5, tol, "Restored P[1]")
						assert_approx_32 (lps.item_as_real_64 (2), 7.7, tol, "Restored P[2]")
					else
						assert ("Wrong storage restored", False)
					end
				end
			end
		end

feature {NONE} -- Helpers

	assert_approx_32 (actual, expected, tol: REAL_64; msg: STRING)
		do
			assert (msg + " Expected " + expected.out + " but got " + actual.out, (actual - expected).abs <= tol)
		end

end
