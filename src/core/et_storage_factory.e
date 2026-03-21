note
	description: "Factory for instantiating storages based on ET_DTYPE."

class
	ET_STORAGE_FACTORY

feature -- Creation

	make_zeros (a_size: INTEGER_32; a_dtype: ET_DTYPE): ET_STORAGE
			-- Create a new ZERO-initialized storage of `a_size` elements with `a_dtype`.
		require
			valid_size: a_size >= 0
		local
			l_float32: ET_STORAGE_REAL_32
			l_float64: ET_STORAGE_REAL_64
			l_int32: ET_STORAGE_INT_32
			l_int64: ET_STORAGE_INT_64
			l_bool: ET_STORAGE_BOOL
		do
			if a_dtype.is_float32 then
				create l_float32.make (a_size)
				Result := l_float32
			elseif a_dtype.is_float64 then
				create l_float64.make (a_size)
				Result := l_float64
			elseif a_dtype.is_int32 then
				create l_int32.make (a_size)
				Result := l_int32
			elseif a_dtype.is_int64 then
				create l_int64.make (a_size)
				Result := l_int64
			elseif a_dtype.is_bool then
				create l_bool.make (a_size)
				Result := l_bool
			else
				check unsupported_dtype: False end
				create l_float32.make (a_size)
				Result := l_float32
			end
		ensure
			result_size: Result.count = a_size
		end

	make_ones (a_size: INTEGER_32; a_dtype: ET_DTYPE): ET_STORAGE
			-- Create a new ONE-initialized storage of `a_size` elements with `a_dtype`.
		require
			valid_size: a_size >= 0
		local
			l_float32: ET_STORAGE_REAL_32
			l_float64: ET_STORAGE_REAL_64
			l_int32: ET_STORAGE_INT_32
			l_int64: ET_STORAGE_INT_64
			l_bool: ET_STORAGE_BOOL
			i: INTEGER_32
		do
			if a_dtype.is_float32 then
				create l_float32.make (a_size)
				from i := 1 until i > a_size loop l_float32.put_real_32 (1.0, i); i := i + 1 end
				Result := l_float32
			elseif a_dtype.is_float64 then
				create l_float64.make (a_size)
				from i := 1 until i > a_size loop l_float64.put_real_64 (1.0, i); i := i + 1 end
				Result := l_float64
			elseif a_dtype.is_int32 then
				create l_int32.make (a_size)
				from i := 1 until i > a_size loop l_int32.put_int_32 (1, i); i := i + 1 end
				Result := l_int32
			elseif a_dtype.is_int64 then
				create l_int64.make (a_size)
				from i := 1 until i > a_size loop l_int64.put_int_64 ({INTEGER_64} 1, i); i := i + 1 end
				Result := l_int64
			elseif a_dtype.is_bool then
				create l_bool.make (a_size)
				from i := 1 until i > a_size loop l_bool.put_boolean (True, i); i := i + 1 end
				Result := l_bool
			else
				check unsupported_dtype: False end
				create l_float32.make (a_size)
				Result := l_float32
			end
		ensure
			result_size: Result.count = a_size
		end

	make_randn (a_size: INTEGER_32; a_dtype: ET_DTYPE): ET_STORAGE
			-- Create a new normal-distributed random storage of `a_size` elements with `a_dtype`.
		require
			valid_size: a_size >= 0
			valid_floating_type: a_dtype.is_floating
		local
			l_float32: ET_STORAGE_REAL_32
			l_float64: ET_STORAGE_REAL_64
			i: INTEGER_32
			l_rand: RANDOM
			l_time: TIME
			u1, u2, z0, z1: REAL_64
			l_math: DOUBLE_MATH
			pi_2: REAL_64
		do
			create l_time.make_now
			create l_rand.set_seed (l_time.milli_second)
			create l_math
			pi_2 := 2.0 * 3.14159265358979323846

			if a_dtype.is_float32 then
				create l_float32.make (a_size)
				from i := 1 until i > a_size loop
					l_rand.forth; u1 := l_rand.double_item
					if u1 = 0.0 then u1 := 0.000001 end
					l_rand.forth; u2 := l_rand.double_item
					z0 := l_math.sqrt (-2.0 * l_math.log (u1)) * l_math.cosine (pi_2 * u2)
					l_float32.put_real_32 (z0.truncated_to_real, i)
					i := i + 1
					if i <= a_size then
						z1 := l_math.sqrt (-2.0 * l_math.log (u1)) * l_math.sine (pi_2 * u2)
						l_float32.put_real_32 (z1.truncated_to_real, i)
						i := i + 1
					end
				end
				Result := l_float32
			elseif a_dtype.is_float64 then
				create l_float64.make (a_size)
				from i := 1 until i > a_size loop
					l_rand.forth; u1 := l_rand.double_item
					if u1 = 0.0 then u1 := 0.000001 end
					l_rand.forth; u2 := l_rand.double_item
					z0 := l_math.sqrt (-2.0 * l_math.log (u1)) * l_math.cosine (pi_2 * u2)
					l_float64.put_real_64 (z0, i)
					i := i + 1
					if i <= a_size then
						z1 := l_math.sqrt (-2.0 * l_math.log (u1)) * l_math.sine (pi_2 * u2)
						l_float64.put_real_64 (z1, i)
						i := i + 1
					end
				end
				Result := l_float64
			else
				check unsupported_floating_type: False end
				create l_float32.make (a_size)
				Result := l_float32
			end
		ensure
			result_size: Result.count = a_size
		end

end
