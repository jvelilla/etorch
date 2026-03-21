note
	description: "Handles type promotion rules for mixed-dtype mathematical operations."

class
	ET_DTYPE_PROMOTER

feature -- Type Promotion

	promoted_dtype (a, b: ET_DTYPE): ET_DTYPE
			-- Returns the resulting dtype when combining `a` and `b`.
			-- Mirrors PyTorch's type promotion rules:
			-- 1. Floating point overrules integer/boolean.
			-- 2. Wider types overrule narrower types.
		require
			valid_a: a /= Void
			valid_b: b /= Void
		do
			if a.is_floating and b.is_floating then
				if a.byte_size >= b.byte_size then
					Result := a
				else
					Result := b
				end
			elseif a.is_floating and not b.is_floating then
				Result := a
			elseif not a.is_floating and b.is_floating then
				Result := b
			else
				-- Both are integers or booleans
				if a.byte_size >= b.byte_size then
					Result := a
				else
					Result := b
				end
			end
		ensure
			result_valid: Result /= Void
		end

end
