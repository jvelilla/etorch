# Generic Eiffel Inline C Integration Guide

**Version:** 2.0  
**Status:** Generic Reference  
**Scope:** Any Eiffel Project using Inline C

---

## 1. Introduction to Inline C in Eiffel

Eiffel's **Inline C** feature allows you to embed C code directly within your Eiffel classes. This serves as a powerful bridge between the high-level safety of Eiffel and the low-level performance or library access of C.

### Why use Inline C?
*   **Performance:** Execute computationally intensive algorithms (like crypto or image processing) in optimized C.
*   **Library Access:** Call specific C library functions without writing a full wrapper C file.
*   **Platform Specifics:** Access OS-level APIs not exposed by standard Eiffel libraries.
*   **Simplicity:** Keep the C implementation close to the Eiffel interface in a single file, avoiding complex build setups.

---

## 2. Core Syntax

The basic syntax for an inline C feature involves the `external` clause specifying `"C inline"` and an `alias` clause containing the actual C code strings.

```eiffel
feature_name (arguments): RESULT_TYPE
    external
        "C inline"
    alias
        "[
            /* Your C code here */
            /* Access arguments with $arg_name */
        ]"
    end
```

### Accessing Eiffel Arguments
To use an Eiffel argument inside the C code, prefix the argument name with `$`.

**Example: Simple Addition**
```eiffel
c_add (a, b: INTEGER): INTEGER
    external
        "C inline"
    alias
        "[
            return $a + $b;
        ]"
    end
```

---

## 3. Eiffel to C Type Mapping

Understanding how Eiffel types map to C types is crucial for data integrity.

| Eiffel Type | C Type (Macro/Typedef) | Standard C Equivalent | Notes |
| :--- | :--- | :--- | :--- |
| `INTEGER_32` | `EIF_INTEGER` | `long` or `int` | Usually 32-bit signed integer. |
| `INTEGER_64` | `EIF_INTEGER_64` | `long long` | 64-bit signed integer. |
| `NATURAL_32` | `EIF_NATURAL_32` | `unsigned int` | 32-bit unsigned integer. |
| `BOOLEAN` | `EIF_BOOLEAN` | `unsigned char` | `EIF_TRUE` (1) or `EIF_FALSE` (0). |
| `CHARACTER_8` | `EIF_CHARACTER` | `unsigned char` | Single byte character. |
| `POINTER` | `EIF_POINTER` | `void*` | Generic pointer. |
| `REAL_64` | `EIF_REAL_64` | `double` | Double precision float. |

> **Note:** Eiffel objects (classes) are passed as `EIF_REFERENCE` or `EIF_OBJECT`, which are pointers to the internal Eiffel object structure. Interacting with these directly in C requires using the **Eiffel Software Runtime (CECIL)** API, which is advanced. For most inline C, prefer passing basic types and pointers.

---

## 4. Working with Pointers and Memory

Passing arrays or memory buffers relies on `POINTER` and the `MANAGED_POINTER` class. `MANAGED_POINTER` provides a safe way to allocate memory that the Garage Collector (GC) won't move or collect unexpectedly while C takes a pointer to it (if you pin it or use it correctly in the scope).

### 4.1. Passing a Buffer (Array) to C

To pass a block of memory (like an array of bytes), access the `item` feature of a `MANAGED_POINTER` or `C_STRING`.

**Example: Modifying a byte array in C**

**Eiffel Side:**
```eiffel
process_buffer
    local
        managed: MANAGED_POINTER
    do
        create managed.make (1024) -- Allocate 1024 bytes
        c_fill_buffer (managed.item, managed.count)
    end
```

**C Side:**
```eiffel
c_fill_buffer (ptr: POINTER; length: INTEGER)
    external
        "C inline"
    alias
        "[
            unsigned char *params = (unsigned char *)$ptr;
            int i;
            for (i = 0; i < $length; i++) {
                params[i] = 0xFF; // Fill with 0xFF
            }
        ]"
    end
```

### 4.2. Using `MANAGED_POINTER` vs `ARRAY`
*   **`ARRAY [T]`**: An Eiffel object. Its internal data location *can move* during GC cycles. Do *not* pass the address of an Eiffel `ARRAY` element directly to C unless you pin the object (advanced).
*   **`MANAGED_POINTER`**: Allocated in unmanaged memory (C heap). Its address (`item`) is stable. This is the **recommended** way to share buffers.

---

## 5. Working with Strings

Eiffel strings (`STRING_8`, `STRING_32`) are objects. To pass them to C as standard C strings (`char*`), use the `C_STRING` helper class.

**Example: Printing a string from C**

```eiffel
print_from_c (a_string: STRING)
    local
        c_str: C_STRING
    do
        create c_str.make (a_string)
        c_print_message (c_str.item)
    end

c_print_message (msg_ptr: POINTER)
    external
        "C inline use <stdio.h>"
    alias
        "[
            printf(\"Message from C: %s\\n\", (char*)$msg_ptr);
        ]"
    end
```
*Note the `use <stdio.h>` clause allows including standard headers.*

---

## 6. Advanced Usage & Structs

You can include custom C structs or headers using the `use` syntax.

### 6.1. Defining Structs Inline
You can define structs directly inside the alias block, though it's cleaner to put them in a separate header file if they are large.

```eiffel
c_calculate_distance (x1, y1, x2, y2: REAL_64): REAL_64
    external
        "C inline use <math.h>"
    alias
        "[
            typedef struct {
                double x;
                double y;
            } Point;
            
            Point p1 = { $x1, $y1 };
            Point p2 = { $x2, $y2 };
            
            double dx = p1.x - p2.x;
            double dy = p1.y - p2.y;
            
            return sqrt(dx*dx + dy*dy);
        ]"
    end
```

### 6.2. Including Custom Headers
If you have a local C header (e.g., `my_lib.h`), you can include it:

```eiffel
external
    "C inline use \"my_lib.h\""
```
Ensure `my_lib.h` is in the include path or the same directory.

---

## 7. Best Practices

1.  **Keep C Minimal:** Logic should stay in Eiffel. Use C only for what Eiffel *cannot* do efficiently or natively.
2.  **Safety First:**
    *   Validate array bounds in Eiffel *before* passing to C.
    *   Check for `NULL` pointers in C if there's any risk.
3.  **Encapsulation:** Wrap all `external "C inline"` features in `feature {NONE}` so they are not accessible to public clients. Expose safe, high-level Eiffel wrappers instead.
4.  **Portability:** Avoid OS-specific headers (like `<windows.h>` or `<unistd.h>`) unless you wrap them in strictly platform-specific classes or use `#ifdef` blocks.

**Example Encapsulation Pattern:**
```eiffel
class MATH_WRAPPER
feature -- Access
    add (a, b: INTEGER): INTEGER
        do
            Result := c_add(a, b)
        end

feature {NONE} -- Implementation
    c_add (a, b: INTEGER): INTEGER
        external "C inline" alias "..." end
end
```

---

## 8. Troubleshooting

| Symptom | Possible Cause | Solution |
| :--- | :--- | :--- |
| **Linker Error:** `unresolved external symbol` | Missing libraries or object files. | Ensure `.lib` or `.a` files are added to the `.ecf` configuration under `external_object` or `library`. |
| **Runtime Crash:** `Segmentation Fault` | Invalid pointer access. | Check generic C pointer logic. Ensure `MANAGED_POINTER` size > 0. Ensure index is within bounds (0 to count-1). |
| **Compiler Error:** `Unknown type` | Missing header inclusion. | Add `use <header.h>` to the external clause. |
| **Garbage Data in C** | Passing Eiffel object address directly. | Use `MANAGED_POINTER` for data buffers or `C_STRING` for strings. |

---

## 9. Conclusion

Inline C is a mechanism to extend Eiffel's capabilities. By following the "Safe Wrapper" pattern—where unsafe C code is hidden behind robust Eiffel contracts—you get the best of both worlds: the raw power of C and the reliability of design-by-contract Eiffel.
