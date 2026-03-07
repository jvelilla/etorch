---
description: Extract an Eiffel project from an existing codebase based on a specification
---
This workflow is designed to build a new Eiffel project by extracting, refactoring, and migrating code from an existing codebase (e.g., a specific branch or project), guided by a specification document.

1. **Initialize Context**: Read the core context files (`Eiffel.md`, `Eiffel_style.md`, `Eiffel_inheritance.md`) and the architectural specification document (e.g., `spec.md` or `SPECIFICATION.md`) using the `view_file` tool.
2. **Setup Project Scaffold**: If the target `.ecf` and directory structure (e.g., `src/`, `tests/`) do not exist, generate them according to the repository structure defined in the specification.
3. **Analyze Source Codebase**: Read the original source files from the existing codebase (the source to be extracted/ported) using `view_file` or `grep_search`.
4. **Iterative Extraction & Contract Enforcement**: For each core element or module described in the specification:
    * Extract and adapt the existing code into the new Eiffel project structure.
    * Apply strict Design by Contract (DbC) principles. Ensure every stateful class has an `invariant` and public routines have `require` and `ensure` clauses.
    * Refactor as necessary to meet the target design outlined in the specification.
5. **Compile Verification**: After extracting a cohesive set of classes, run the Eiffel compiler (e.g., `ec -config <PROJECT_NAME>.ecf -target <PROJECT_NAME> -c_compile`) to verify syntax, types, and contracts.
// turbo
6. **Autocorrect Architecture**: If compilation fails, analyze the compiler output and fix the extracted code until it is structurally sound.
7. **Notify User**: Stop and present the newly extracted and verified modules to the user for review before proceeding with the rest of the extraction plan.
