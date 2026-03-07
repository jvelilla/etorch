---
description: Start a new Eiffel project using Contract-First requirements
---
This workflow generates the foundational structure and initial Draft Contracts for a new Eiffel project using the Context Engineering methodology.

1. **Initialize Context**: Read the detailed requirements from the user and the core context files (`Eiffel.md`, `Eiffel_style.md`, `Eiffel_inheritance.md`, `llm_prompt.md`) using the `view_file` tool.
2. **Project Scaffolding**: Generate the `.ecf` file, `src/` directory, and `testing/` directory following the exact constraints found in `llm_prompt.md`.
3. **Draft Contracts & Architect**: Using the provided requirements, create the initial deferred (or effective) Eiffel classes. 
    * Ensure the architecture respects the BON method concepts discussed in `Eiffel_inheritance.md`.
    * Ensure every stateful class has an `invariant`.
    * Ensure all public routines outline their responsibilities using explicit `require` and `ensure` clauses.
    * Do NOT implement the actual bodies (`do ... end`) of the routines yet, unless trivial.
4. **Compile Check**: Run `ec -config <PROJECT_NAME>.ecf -target <PROJECT_NAME> -c_compile` to verify that the proposed architecture and contracts are syntactically valid.
// turbo
5. **Autocorrect Architecture**: If compilation fails, analyze and fix the contracts/signatures until they are clean.
6. **Notify User**: Once the initial architecture compiles cleanly, stop and present the generated `.ecf` and Eiffel class files to the user for structural review.
