---
description: Generate an Eiffel specification from requirements or reverse-engineer existing code
---
This workflow generates formal Eiffel specification classes (using Design by Contract and the BON method) based on either provided requirement documents or by reverse-engineering existing source code.

1. **Initialize Context**: Read the core context files (`Eiffel.md`, `Eiffel_style.md`, `Eiffel_inheritance.md`) using the `view_file` tool to ensure the agent understands Eiffel principles.
2. **Read Inputs**: Read the provided requirement documents, input context files, or existing source code (if reverse-engineering) provided by the user.
3. **Analyze and Map to BON**: Analyze the inputs to identify the core concepts, their relationships, and their responsibilities. Map these to BON (Business Object Notation) concepts: classes, commands, queries, and constraints.
4. **Draft Eiffel Contracts**: Create the deferred (or effective, if appropriate) Eiffel classes representing the specification. 
    * Ensure the architecture respects the Command-Query Separation (CQS) principle.
    * Ensure every stateful class has a strong `invariant`.
    * Ensure all public routines (commands and queries) explicitly define their contractual obligations using `require` and `ensure` clauses.
    * Use descriptive naming conventions as specified in `Eiffel_style.md`.
5. **Output Specification**: Save the generated Eiffel specification classes (`.e` files). If a project structure (`.ecf`) does not exist around them, consider suggesting the user run `/eiffel_new_project` or place them in an appropriate `spec/` folder.
6. **Compile Verification (Optional but Recommended)**: If the files are part of an existing ECF, run the compiler (`ec -c_compile`) to verify the syntax and types of the new specification.
7. **Notify User**: Stop and present the generated Eiffel specification to the user for structural review.
