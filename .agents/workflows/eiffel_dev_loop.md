---
description: Run the Eiffel Context Engineering development loop
---
This workflow automates the Eiffel Design-by-Contract development lifecycle with Antigravity.

1. **Initialize Context**: Read the detailed requirements from the user and the core context files (`Eiffel.md`, `Eiffel_style.md`, `Eiffel_inheritance.md`, `llm_prompt.md`) using the `view_file` tool.
2. **Project Check & Scaffolding**: Check if an `.ecf` file exists in the directory. If it already exists, **skip this step entirely**. If it does not exist, generate the `.ecf` file, `src/` directory, and `testing/` directory following the exact constraints found in `llm_prompt.md`.
3. **Draft Contracts**: Create or update the Eiffel specification classes. Ensure every stateful class has an `invariant`, and corresponding routines have explicit `require` and `ensure` clauses.
4. **Compile Phase**: Use the `run_command` tool to compile the target code. Determine the project name and execute: `ec -config <PROJECT_NAME>.ecf -target <PROJECT_NAME> -c_compile -finalize`
// turbo
5. **Autocorrect Loop**: If compilation fails, analyze the compiler output, fix the specified Eiffel file, and return to Step 4. Do not proceed until compilation is clean.
6. **Testing Phase**: Run testing by executing the command: `ec -config <PROJECT_NAME>.ecf -target tests -clean -c_compile -finalize -tests`. Verify that all DbC assertions pass without violations.
// turbo
7. **Git Versioning**: Upon full success (clean compile + passed tests), execute `git add .` and `git commit -m "feat: [Agent] <Desc> (DbC verified)"` using `run_command`.
