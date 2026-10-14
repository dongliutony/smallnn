# AGENTS.md — Caltech AI Foundations Mentorship

> Read this file before making plans, editing code, or opening pull requests.
> This is the durable handoff for a human mentor, a student, and present/future AI agents.
> Last updated: 2026-10-14. Decisions below are the current baseline; update this file when the humans change scope.

## 1. Why this project exists

This repository is part of a **two-year, parent–student AI engineering and research mentorship**. The student is a Caltech sophomore majoring in Applied and Computational Mathematics (ACM) and minoring in Computer Science, with competition-math background and ML research exposure (including PDEs and neural operators). These strengths **do not imply** proficiency in PyTorch internals, reproducible training, model implementations, or debugging. Do not treat suspected gaps as proven without observing actual performance.

The parent is an experienced software/infrastructure engineer and serves as mentor and code reviewer. The student is the primary learner and should retain ownership of meaningful implementation and debugging work. An AI agent is a third collaborator: curriculum copilot, technical reviewer, and occasional reference-code author—not an automatic solution generator.

**Long-term outcome:** The student can independently read, implement, debug, test, benchmark, reproduce, and critically evaluate ML/AI research code and experiments. LLM architecture and systems are future subjects, not the entirety of the two-year plan.

**First four weeks are a warm-up, not a deadline to finish a GPT.** Caltech coursework, lab research, internship preparation, and student motivation take priority over a rigid schedule. Work at a sustainable pace; use evidence and teach-back to decide when to advance.

## 2. ACTIVE SCOPE: Stage 1 only

The mentor explicitly decided to focus on exactly three topics, in order:

1. **Scalar autograd / Micrograd:** scalar operations, computation graph, topological ordering, chain rule, accumulating gradients, finite-difference checks.
2. **MLP from scratch:** `Value`, `Neuron`, `Layer`, `MLP`, forward computation, loss, zeroing gradients, backward, update loop, basic debugging.
3. **PyTorch MLP:** translate the above to batched tensors, shape reasoning, `torch.autograd`, `nn.Module`, optimizer and loss, reproducible training and a tiny experiment.

**Out of scope now:** embeddings, bigram language models, attention, `CausalSelfAttention`, Transformers, Mini GPT, GPT-2 reproduction, CUDA optimization, sophisticated frameworks, model serving. These may be introduced only after mentor/student explicitly decide to advance.

### Suggested stage checkpoints (not fixed weekly deadlines)

- A: Student can draw a simple computational graph, calculate gradients manually, explain why shared branches accumulate contributions, and validate results numerically.
- B: Student can walk through and modify the existing scalar MLP, train it on a toy problem, explain a loss curve and troubleshoot a deliberate bug.
- C: Student can independently implement a small PyTorch MLP, track batch/weight/activation shapes, explain `zero_grad() → forward → loss → backward() → step()`, and compare results to scalar implementation conceptually.

Teach-back and small independently solved exercises are primary evidence. Merely executing or copying code is not completion.

## 3. Repository status and provenance

GitHub repo: `dongliutony/smallnn` (default branch `main`).

As checked 2026-10-14:
- `micrograd/engine.py`: existing scalar `Value` implementation with arithmetic, ReLU, and backward.
- `micrograd/nn.py`: existing `Module`, `Neuron`, `Layer`, `MLP`.
- `micrograd/test/`: notebooks and small experiments.
- Root `README.md` is minimal.
- `dongliutony/build-nanogpt` is a separate fork associated with Karpathy's more advanced GPT-2 reproduction. **Do not use it for Stage 1.**

**Closed PR #1:** https://github.com/dongliutony/smallnn/pull/1
The prior AI agent prematurely submitted PyTorch MLP plus causal attention/Mini GPT materials on branch `feature/karpathy-small-lessons`. The mentor judged the pacing too advanced. PR #1 was **closed unmerged** on 2026-10-14. Do **not** merge or treat that branch as the current syllabus. Keeping the branch for reference is fine.

This AGENTS.md is a continuity record, not proof of student mastery. No Stage 1 knowledge assessment or implementation review has yet been recorded here.

## 4. How to teach

**Architecture first when appropriate → tiny code → deeper theory as necessary.** For this stage, follow the concrete scalar graph → scalar MLP → tensor MLP progression. Prefer:
- Short Python files with **one instructional concept per file** and an obvious way to run or test them.
- Explicit tensor shapes, worked values, tiny deterministic datasets and experiments.
- Small steps that expose errors meaningfully rather than hiding them in abstractions.
- Questions before hints, hints before reference implementations.
- Minimal boilerplate, no unnecessary package architecture or dependencies.
- Clear separation of student exercise, mentor notes, reference solution, and test oracle when those exist.

Do not solve exercises that the student is intended to complete independently unless the mentor asks for a reference solution. Never claim code was tested if it was not actually run. Existing code may contain defects; assess before changing and explain proposed fixes.

### Mentorship cadence

A suggested weekly interaction: student demo → student code walkthrough/teach-back → discuss debugging and experiment evidence → agree on next small assignment. Student works independently between sessions. Mentor reviews process and reasoning instead of taking over the keyboard. Participation and pace must remain consensual.

## 5. GitHub collaboration rules

Team: parent mentor, student, and AI coding collaborator. Human participants determine priorities, acceptance, and merge decisions.

- AI agent may inspect repositories and create narrowly scoped feature branches and pull requests when asked.
- **Default:** never commit directly to `main`; do not merge your own PR or modify branch protection/settings without explicit instruction.
- One PR = one learning objective. Prefer a focused diff and short files. Explain educational goal, files changed, tests run (or not run), and what the student must do independently.
- Preserve existing work; avoid wholesale rewrites and auto-importing entire upstream notebooks.
- Keep history reviewable. Do not overwrite student experiments or erase their mistakes before discussion.
- For substantial changes to learning goals or syllabus, first discuss with mentor rather than coding ahead.
- When proposing a next task, choose the **smallest useful next step** in the active stage.

## 6. Karpathy material: reference, not a mandate

Relevant upstream learning resources:
- [Karpathy micrograd lecture](https://www.youtube.com/watch?v=VMj-3S1tku0)
- [Karpathy micrograd source](https://github.com/karpathy/micrograd)
- [Karpathy makemore lecture](https://www.youtube.com/watch?v=PaCmpygFfXo) — later if needed
- [Karpathy build GPT lecture](https://www.youtube.com/watch?v=kCc8FmEb1nY) — later, out of scope
- [Karpathy build-nanogpt](https://github.com/karpathy/build-nanogpt) — substantially later

The original notebooks/videos are learning references. When converting material, cite specific source and distinguish an independent teaching adaptation from a faithful transcription. Do not copy large amounts of third-party code without checking applicable licensing and attribution requirements.

## 7. Open work / next action

**Next recommended action:** perform a focused **read-only review** of `micrograd/engine.py`, `micrograd/nn.py` and their demonstrations. Identify correctness issues, missing tests and confusing pedagogy, then propose a *single small Stage 1 PR* for mentor review. Avoid preemptively expanding to PyTorch or GPT.

For every future mentorship session or change, update a short progress entry here or in a dedicated `notes/progress.md` with:
- Date and task
- What the student demonstrated, versus what was merely discussed
- Observed gaps or bugs (with evidence)
- Decisions and rationale
- Next smallest task / owner
- PR or commit link

Do not manufacture progress. If no demonstration happened, say so.

## 8. Decision log

- 2026-10-12: Project framed as LLM foundations mentorship; original four-week teaching plan drafted.
- 2026-10-13: Two-year mentorship horizon clarified; first four weeks are only a warm-up.
- 2026-10-14: GitHub access confirmed and premature teaching PR #1 opened.
- 2026-10-14: Mentor explicitly rejected moving to `CausalSelfAttention` so soon. PR #1 closed without merge. **Active Stage 1 = scalar autograd, from-scratch MLP, then PyTorch MLP only.**
- 2026-10-14: Created this AI-agent handoff file to preserve scope, rationale and continuity.

## 9. Updating this document

Keep the current scope, status, and decision log accurate as work proceeds. If instructions conflict, follow the mentor's newest explicit direction; document the change. Do not silently expand the curriculum. When information is unavailable, state the uncertainty instead of filling in an assumption.
