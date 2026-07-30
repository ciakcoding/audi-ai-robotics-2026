# Repository Setup

This guide explains how to set up the repository, configure your local workspace, and follow the team development workflow.

---

## Clone the Repository

Clone the repository:

```bash
git clone https://github.com/aim-t/audi-ai-robotics-2026.git
cd audi-ai-robotics-2026
````

---

## Environment Setup

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

---

## Fetch Remote Branches

Fetch all remote branches:

```bash
git fetch --all
```

---

## Available Branches

List available remote branches:

```bash
git branch -r
```

Expected output:

```text
origin/main
origin/develop
origin/feature/simulation
origin/feature/rl
origin/feature/evaluation
origin/feature/integration
```

To view both local and remote branches:

```bash
git branch -a
```

---
## Switching among branches

```bash
git switch demo/final-gui
```
---

## Checkout Your Feature Branch

Create a local tracking branch from the remote branch.

### Simulation

```bash
git checkout -b feature/simulation origin/feature/simulation
```

### RL

```bash
git checkout -b feature/rl origin/feature/rl
```

### Evaluation

```bash
git checkout -b feature/evaluation origin/feature/evaluation
```

### Integration

```bash
git checkout -b feature/integration origin/feature/integration
```

---

## Verify Current Branch

Check your current branch:

```bash
git branch
```

Example:

```text
* feature/rl
  develop
  main
```

The branch marked with `*` is the active branch.

---

# Repository Structure

```
assets/          # Data and static resources
baselines/       # Baseline implementations
docs/            # Documentation
envs/            # Simulation environments
evaluation/      # Evaluation scripts
models/          # Model definitions
outputs/         # Generated results
scripts/         # Utility scripts
tests/           # Tests
train/           # Training pipelines

README.md
requirements.txt
.gitignore
```

---

# Daily Workflow

## Update Your Branch

Before starting work:

```bash
git pull
```

---

## Stage Changes

Add modified files:

```bash
git add .
```

---

## Commit Changes

Create a commit:

```bash
git commit -m "feat: short description"
```

---

## Push Changes

Push your branch:

```bash
git push
```

---

# Branch Workflow

The repository follows the workflow:

```
feature/*
      |
      | Pull Request
      v
develop
      |
      | Integration / Release
      v
main
```

## Rules

* Never commit directly to `main`.
* Changes must be merged through pull requests.
* Feature branches should be created from `develop`.
* Keep branches synchronized with the latest changes from `develop`.

---

# Branch Ownership

| Branch              | Owner            |
| ------------------- | ---------------- |
| feature/simulation  | Simulation Lead  |
| feature/rl          | RL Lead          |
| feature/evaluation  | Evaluation Lead  |
| feature/integration | Integration Lead |

Only the assigned owner should commit directly to their feature branch.

Changes to another teammate's branch should be discussed before modifying.

---

# Commit Messages

Use the format:

```text
<type>: <short description>
```

Keep commit messages short, descriptive, and written in the present tense.

## Common Types

| Type       | Use                                         |
| ---------- | ------------------------------------------- |
| `feat`     | New feature                                 |
| `fix`      | Bug fix                                     |
| `docs`     | Documentation changes                       |
| `refactor` | Code improvement without behavior change    |
| `test`     | Adding or updating tests                    |
| `chore`    | Repository, configuration, or setup changes |

---

## Examples

```text
feat: add PPO training pipeline
fix: correct reward function
docs: update repository setup
refactor: simplify environment interface
test: add simulation unit tests
chore: initialize project structure
```

---

# Team Guidelines

* Work only on your assigned feature branch.
* Keep commits focused and descriptive.
* Push changes frequently.
* Do not modify another teammate's branch without discussion.
* Create pull requests before merging into `develop`.
* Resolve merge conflicts before requesting review.


