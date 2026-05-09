# LOCAL_AGENTS.example.md

Copy this file to `LOCAL_AGENTS.md` for machine-specific agent instructions.
`LOCAL_AGENTS.md` is intentionally untracked.

## Optional local tool routing

- Use standard `git` and `gh` CLI commands first for git and GitHub operations.
- Use repo-specific git/GitHub MCP namespaces, such as
  `<repo_git_mcp_namespace>` and `<repo_github_mcp_namespace>`, only when the
  CLI has a gap, the CLI is unavailable, or the user explicitly requests MCP
  usage.

## Optional local workflow preferences

- Prefer shell commands for git/GitHub work unless one of the MCP exceptions
  above applies.
- Keep local secrets in environment variables, not in repository files.
