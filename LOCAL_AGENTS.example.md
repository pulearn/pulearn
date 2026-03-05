# LOCAL_AGENTS.example.md

Copy this file to `LOCAL_AGENTS.md` for machine-specific agent instructions.
`LOCAL_AGENTS.md` is intentionally untracked.

## Optional local MCP routing

- For this repo, prefer a repo-specific git MCP namespace such as
  `<repo_git_mcp_namespace>` for git operations.
- For GitHub operations, prefer a repo-specific namespace like
  `<repo_github_mcp_namespace>` when available.
- Fall back to shared (non-repo-specific) GitHub MCPs only when repo-specific
  MCPs are unavailable.

## Optional local workflow preferences

- Prefer MCP tools over shell when both are available.
- Keep local secrets in environment variables, not in repository files.
