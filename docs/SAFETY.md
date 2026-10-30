# Safety and privacy

## Default-deny behavior

- Offline fixtures are the recommended demonstration path.
- Research-note writes default to disabled.
- Ranking-view and paper-evaluation writes have separate gates and also default to disabled.
- The project does not place trades or connect to a brokerage.
- Redirects and oversized responses are rejected at provider boundaries.
- Errors are sanitized before they enter run summaries.

## Local-only material

Keep these out of Git:

- `.env` and credentials
- SQLite databases, WAL files, and backups
- raw provider caches
- personal watchlists or entity registries
- generated reports and evaluation captures
- machine-specific paths and automation state

The repository's `.gitignore` blocks these common paths and file types. Before publishing a change, inspect both the staged files and the commit history; deleting a secret in a later commit does not remove it from history.

## If a credential is exposed

Revoke or rotate the credential with its provider first. Removing it from a file is not sufficient. Then clean the repository history before publishing or distributing another clone.
