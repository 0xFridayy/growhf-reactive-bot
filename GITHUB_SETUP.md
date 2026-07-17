# Push to GitHub

One-time setup to get this repo on GitHub.

## 1. Create GitHub Repo

Go to https://github.com/new

- **Repository name**: `growhf-reactive-bot` (or similar)
- **Description**: "Reactive volume+price spike detector for OKX & Hyperliquid. Small-account optimized ($300 sizing)."
- **Public** or **Private** (your choice)
- Do NOT initialize with README (we have one)
- Click **Create repository**

Copy the HTTPS URL (e.g., `https://github.com/yourusername/growhf-reactive-bot.git`)

## 2. Push Local Repo

```bash
cd "C:\Users\jason\Desktop\NeoBDM _ Broker Stalker_files\crypto-perp-screener"

# Add remote
git remote add origin https://github.com/YOUR_USERNAME/growhf-reactive-bot.git

# Verify
git remote -v

# Push (might prompt for GitHub token)
git branch -M main
git push -u origin main
```

**If prompted for credentials:**
- Use GitHub username
- Generate a Personal Access Token: https://github.com/settings/tokens
  - Scopes: `repo`, `workflow`
  - Use token as password

## 3. Verify

Visit `https://github.com/YOUR_USERNAME/growhf-reactive-bot`

You should see:
- `growhf_reactive_bot.py` (main bot)
- `README.md` (documentation)
- `QUICKSTART.md` (quick start)
- `deploy/` folder (daemon configs)
- `.github/workflows/` (CI/CD)

## 4. Optional: GitHub Pages

Enable GitHub Pages to host documentation:

- Go to Settings → Pages
- Source: `main` branch, root folder
- Wait for build

## 5. Optional: GitHub Actions Secrets

If adding OKX API keys (for execution):

- Settings → Secrets and variables → Actions
- Add secrets:
  - `OKX_API_KEY`
  - `OKX_API_SECRET`
  - `OKX_PASSPHRASE`

These can be used in workflows without exposing in logs.

## Future Updates

After first push, just use:

```bash
git add -A
git commit -m "Fix: improve spike detection"
git push
```

---

Done! Repo is now public and shareable.
