# Pulls training results (alpha_ml/alpha_status.json + history) that the
# GitHub Actions training job commits back, so the local bot's /alpha_status
# reflects the latest run.
#
# Safe by design: fast-forward only. It will never merge, rebase, reset or
# discard local work — if the branch has diverged it logs and gives up, so a
# background task can never eat uncommitted changes.
#
# The running bot does NOT need restarting: status.py re-reads the JSON on
# every /alpha_status call, so a pull takes effect immediately.

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repo

$logDir = Join-Path $repo "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$log = Join-Path $logDir "sync_repo.log"

function Write-Log($msg) {
    $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Add-Content -Path $log -Value $line -Encoding utf8
    Write-Host $line
}

try {
    $branch = (git rev-parse --abbrev-ref HEAD).Trim()
    $upstream = (git rev-parse --abbrev-ref --symbolic-full-name "@{u}" 2>$null)
    if (-not $upstream) {
        Write-Log "SKIP: branch '$branch' has no upstream configured"
        exit 0
    }
    $upstream = $upstream.Trim()

    git fetch origin --quiet
    if ($LASTEXITCODE -ne 0) { Write-Log "SKIP: fetch failed (offline?)"; exit 0 }

    $behind = (git rev-list --count "HEAD..$upstream").Trim()
    if ($behind -eq "0") { exit 0 }   # already current: stay quiet, this runs often

    # --ff-only: refuses rather than creating a merge commit if we have diverged.
    $out = git merge --ff-only "$upstream" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Log "pulled $behind commit(s) from $upstream"
    } else {
        Write-Log "SKIP: cannot fast-forward $branch onto $upstream (diverged) -- $out"
    }
} catch {
    Write-Log "ERROR: $_"
}
