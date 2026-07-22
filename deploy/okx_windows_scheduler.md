# Run the OKX Bots Locally on Windows (Task Scheduler)

Run both bots 24/7 on your own PC — no cloud droplet, no monthly cost. The only
requirement is that **the PC is powered on and online** when you want alerts
(a laptop that sleeps won't send alerts while asleep).

You run **two** bots as **two** scheduled tasks, both sharing the same
`config.json` and the same Telegram bot/chat:

| Task | Script | Does |
|------|--------|------|
| `OKX-Bot`   | `okx_tele_bot.py`      | `/analyze` commands + OI/funding-flip alerts |
| `OKX-Spike` | `okx_perp_screener.py` | reactive price + volume spike alerts |

Only `OKX-Bot` listens for Telegram commands, so there's no conflict on the
shared bot token.

---

## Step 0: Install

1. Install **Python 3.10+** from <https://python.org> — tick **"Add Python to PATH"**.
2. Double-click **`deploy\install_okx_windows.bat`**. It builds the venv, installs
   dependencies, and creates `run_okx_bot.bat` and `run_okx_spike.bat` in the
   repo folder.
3. Edit **`config.json`** — set `telegram_bot_token` and `telegram_chat_id`.
4. Test each bot by double-clicking `run_okx_bot.bat` (you should get
   "OKX Telegram bot online" in Telegram) and `run_okx_spike.bat`. Close the
   windows once you've confirmed they work.

---

## Step 1: Create the hidden launchers (no console window)

In the **repo root** (same folder as the `.py` files), create two `.vbs` files.
These launch the bots without a visible terminal window.

`run_okx_bot_hidden.vbs`:

```vbs
Set objFSO = CreateObject("Scripting.FileSystemObject")
strPath = objFSO.GetParentFolderName(WScript.ScriptFullName)
Set objShell = CreateObject("WScript.Shell")
objShell.Run chr(34) & strPath & "\run_okx_bot.bat" & chr(34), 0, False
```

`run_okx_spike_hidden.vbs`:

```vbs
Set objFSO = CreateObject("Scripting.FileSystemObject")
strPath = objFSO.GetParentFolderName(WScript.ScriptFullName)
Set objShell = CreateObject("WScript.Shell")
objShell.Run chr(34) & strPath & "\run_okx_spike.bat" & chr(34), 0, False
```

---

## Step 2: Create the scheduled tasks

Do this **twice** — once for each bot. Open Task Scheduler (`Win+R` →
`taskschd.msc`), then **Create Task** (not "Basic Task").

### Task 1 — OKX-Bot

**General tab**
- Name: `OKX-Bot`
- Select **"Run whether user is logged on or not"**
- Check **"Run with highest privileges"**

**Triggers tab** → New
- Begin the task: **At startup**
- Check **"Repeat task every 5 minutes"**, Duration: **Indefinitely**
  (this restarts the bot if it ever crashes)

**Actions tab** → New
- Action: **Start a program**
- Program/script: `C:\Windows\System32\wscript.exe`
- Add arguments: `"C:\path\to\repo\run_okx_bot_hidden.vbs"`
- Start in: `C:\path\to\repo`

**Conditions tab**
- Uncheck **"Start the task only if the computer is on AC power"**
- Uncheck **"Stop if the computer switches to battery power"**

**Settings tab**
- Check **"Run task as soon as possible after a scheduled start is missed"**
- Check **"If the task fails, restart every 1 minute"**, up to 3 attempts
- **"Do not start a new instance"** as the concurrency rule (so it never
  double-launches)

Click OK.

### Task 2 — OKX-Spike

Repeat exactly the same steps with:
- Name: `OKX-Spike`
- Arguments: `"C:\path\to\repo\run_okx_spike_hidden.vbs"`

> Replace `C:\path\to\repo` with your real path, e.g.
> `C:\Users\You\growhf-reactive-bot`.

---

## Step 3: Start them now

In Task Scheduler, right-click each task → **Run**. Within a few seconds you
should see the "online" message in Telegram and, on the next scan, any
OI/funding-flip or spike alerts.

---

## Managing the bots

**Are they running?**
```powershell
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id, StartTime
```

**Stop one** (Task Scheduler will restart it on its 5-min trigger, so disable
the task first if you want it to stay down):
```powershell
# Disable so it doesn't auto-restart, then kill:
Get-ScheduledTask -TaskName OKX-Bot | Disable-ScheduledTask
Get-Process python | Where-Object { $_.Path -like "*venv*" } | Stop-Process -Force
```

**Update the bots** (after `git pull`): just let the 5-minute trigger restart
them, or right-click each task → **End**, then **Run**.

**Remove:** Task Scheduler → right-click task → **Delete**.

---

## Troubleshooting

- **No "online" message** → run `run_okx_bot.bat` directly in a terminal and
  read the error. Usually a bad token/chat_id in `config.json`, or Python not
  on PATH.
- **`409 Conflict` from Telegram** → two processes are polling `getUpdates` on
  the same token. Only `okx_tele_bot.py` should do that — make sure you didn't
  start two copies of `OKX-Bot`.
- **Alerts stop overnight** → the PC slept. Set the power plan to never sleep,
  or run it on an always-on machine (an old PC, a mini-PC, or a Raspberry Pi).

---

## Note on the GitHub auto-deploy

The `.github/workflows/deploy.yml` and `install_okx_bot.sh` in this repo are for
a **Linux server** deployment. If you're running locally on Windows you don't
need them or any GitHub secrets — to update, just `git pull` and let Task
Scheduler restart the bots.

---

## Mac / Linux instead?

Not on Windows? On macOS/Linux the simplest local equivalent is a background
run inside a `tmux`/`screen` session, or a user-level `systemd --user` service
on Linux. The two `run_*` commands are just:

```bash
cd /path/to/repo
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt
python okx_tele_bot.py      # in one session
python okx_perp_screener.py # in another
```
