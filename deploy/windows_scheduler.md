# Windows Task Scheduler Setup (Always-Run Daemon)

Run GrowiHF bot 24/7 using Windows Task Scheduler (no Terminal window).

## Step 1: Create Python Wrapper Script

Create `run_bot_hidden.vbs` in the screener folder:

```vbs
Set objFSO = CreateObject("Scripting.FileSystemObject")
strPath = objFSO.GetParentFolderName(WScript.ScriptFullName)
Set objShell = CreateObject("WScript.Shell")
objShell.Run chr(34) & strPath & "\run_bot.bat" & chr(34), 0, False
```

This runs the bot WITHOUT showing the Terminal window.

## Step 2: Create the Task

1. **Open Task Scheduler**
   - Press `Win+R`, type `taskschd.msc`, press Enter

2. **Create New Task**
   - Right-click "Task Scheduler Library" → "Create Task"
   - Name: `GrowiHF-Bot`
   - Check "Run with highest privileges"
   - Check "Run whether user is logged in or not"

3. **Triggers Tab**
   - Click "New"
   - Begin the task: `At startup`
   - Repeat task: `Every 5 minutes` (if crashes, restarts)
   - Duration: Indefinitely
   - Click OK

4. **Actions Tab**
   - Click "New"
   - Action: `Start a program`
   - Program: `C:\Windows\System32\cscript.exe`
   - Arguments: `C:\path\to\screener\run_bot_hidden.vbs`
   - Start in: `C:\path\to\screener`
   - Click OK

5. **Conditions Tab**
   - Uncheck "Stop if computer switches to battery power"
   - Uncheck "Don't start if on battery"

6. **Settings Tab**
   - Check "Run task as soon as possible after a scheduled start is missed"
   - Check "If the task fails, restart every 1 minute" (up to 10 attempts)
   - Check "Stop the task if it runs longer than 72 hours"
   - Click OK

7. **Verify**
   - The task should appear in the list
   - Right-click → "Run" to test immediately

## Step 3: Monitor

### View Logs

```bash
# PowerShell (run as admin)
Get-WinEvent -LogName "Application" | Where-Object {$_.ProviderName -like "*GrowiHF*"} | Format-Table TimeCreated, Message
```

Or check `growhf_bot.log` in the screener folder.

### Restart Bot

```bash
# PowerShell (run as admin)
Get-Process python | Where-Object {$_.Path -like "*growhf*"} | Stop-Process -Force
```

Task Scheduler will auto-restart it (see Triggers).

## Step 4: Stop or Disable

- **Disable**: Task Scheduler → right-click task → Disable
- **Delete**: Task Scheduler → right-click task → Delete
- **Logs**: Event Viewer → Windows Logs → Application

## Troubleshooting

**Task runs but no bot activity:**
- Check `growhf_bot.log` for errors
- Verify `config.json` is correct
- Test bot manually: `python growhf_reactive_bot.py` in Terminal

**Task fails to start:**
- Verify Python is installed: `python --version` in Command Prompt
- Verify paths in the task are correct (full absolute paths)
- Run Task Scheduler as Administrator

**Bot crashes but doesn't restart:**
- Increase retry attempts in Settings tab
- Reduce "Every 5 minutes" repeat to "Every 1 minute"

## Alternative: NSSM (Non-Sucking Service Manager)

For a true Windows Service (more robust):

```bash
# Install NSSM
choco install nssm

# Create service
nssm install GrowiHF "C:\path\to\screener\run_bot.bat"
nssm start GrowiHF

# View logs
nssm edit GrowiHF
```

See https://nssm.cc/usage for details.
