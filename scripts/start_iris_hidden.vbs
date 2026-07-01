' Launch Iris with no visible console window.
' Intended for the Windows Startup folder so the app runs in the background
' at login and the Ctrl+Alt+E / Ctrl+Alt+Q hotkeys are always available.
'
' It prefers the project's .venv (pythonw.exe = no console); if that is not
' found it falls back to a system-wide pythonw on PATH.
Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

' Project root = parent folder of this script's folder (scripts\..).
base = fso.GetParentFolderName(fso.GetParentFolderName(WScript.ScriptFullName))
sh.CurrentDirectory = base

venvPy = base & "\.venv\Scripts\pythonw.exe"
If fso.FileExists(venvPy) Then
    ' 0 = hidden window, False = do not wait for it to finish.
    sh.Run """" & venvPy & """ main.py", 0, False
Else
    sh.Run "pythonw main.py", 0, False
End If
