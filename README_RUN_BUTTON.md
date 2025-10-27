How to use the Run / Debug button in VS Code for this project

1. Open this folder in VS Code.
2. Make sure the Python extension is installed.
3. The workspace is configured to use the venv at `.venv`.
   - This is set in `.vscode/settings.json`.
4. Open the Run and Debug view (Ctrl+Shift+D or the Run icon).
5. In the configuration dropdown, choose `Python: Run SmallWrldModel` to run the model file.
6. Press the green arrow (Run) or F5 to start the debug/run session.

Notes:
- `Python: Current File (WS venv)` will run whatever file you currently have open using the workspace venv.
- If you don't see the venv in the interpreter list, open the Command Palette and run `Python: Select Interpreter`, then select `/Users/prx./Documents/SNA/.venv/bin/python`.
- For headless servers, the plot window may not appear; consider saving the plot to a file instead.
