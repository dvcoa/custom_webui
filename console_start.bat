@echo off

goto :activate_venv

:launch
%PYTHON% whisper_stt.py %*
pause

:activate_venv
set PYTHON="%~dp0\venv\Scripts\Python.exe"
echo venv %PYTHON%
goto :launch

:endofscript

echo.
echo Launch unsuccessful. Exiting.
pause


