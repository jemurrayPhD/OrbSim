@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%"

set "VENV_ACTIVATE=%SCRIPT_DIR%.venv\Scripts\activate.bat"
if exist "%VENV_ACTIVATE%" (
  call "%VENV_ACTIVATE%"
)

set "PYTHONPATH=%SCRIPT_DIR%src;%PYTHONPATH%"

python -m orbsim.app %*
popd
