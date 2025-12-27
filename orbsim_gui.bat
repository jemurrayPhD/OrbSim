@echo off
setlocal

set "VENV_ACTIVATE=.venv\Scripts\activate.bat"
if exist "%VENV_ACTIVATE%" (
  call "%VENV_ACTIVATE%"
)

python -m orbsim.app %*
