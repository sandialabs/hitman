:: Open cmd folder with conda environment activated
::
:: ENV_NAME can be extracted from environment.yml file automatically.
:: If conda environment is not defined locally, will automatically create conda environment from *.yml if found
::
:: Windows only :)

@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

:: Specify repo location. Note, %~dp0 is the drive/folder of THIS file.
set repo_path="%~dp0"
:: Specify conda env name. If not defined, looks in environment.yml
set ENV_NAME=

:: Environment file
SET ENV_FILE=environment.yml

:: Activate conda - before (potentially) changing drives
call "%HOMEDRIVE%%HOMEPATH%\AppData\Local\anaconda3\Scripts\activate.bat"

:: Change directory to repo (may be different drive)
cd /d %repo_path%


:: Establish environment name, if not user-specified
IF NOT defined ENV_NAME (
    :: Look for *.yml
    if not exist %ENV_FILE% (
        echo Could not find %ENV_FILE%
    ) ELSE (
        echo Found %ENV_FILE%
        :: Extract the name from *.yml
        for /f "tokens=1,* delims=: " %%A in ('findstr /C:"name:" "%ENV_FILE%"') do (
            SET ENV_NAME=%%B
        )
        echo Found ENV_NAME=!ENV_NAME!
    )
)

:: Ensure conda env exists
IF defined ENV_NAME (
    :: Check if the conda environment exists
    call conda env list | findstr /C:"%ENV_NAME%" >nul
    IF !ERRORLEVEL! NEQ 0 (
        :: Env does not exist, attempt to install from *.yml
        echo    The conda environment "%ENV_NAME%" does not exist.
        echo    Installing from "%ENV_FILE%"...
        call conda env create -f "%ENV_FILE%"
        IF !ERRORLEVEL! EQU 0 (
            echo    Environment "!ENV_NAME!" has been created successfully.
        ) ELSE (
            echo Failed to create the environment.
            SET ENV_NAME=
        )
    ) else (
    echo Found existing conda environment.
    )
) else (
    echo ENV_NAME is not defined! Could not find conda env.
)

:: Activate environment
IF defined ENV_NAME (
    :: Note, call required because 'conda activate' does not play well with bat files
    call activate %ENV_NAME%
) else (
    echo ENV_NAME is not defined! Could not activate conda env.
)

:: Leave terminal open
cmd /k
