@echo off
rem ===========================================================================
rem get_slakh.cmd - download Slakh2100 (CC BY 4.0) and build the drumless
rem training backings. Safe to stop and re-run at any time:
rem   step 1 resumes the download from the last byte (curl -C -),
rem   step 2 rebuilds the backings from the local archive in minutes.
rem Needs ~104 GB for the archive plus ~45 GB for the backings.
rem ===========================================================================
setlocal
cd /d "%~dp0\..\.."

set ARCHIVE=data\slakh2100_flac_redux.tar.gz
set URL=https://zenodo.org/api/records/4599666/files/slakh2100_flac_redux.tar.gz/content

if not exist data mkdir data

echo [1/2] Downloading %ARCHIVE% (resumes if partially downloaded)...
:retry
curl -L -C - --retry 20 --retry-delay 15 -A "drum_along-dataset-builder/1.0" -o "%ARCHIVE%" "%URL%"
if errorlevel 1 (
    echo Download interrupted, retrying in 30 s. Close this window to stop; re-run to resume.
    timeout /t 30 /nobreak >nul
    goto retry
)

echo [2/2] Building drumless backings from the local archive...
python -I scripts\data\slakh_stream_build.py data\slakh --workers 4 --url "%ARCHIVE%"
if errorlevel 1 (
    echo Build failed. Re-run this script; the download will not repeat.
    exit /b 1
)
echo Done. Tell Claude the Slakh data is ready.
endlocal
