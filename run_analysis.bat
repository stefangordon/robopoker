@echo off
REM Run DuckDB analysis on blueprint parquet file

REM Default to the shard 0 file if no argument provided
set PARQUET_FILE=%1
if "%PARQUET_FILE%"=="" set PARQUET_FILE=pgcopy\blueprint_clusters_0_1.parquet

REM Check if file exists
if not exist "%PARQUET_FILE%" (
    echo Error: File '%PARQUET_FILE%' not found.
    echo Usage: run_analysis.bat [parquet_file]
    exit /b 1
)

echo Analyzing: %PARQUET_FILE%
echo Running DuckDB analysis...
echo.

REM Run the Python script instead as it's more portable on Windows
python analyze_blueprint.py "%PARQUET_FILE%"