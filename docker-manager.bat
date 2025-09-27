@echo off
setlocal

if "%1"=="dev" (
    echo Starting development environment with hot reload...
    docker-compose -f docker-compose.dev.yml up --build
) else if "%1"=="dev-bg" (
    echo Starting development environment in background...
    docker-compose -f docker-compose.dev.yml up -d --build
) else if "%1"=="prod" (
    echo Starting production environment...
    docker-compose -f docker-compose.prod.yml up --build
) else if "%1"=="prod-bg" (
    echo Starting production environment in background...
    docker-compose -f docker-compose.prod.yml up -d --build
) else if "%1"=="stop-dev" (
    echo Stopping development environment...
    docker-compose -f docker-compose.dev.yml down
) else if "%1"=="stop-prod" (
    echo Stopping production environment...
    docker-compose -f docker-compose.prod.yml down
) else if "%1"=="logs-dev" (
    echo Showing development logs...
    docker-compose -f docker-compose.dev.yml logs -f
) else if "%1"=="logs-prod" (
    echo Showing production logs...
    docker-compose -f docker-compose.prod.yml logs -f
) else if "%1"=="clean" (
    echo Cleaning up all containers and volumes...
    docker-compose -f docker-compose.dev.yml down -v
    docker-compose -f docker-compose.prod.yml down -v
) else (
    echo Usage: %0 [command]
    echo.
    echo Commands:
    echo   dev       - Start development environment
    echo   dev-bg    - Start development environment in background
    echo   prod      - Start production environment
    echo   prod-bg   - Start production environment in background
    echo   stop-dev  - Stop development environment
    echo   stop-prod - Stop production environment
    echo   logs-dev  - Show development logs
    echo   logs-prod - Show production logs
    echo   clean     - Clean up all containers and volumes
)
