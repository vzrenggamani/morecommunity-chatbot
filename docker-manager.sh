#!/bin/bash

# Docker management script for Unix systems

case "$1" in
    "dev")
        echo "Starting development environment with hot reload..."
        docker-compose -f docker-compose.dev.yml up --build
        ;;
    "dev-bg")
        echo "Starting development environment in background..."
        docker-compose -f docker-compose.dev.yml up -d --build
        ;;
    "prod")
        echo "Starting production environment..."
        docker-compose -f docker-compose.prod.yml up --build
        ;;
    "prod-bg")
        echo "Starting production environment in background..."
        docker-compose -f docker-compose.prod.yml up -d --build
        ;;
    "stop-dev")
        echo "Stopping development environment..."
        docker-compose -f docker-compose.dev.yml down
        ;;
    "stop-prod")
        echo "Stopping production environment..."
        docker-compose -f docker-compose.prod.yml down
        ;;
    "logs-dev")
        echo "Showing development logs..."
        docker-compose -f docker-compose.dev.yml logs -f
        ;;
    "logs-prod")
        echo "Showing production logs..."
        docker-compose -f docker-compose.prod.yml logs -f
        ;;
    "clean")
        echo "Cleaning up all containers and volumes..."
        docker-compose -f docker-compose.dev.yml down -v
        docker-compose -f docker-compose.prod.yml down -v
        ;;
    *)
        echo "Usage: $0 [command]"
        echo
        echo "Commands:"
        echo "  dev       - Start development environment"
        echo "  dev-bg    - Start development environment in background"
        echo "  prod      - Start production environment"
        echo "  prod-bg   - Start production environment in background"
        echo "  stop-dev  - Stop development environment"
        echo "  stop-prod - Stop production environment"
        echo "  logs-dev  - Show development logs"
        echo "  logs-prod - Show production logs"
        echo "  clean     - Clean up all containers and volumes"
        ;;
esac
