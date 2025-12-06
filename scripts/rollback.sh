#!/bin/bash
# =============================================================================
# CineMatch V2.1.6 - Automated Rollback Script
# Task 4.8: Automated rollback procedures
# =============================================================================
# This script provides automated rollback capabilities for CineMatch deployments.
#
# Usage:
#   ./rollback.sh [options]
#
# Options:
#   --environment, -e    Target environment (dev, staging, prod)
#   --revision, -r       Specific revision to rollback to
#   --component, -c      Component to rollback (ui, api, all)
#   --dry-run            Show what would be done without making changes
#   --force              Skip confirmation prompts
#   --help, -h           Show this help message
#
# Examples:
#   ./rollback.sh -e production                    # Rollback to previous version
#   ./rollback.sh -e staging -r 5                  # Rollback to revision 5
#   ./rollback.sh -e prod -c api --dry-run        # Preview API rollback
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="cinematch"
LOG_FILE="${SCRIPT_DIR}/logs/rollback-$(date +%Y%m%d-%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT=""
REVISION=""
COMPONENT="all"
DRY_RUN=false
FORCE=false

# =============================================================================
# Logging Functions
# =============================================================================

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Console output
    case "$level" in
        INFO)  echo -e "${GREEN}[INFO]${NC} $message" ;;
        WARN)  echo -e "${YELLOW}[WARN]${NC} $message" ;;
        ERROR) echo -e "${RED}[ERROR]${NC} $message" ;;
        DEBUG) echo -e "${BLUE}[DEBUG]${NC} $message" ;;
    esac
    
    # File output
    mkdir -p "$(dirname "$LOG_FILE")"
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
}

info()  { log INFO "$@"; }
warn()  { log WARN "$@"; }
error() { log ERROR "$@"; }
debug() { log DEBUG "$@"; }

# =============================================================================
# Helper Functions
# =============================================================================

show_help() {
    cat << EOF
CineMatch Rollback Script v2.1.6

Usage: $0 [options]

Options:
    --environment, -e    Target environment (dev, staging, prod)
    --revision, -r       Specific revision to rollback to
    --component, -c      Component to rollback (ui, api, all)
    --dry-run            Show what would be done without making changes
    --force              Skip confirmation prompts
    --help, -h           Show this help message

Examples:
    $0 -e production                      # Rollback to previous version
    $0 -e staging -r 5                    # Rollback to revision 5
    $0 -e prod -c api --dry-run          # Preview API rollback

EOF
}

confirm() {
    if [[ "$FORCE" == "true" ]]; then
        return 0
    fi
    
    local prompt="${1:-Are you sure?}"
    read -r -p "$prompt [y/N] " response
    case "$response" in
        [yY][eE][sS]|[yY]) return 0 ;;
        *) return 1 ;;
    esac
}

check_prerequisites() {
    info "Checking prerequisites..."
    
    local missing=()
    
    # Check required commands
    for cmd in kubectl helm jq; do
        if ! command -v "$cmd" &> /dev/null; then
            missing+=("$cmd")
        fi
    done
    
    if [[ ${#missing[@]} -gt 0 ]]; then
        error "Missing required commands: ${missing[*]}"
        exit 1
    fi
    
    # Check kubectl connection
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    info "All prerequisites satisfied"
}

get_namespace() {
    case "$ENVIRONMENT" in
        dev|development)
            echo "${PROJECT_NAME}-dev"
            ;;
        staging|stage)
            echo "${PROJECT_NAME}-staging"
            ;;
        prod|production)
            echo "${PROJECT_NAME}-prod"
            ;;
        *)
            error "Unknown environment: $ENVIRONMENT"
            exit 1
            ;;
    esac
}

get_release_name() {
    local namespace=$(get_namespace)
    echo "${PROJECT_NAME}-${ENVIRONMENT}"
}

# =============================================================================
# Rollback Functions
# =============================================================================

list_revisions() {
    local namespace=$(get_namespace)
    local release=$(get_release_name)
    
    info "Listing Helm revisions for $release in $namespace..."
    
    helm history "$release" -n "$namespace" --max 10 2>/dev/null || {
        warn "No Helm release found, checking Kubernetes deployments..."
        kubectl rollout history deployment -n "$namespace" -l app.kubernetes.io/name=$PROJECT_NAME
    }
}

get_current_revision() {
    local namespace=$(get_namespace)
    local release=$(get_release_name)
    
    helm status "$release" -n "$namespace" -o json 2>/dev/null | jq -r '.version' || echo "0"
}

get_previous_revision() {
    local current=$(get_current_revision)
    echo $((current - 1))
}

rollback_helm() {
    local namespace=$(get_namespace)
    local release=$(get_release_name)
    local target_revision="${REVISION:-$(get_previous_revision)}"
    
    info "Rolling back Helm release $release to revision $target_revision..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "[DRY-RUN] Would execute: helm rollback $release $target_revision -n $namespace"
        helm rollback "$release" "$target_revision" -n "$namespace" --dry-run
        return 0
    fi
    
    # Create rollback record
    local rollback_record="${SCRIPT_DIR}/logs/rollback-record-$(date +%Y%m%d-%H%M%S).json"
    cat > "$rollback_record" << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "environment": "$ENVIRONMENT",
    "release": "$release",
    "from_revision": $(get_current_revision),
    "to_revision": $target_revision,
    "component": "$COMPONENT",
    "initiated_by": "${USER:-unknown}"
}
EOF
    
    # Execute rollback
    if helm rollback "$release" "$target_revision" -n "$namespace" --wait --timeout 10m; then
        info "Helm rollback completed successfully"
        return 0
    else
        error "Helm rollback failed"
        return 1
    fi
}

rollback_deployment() {
    local namespace=$(get_namespace)
    local deployment="$1"
    local target_revision="${REVISION:-0}"  # 0 means previous
    
    info "Rolling back deployment $deployment..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        if [[ "$target_revision" == "0" ]]; then
            info "[DRY-RUN] Would execute: kubectl rollout undo deployment/$deployment -n $namespace"
        else
            info "[DRY-RUN] Would execute: kubectl rollout undo deployment/$deployment -n $namespace --to-revision=$target_revision"
        fi
        return 0
    fi
    
    if [[ "$target_revision" == "0" ]]; then
        kubectl rollout undo deployment/"$deployment" -n "$namespace"
    else
        kubectl rollout undo deployment/"$deployment" -n "$namespace" --to-revision="$target_revision"
    fi
    
    # Wait for rollout to complete
    kubectl rollout status deployment/"$deployment" -n "$namespace" --timeout=5m
}

rollback_component() {
    local namespace=$(get_namespace)
    
    case "$COMPONENT" in
        ui)
            rollback_deployment "${PROJECT_NAME}-ui"
            ;;
        api)
            rollback_deployment "${PROJECT_NAME}-api"
            ;;
        all)
            info "Rolling back all components..."
            rollback_helm
            ;;
        *)
            error "Unknown component: $COMPONENT"
            exit 1
            ;;
    esac
}

# =============================================================================
# Health Check Functions
# =============================================================================

verify_rollback() {
    local namespace=$(get_namespace)
    
    info "Verifying rollback..."
    
    # Check deployment status
    local unhealthy_pods=$(kubectl get pods -n "$namespace" \
        -l app.kubernetes.io/name=$PROJECT_NAME \
        -o jsonpath='{.items[?(@.status.phase!="Running")].metadata.name}')
    
    if [[ -n "$unhealthy_pods" ]]; then
        warn "Some pods are not running: $unhealthy_pods"
        return 1
    fi
    
    # Check readiness
    local not_ready=$(kubectl get pods -n "$namespace" \
        -l app.kubernetes.io/name=$PROJECT_NAME \
        -o jsonpath='{.items[?(@.status.conditions[?(@.type=="Ready")].status!="True")].metadata.name}')
    
    if [[ -n "$not_ready" ]]; then
        warn "Some pods are not ready: $not_ready"
        return 1
    fi
    
    # Run health checks
    info "Running health checks..."
    
    # Get service endpoints
    local ui_service="${PROJECT_NAME}-ui"
    local api_service="${PROJECT_NAME}-api"
    
    # Port-forward and check health (background)
    kubectl port-forward svc/"$ui_service" 8501:8501 -n "$namespace" &
    local ui_pf_pid=$!
    kubectl port-forward svc/"$api_service" 8000:8000 -n "$namespace" &
    local api_pf_pid=$!
    
    sleep 5
    
    # Check UI health
    if curl -sf http://localhost:8501/_stcore/health > /dev/null; then
        info "UI health check passed"
    else
        warn "UI health check failed"
    fi
    
    # Check API health
    if curl -sf http://localhost:8000/health > /dev/null; then
        info "API health check passed"
    else
        warn "API health check failed"
    fi
    
    # Cleanup port-forwards
    kill $ui_pf_pid $api_pf_pid 2>/dev/null || true
    
    info "Rollback verification completed"
    return 0
}

# =============================================================================
# Notification Functions
# =============================================================================

send_notification() {
    local status="$1"
    local message="$2"
    
    # Slack notification (if webhook configured)
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        local color
        case "$status" in
            success) color="good" ;;
            warning) color="warning" ;;
            failure) color="danger" ;;
        esac
        
        curl -s -X POST "$SLACK_WEBHOOK_URL" \
            -H 'Content-type: application/json' \
            -d "{
                \"attachments\": [{
                    \"color\": \"$color\",
                    \"title\": \"CineMatch Rollback - ${ENVIRONMENT}\",
                    \"text\": \"$message\",
                    \"footer\": \"Initiated by ${USER:-unknown}\",
                    \"ts\": $(date +%s)
                }]
            }" || true
    fi
    
    debug "Notification sent: $status - $message"
}

# =============================================================================
# Main
# =============================================================================

main() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --environment|-e)
                ENVIRONMENT="$2"
                shift 2
                ;;
            --revision|-r)
                REVISION="$2"
                shift 2
                ;;
            --component|-c)
                COMPONENT="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --force)
                FORCE=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Validate environment
    if [[ -z "$ENVIRONMENT" ]]; then
        error "Environment is required. Use --environment or -e"
        show_help
        exit 1
    fi
    
    # Start rollback process
    info "Starting CineMatch rollback process..."
    info "Environment: $ENVIRONMENT"
    info "Component: $COMPONENT"
    info "Revision: ${REVISION:-previous}"
    info "Dry run: $DRY_RUN"
    
    # Check prerequisites
    check_prerequisites
    
    # List current revisions
    list_revisions
    
    # Confirm rollback
    echo ""
    if ! confirm "Do you want to proceed with the rollback?"; then
        info "Rollback cancelled"
        exit 0
    fi
    
    # Execute rollback
    info "Executing rollback..."
    send_notification "warning" "Rollback initiated for $COMPONENT component(s)"
    
    if rollback_component; then
        info "Rollback executed successfully"
        
        # Verify rollback
        if verify_rollback; then
            info "Rollback verified successfully"
            send_notification "success" "Rollback completed successfully"
        else
            warn "Rollback verification had warnings"
            send_notification "warning" "Rollback completed with warnings"
        fi
    else
        error "Rollback failed"
        send_notification "failure" "Rollback failed - manual intervention may be required"
        exit 1
    fi
    
    info "Rollback process completed"
    info "Log file: $LOG_FILE"
}

# Run main
main "$@"
