#!/bin/bash
# =============================================================================
# CineMatch V2.1.6 - Backup and Recovery Script
# Task 4.9: Backup and recovery procedures
# =============================================================================
# Comprehensive backup and recovery for CineMatch data and models.
#
# Backup types:
#   - Full: Complete backup of all data and models
#   - Incremental: Changes since last backup
#   - Models: ML models only
#   - Data: Dataset files only
#   - Config: Configuration and secrets
#
# Storage backends:
#   - Local filesystem
#   - AWS S3
#   - Google Cloud Storage
#   - Azure Blob Storage
#
# Usage:
#   ./backup.sh backup [options]      # Create backup
#   ./backup.sh restore [options]     # Restore from backup
#   ./backup.sh list [options]        # List available backups
#   ./backup.sh verify [options]      # Verify backup integrity
#   ./backup.sh cleanup [options]     # Remove old backups
#
# Examples:
#   ./backup.sh backup --type full --destination s3
#   ./backup.sh restore --backup-id 20240115-120000 --component models
#   ./backup.sh list --destination local --limit 10
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PROJECT_NAME="cinematch"
VERSION="2.1.6"

# Default paths
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data}"
MODELS_DIR="${MODELS_DIR:-$PROJECT_ROOT/models}"
CONFIG_DIR="${CONFIG_DIR:-$PROJECT_ROOT/config}"
BACKUP_LOCAL_DIR="${BACKUP_LOCAL_DIR:-$PROJECT_ROOT/backups}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"

# Backup settings
BACKUP_RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-30}"
BACKUP_COMPRESSION="${BACKUP_COMPRESSION:-gzip}"  # gzip, bzip2, xz, none
BACKUP_ENCRYPT="${BACKUP_ENCRYPT:-false}"
BACKUP_ENCRYPT_KEY="${BACKUP_ENCRYPT_KEY:-}"

# Cloud storage settings
AWS_S3_BUCKET="${AWS_S3_BUCKET:-}"
AWS_S3_PREFIX="${AWS_S3_PREFIX:-cinematch/backups}"
GCS_BUCKET="${GCS_BUCKET:-}"
GCS_PREFIX="${GCS_PREFIX:-cinematch/backups}"
AZURE_CONTAINER="${AZURE_CONTAINER:-}"
AZURE_PREFIX="${AZURE_PREFIX:-cinematch/backups}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# =============================================================================
# Logging
# =============================================================================

LOG_FILE="$LOG_DIR/backup-$(date +%Y%m%d).log"
mkdir -p "$LOG_DIR"

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case "$level" in
        INFO)  echo -e "${GREEN}[INFO]${NC} $message" ;;
        WARN)  echo -e "${YELLOW}[WARN]${NC} $message" ;;
        ERROR) echo -e "${RED}[ERROR]${NC} $message" ;;
        DEBUG) [[ "${DEBUG:-false}" == "true" ]] && echo -e "${BLUE}[DEBUG]${NC} $message" ;;
    esac
    
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
    cat << 'EOF'
CineMatch Backup and Recovery Script v2.1.6

Usage:
    backup.sh <command> [options]

Commands:
    backup      Create a new backup
    restore     Restore from a backup
    list        List available backups
    verify      Verify backup integrity
    cleanup     Remove old backups

Options:
    --type, -t          Backup type: full, incremental, models, data, config (default: full)
    --destination, -d   Storage destination: local, s3, gcs, azure (default: local)
    --backup-id, -b     Backup ID for restore/verify operations
    --component, -c     Component to restore: all, models, data, config (default: all)
    --retention, -r     Retention period in days for cleanup (default: 30)
    --compress          Compression: gzip, bzip2, xz, none (default: gzip)
    --encrypt           Enable encryption (requires BACKUP_ENCRYPT_KEY)
    --dry-run           Show what would be done without making changes
    --verbose, -v       Enable verbose output
    --help, -h          Show this help message

Examples:
    # Create full backup to S3
    ./backup.sh backup --type full --destination s3

    # Restore models from specific backup
    ./backup.sh restore --backup-id 20240115-120000 --component models

    # List recent backups
    ./backup.sh list --destination local --limit 10

    # Cleanup old backups
    ./backup.sh cleanup --retention 30

Environment Variables:
    DATA_DIR            Path to data directory
    MODELS_DIR          Path to models directory
    BACKUP_LOCAL_DIR    Local backup storage path
    AWS_S3_BUCKET       S3 bucket name
    AWS_S3_PREFIX       S3 key prefix
    GCS_BUCKET          GCS bucket name
    AZURE_CONTAINER     Azure container name
    BACKUP_ENCRYPT_KEY  Encryption key for encrypted backups

EOF
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        error "Required command not found: $1"
        return 1
    fi
    return 0
}

generate_backup_id() {
    echo "$(date +%Y%m%d-%H%M%S)"
}

get_compression_ext() {
    case "$BACKUP_COMPRESSION" in
        gzip)  echo ".gz" ;;
        bzip2) echo ".bz2" ;;
        xz)    echo ".xz" ;;
        none)  echo "" ;;
        *)     echo ".gz" ;;
    esac
}

compress_file() {
    local input="$1"
    local output="${2:-$input$(get_compression_ext)}"
    
    case "$BACKUP_COMPRESSION" in
        gzip)  gzip -c "$input" > "$output" ;;
        bzip2) bzip2 -c "$input" > "$output" ;;
        xz)    xz -c "$input" > "$output" ;;
        none)  cp "$input" "$output" ;;
    esac
    
    echo "$output"
}

decompress_file() {
    local input="$1"
    local output="$2"
    
    case "${input##*.}" in
        gz)  gzip -dc "$input" > "$output" ;;
        bz2) bzip2 -dc "$input" > "$output" ;;
        xz)  xz -dc "$input" > "$output" ;;
        *)   cp "$input" "$output" ;;
    esac
}

encrypt_file() {
    local input="$1"
    local output="${2:-$input.enc}"
    
    if [[ "$BACKUP_ENCRYPT" == "true" ]] && [[ -n "$BACKUP_ENCRYPT_KEY" ]]; then
        openssl enc -aes-256-cbc -salt -pbkdf2 \
            -pass "pass:$BACKUP_ENCRYPT_KEY" \
            -in "$input" -out "$output"
        echo "$output"
    else
        echo "$input"
    fi
}

decrypt_file() {
    local input="$1"
    local output="$2"
    
    if [[ "${input##*.}" == "enc" ]] && [[ -n "$BACKUP_ENCRYPT_KEY" ]]; then
        openssl enc -d -aes-256-cbc -pbkdf2 \
            -pass "pass:$BACKUP_ENCRYPT_KEY" \
            -in "$input" -out "$output"
    else
        cp "$input" "$output"
    fi
}

calculate_checksum() {
    local file="$1"
    sha256sum "$file" | cut -d' ' -f1
}

# =============================================================================
# Backup Functions
# =============================================================================

create_backup_manifest() {
    local backup_id="$1"
    local backup_type="$2"
    local backup_path="$3"
    local files_list="$4"
    
    cat > "$backup_path/manifest.json" << EOF
{
    "backup_id": "$backup_id",
    "type": "$backup_type",
    "version": "$VERSION",
    "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "compression": "$BACKUP_COMPRESSION",
    "encrypted": $BACKUP_ENCRYPT,
    "source_host": "$(hostname)",
    "source_user": "${USER:-unknown}",
    "files": [
$(echo "$files_list" | sed 's/^/        "/;s/$/",/' | sed '$ s/,$//')
    ]
}
EOF
}

backup_models() {
    local backup_path="$1"
    local backup_id="$2"
    
    info "Backing up models..."
    
    local models_backup="$backup_path/models.tar"
    local files_backed_up=""
    
    if [[ -d "$MODELS_DIR" ]] && [[ -n "$(ls -A "$MODELS_DIR" 2>/dev/null)" ]]; then
        tar -cvf "$models_backup" -C "$(dirname "$MODELS_DIR")" "$(basename "$MODELS_DIR")"
        
        # Compress
        local compressed=$(compress_file "$models_backup")
        rm -f "$models_backup"
        
        # Encrypt if enabled
        local final=$(encrypt_file "$compressed")
        [[ "$final" != "$compressed" ]] && rm -f "$compressed"
        
        # Calculate checksum
        local checksum=$(calculate_checksum "$final")
        echo "$checksum  $(basename "$final")" >> "$backup_path/checksums.txt"
        
        files_backed_up="$(basename "$final")"
        info "Models backup created: $(basename "$final") ($(du -h "$final" | cut -f1))"
    else
        warn "No models found to backup"
    fi
    
    echo "$files_backed_up"
}

backup_data() {
    local backup_path="$1"
    local backup_id="$2"
    
    info "Backing up data..."
    
    local data_backup="$backup_path/data.tar"
    local files_backed_up=""
    
    if [[ -d "$DATA_DIR" ]] && [[ -n "$(ls -A "$DATA_DIR" 2>/dev/null)" ]]; then
        tar -cvf "$data_backup" -C "$(dirname "$DATA_DIR")" "$(basename "$DATA_DIR")"
        
        local compressed=$(compress_file "$data_backup")
        rm -f "$data_backup"
        
        local final=$(encrypt_file "$compressed")
        [[ "$final" != "$compressed" ]] && rm -f "$compressed"
        
        local checksum=$(calculate_checksum "$final")
        echo "$checksum  $(basename "$final")" >> "$backup_path/checksums.txt"
        
        files_backed_up="$(basename "$final")"
        info "Data backup created: $(basename "$final") ($(du -h "$final" | cut -f1))"
    else
        warn "No data found to backup"
    fi
    
    echo "$files_backed_up"
}

backup_config() {
    local backup_path="$1"
    local backup_id="$2"
    
    info "Backing up configuration..."
    
    local config_backup="$backup_path/config.tar"
    local files_backed_up=""
    
    # Backup config files
    local config_files=()
    [[ -d "$CONFIG_DIR" ]] && config_files+=("$CONFIG_DIR")
    [[ -f "$PROJECT_ROOT/.streamlit/config.toml" ]] && config_files+=("$PROJECT_ROOT/.streamlit")
    [[ -f "$PROJECT_ROOT/pyproject.toml" ]] && config_files+=("$PROJECT_ROOT/pyproject.toml")
    
    if [[ ${#config_files[@]} -gt 0 ]]; then
        tar -cvf "$config_backup" -C "$PROJECT_ROOT" "${config_files[@]/#$PROJECT_ROOT\//}"
        
        local compressed=$(compress_file "$config_backup")
        rm -f "$config_backup"
        
        local final=$(encrypt_file "$compressed")
        [[ "$final" != "$compressed" ]] && rm -f "$compressed"
        
        local checksum=$(calculate_checksum "$final")
        echo "$checksum  $(basename "$final")" >> "$backup_path/checksums.txt"
        
        files_backed_up="$(basename "$final")"
        info "Config backup created: $(basename "$final")"
    fi
    
    echo "$files_backed_up"
}

do_backup() {
    local backup_type="${BACKUP_TYPE:-full}"
    local destination="${DESTINATION:-local}"
    local dry_run="${DRY_RUN:-false}"
    
    local backup_id=$(generate_backup_id)
    local backup_path="$BACKUP_LOCAL_DIR/$backup_id"
    
    info "Starting $backup_type backup (ID: $backup_id)"
    
    if [[ "$dry_run" == "true" ]]; then
        info "[DRY-RUN] Would create backup at: $backup_path"
        return 0
    fi
    
    # Create backup directory
    mkdir -p "$backup_path"
    touch "$backup_path/checksums.txt"
    
    local all_files=""
    
    # Perform backup based on type
    case "$backup_type" in
        full)
            all_files+=$(backup_models "$backup_path" "$backup_id")$'\n'
            all_files+=$(backup_data "$backup_path" "$backup_id")$'\n'
            all_files+=$(backup_config "$backup_path" "$backup_id")
            ;;
        models)
            all_files=$(backup_models "$backup_path" "$backup_id")
            ;;
        data)
            all_files=$(backup_data "$backup_path" "$backup_id")
            ;;
        config)
            all_files=$(backup_config "$backup_path" "$backup_id")
            ;;
        *)
            error "Unknown backup type: $backup_type"
            exit 1
            ;;
    esac
    
    # Create manifest
    create_backup_manifest "$backup_id" "$backup_type" "$backup_path" "$all_files"
    
    # Upload to remote storage if needed
    case "$destination" in
        local)
            info "Backup stored locally at: $backup_path"
            ;;
        s3)
            upload_to_s3 "$backup_path" "$backup_id"
            ;;
        gcs)
            upload_to_gcs "$backup_path" "$backup_id"
            ;;
        azure)
            upload_to_azure "$backup_path" "$backup_id"
            ;;
    esac
    
    # Calculate total size
    local total_size=$(du -sh "$backup_path" | cut -f1)
    info "Backup completed successfully (Total size: $total_size)"
    info "Backup ID: $backup_id"
}

# =============================================================================
# Upload Functions
# =============================================================================

upload_to_s3() {
    local backup_path="$1"
    local backup_id="$2"
    
    check_command aws || return 1
    
    info "Uploading to S3: s3://$AWS_S3_BUCKET/$AWS_S3_PREFIX/$backup_id/"
    
    aws s3 cp "$backup_path" "s3://$AWS_S3_BUCKET/$AWS_S3_PREFIX/$backup_id/" \
        --recursive \
        --storage-class STANDARD_IA
    
    info "S3 upload completed"
}

upload_to_gcs() {
    local backup_path="$1"
    local backup_id="$2"
    
    check_command gsutil || return 1
    
    info "Uploading to GCS: gs://$GCS_BUCKET/$GCS_PREFIX/$backup_id/"
    
    gsutil -m cp -r "$backup_path/*" "gs://$GCS_BUCKET/$GCS_PREFIX/$backup_id/"
    
    info "GCS upload completed"
}

upload_to_azure() {
    local backup_path="$1"
    local backup_id="$2"
    
    check_command az || return 1
    
    info "Uploading to Azure Blob: $AZURE_CONTAINER/$AZURE_PREFIX/$backup_id/"
    
    az storage blob upload-batch \
        --destination "$AZURE_CONTAINER" \
        --destination-path "$AZURE_PREFIX/$backup_id" \
        --source "$backup_path"
    
    info "Azure upload completed"
}

# =============================================================================
# Restore Functions
# =============================================================================

do_restore() {
    local backup_id="${BACKUP_ID:-}"
    local destination="${DESTINATION:-local}"
    local component="${COMPONENT:-all}"
    local dry_run="${DRY_RUN:-false}"
    
    if [[ -z "$backup_id" ]]; then
        error "Backup ID is required for restore"
        exit 1
    fi
    
    info "Starting restore from backup: $backup_id"
    
    local backup_path="$BACKUP_LOCAL_DIR/$backup_id"
    
    # Download from remote if needed
    case "$destination" in
        s3)
            download_from_s3 "$backup_id"
            backup_path="$BACKUP_LOCAL_DIR/$backup_id"
            ;;
        gcs)
            download_from_gcs "$backup_id"
            backup_path="$BACKUP_LOCAL_DIR/$backup_id"
            ;;
        azure)
            download_from_azure "$backup_id"
            backup_path="$BACKUP_LOCAL_DIR/$backup_id"
            ;;
    esac
    
    if [[ ! -d "$backup_path" ]]; then
        error "Backup not found: $backup_path"
        exit 1
    fi
    
    # Verify checksums
    verify_backup_checksums "$backup_path" || {
        error "Checksum verification failed"
        exit 1
    }
    
    if [[ "$dry_run" == "true" ]]; then
        info "[DRY-RUN] Would restore from: $backup_path"
        return 0
    fi
    
    # Restore components
    case "$component" in
        all)
            restore_models "$backup_path"
            restore_data "$backup_path"
            restore_config "$backup_path"
            ;;
        models)
            restore_models "$backup_path"
            ;;
        data)
            restore_data "$backup_path"
            ;;
        config)
            restore_config "$backup_path"
            ;;
    esac
    
    info "Restore completed successfully"
}

restore_models() {
    local backup_path="$1"
    
    local archive=$(find "$backup_path" -name "models.tar*" | head -1)
    if [[ -z "$archive" ]]; then
        warn "No models backup found"
        return
    fi
    
    info "Restoring models..."
    
    local temp_tar=$(mktemp)
    
    # Decrypt if needed
    decrypt_file "$archive" "$temp_tar.dec"
    
    # Decompress
    decompress_file "$temp_tar.dec" "$temp_tar"
    rm -f "$temp_tar.dec"
    
    # Extract
    tar -xvf "$temp_tar" -C "$(dirname "$MODELS_DIR")"
    rm -f "$temp_tar"
    
    info "Models restored to: $MODELS_DIR"
}

restore_data() {
    local backup_path="$1"
    
    local archive=$(find "$backup_path" -name "data.tar*" | head -1)
    if [[ -z "$archive" ]]; then
        warn "No data backup found"
        return
    fi
    
    info "Restoring data..."
    
    local temp_tar=$(mktemp)
    
    decrypt_file "$archive" "$temp_tar.dec"
    decompress_file "$temp_tar.dec" "$temp_tar"
    rm -f "$temp_tar.dec"
    
    tar -xvf "$temp_tar" -C "$(dirname "$DATA_DIR")"
    rm -f "$temp_tar"
    
    info "Data restored to: $DATA_DIR"
}

restore_config() {
    local backup_path="$1"
    
    local archive=$(find "$backup_path" -name "config.tar*" | head -1)
    if [[ -z "$archive" ]]; then
        warn "No config backup found"
        return
    fi
    
    info "Restoring configuration..."
    
    local temp_tar=$(mktemp)
    
    decrypt_file "$archive" "$temp_tar.dec"
    decompress_file "$temp_tar.dec" "$temp_tar"
    rm -f "$temp_tar.dec"
    
    tar -xvf "$temp_tar" -C "$PROJECT_ROOT"
    rm -f "$temp_tar"
    
    info "Configuration restored"
}

# =============================================================================
# List and Verify Functions
# =============================================================================

do_list() {
    local destination="${DESTINATION:-local}"
    local limit="${LIMIT:-20}"
    
    info "Listing backups from $destination..."
    
    case "$destination" in
        local)
            if [[ -d "$BACKUP_LOCAL_DIR" ]]; then
                ls -la "$BACKUP_LOCAL_DIR" | tail -n "$limit"
            else
                warn "No local backups found"
            fi
            ;;
        s3)
            aws s3 ls "s3://$AWS_S3_BUCKET/$AWS_S3_PREFIX/" | tail -n "$limit"
            ;;
        gcs)
            gsutil ls "gs://$GCS_BUCKET/$GCS_PREFIX/" | tail -n "$limit"
            ;;
        azure)
            az storage blob list --container-name "$AZURE_CONTAINER" --prefix "$AZURE_PREFIX" | head -n "$limit"
            ;;
    esac
}

verify_backup_checksums() {
    local backup_path="$1"
    
    if [[ ! -f "$backup_path/checksums.txt" ]]; then
        warn "No checksums file found"
        return 0
    fi
    
    info "Verifying checksums..."
    
    cd "$backup_path"
    if sha256sum -c checksums.txt; then
        info "All checksums verified"
        return 0
    else
        error "Checksum verification failed"
        return 1
    fi
}

do_verify() {
    local backup_id="${BACKUP_ID:-}"
    
    if [[ -z "$backup_id" ]]; then
        error "Backup ID is required for verification"
        exit 1
    fi
    
    local backup_path="$BACKUP_LOCAL_DIR/$backup_id"
    
    if [[ ! -d "$backup_path" ]]; then
        error "Backup not found: $backup_path"
        exit 1
    fi
    
    verify_backup_checksums "$backup_path"
}

# =============================================================================
# Cleanup Function
# =============================================================================

do_cleanup() {
    local destination="${DESTINATION:-local}"
    local retention="${RETENTION:-$BACKUP_RETENTION_DAYS}"
    local dry_run="${DRY_RUN:-false}"
    
    info "Cleaning up backups older than $retention days..."
    
    case "$destination" in
        local)
            if [[ -d "$BACKUP_LOCAL_DIR" ]]; then
                local old_backups=$(find "$BACKUP_LOCAL_DIR" -maxdepth 1 -type d -mtime +$retention)
                
                if [[ -n "$old_backups" ]]; then
                    echo "$old_backups" | while read dir; do
                        if [[ "$dry_run" == "true" ]]; then
                            info "[DRY-RUN] Would remove: $dir"
                        else
                            info "Removing: $dir"
                            rm -rf "$dir"
                        fi
                    done
                else
                    info "No old backups to clean up"
                fi
            fi
            ;;
        s3)
            # S3 lifecycle rules should handle this
            info "Use S3 lifecycle rules for automatic cleanup"
            ;;
    esac
    
    info "Cleanup completed"
}

# =============================================================================
# Main
# =============================================================================

main() {
    local command="${1:-}"
    shift || true
    
    # Parse global options
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --type|-t)
                BACKUP_TYPE="$2"
                shift 2
                ;;
            --destination|-d)
                DESTINATION="$2"
                shift 2
                ;;
            --backup-id|-b)
                BACKUP_ID="$2"
                shift 2
                ;;
            --component|-c)
                COMPONENT="$2"
                shift 2
                ;;
            --retention|-r)
                RETENTION="$2"
                shift 2
                ;;
            --compress)
                BACKUP_COMPRESSION="$2"
                shift 2
                ;;
            --encrypt)
                BACKUP_ENCRYPT=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --verbose|-v)
                DEBUG=true
                shift
                ;;
            --limit)
                LIMIT="$2"
                shift 2
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
    
    # Execute command
    case "$command" in
        backup)
            do_backup
            ;;
        restore)
            do_restore
            ;;
        list)
            do_list
            ;;
        verify)
            do_verify
            ;;
        cleanup)
            do_cleanup
            ;;
        ""|--help|-h)
            show_help
            exit 0
            ;;
        *)
            error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
