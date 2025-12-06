# =============================================================================
# CineMatch V2.1.6 - Automated Rollback Script (PowerShell)
# Task 4.8: Automated rollback procedures
# =============================================================================
# This script provides automated rollback capabilities for CineMatch deployments.
#
# Usage:
#   .\rollback.ps1 [-Environment <env>] [-Revision <rev>] [-Component <comp>] [-DryRun] [-Force]
#
# Examples:
#   .\rollback.ps1 -Environment production
#   .\rollback.ps1 -Environment staging -Revision 5
#   .\rollback.ps1 -Environment prod -Component api -DryRun
# =============================================================================

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('dev', 'development', 'staging', 'stage', 'prod', 'production')]
    [string]$Environment,
    
    [Parameter(Mandatory=$false)]
    [int]$Revision = 0,
    
    [Parameter(Mandatory=$false)]
    [ValidateSet('ui', 'api', 'all')]
    [string]$Component = 'all',
    
    [Parameter(Mandatory=$false)]
    [switch]$DryRun,
    
    [Parameter(Mandatory=$false)]
    [switch]$Force
)

# =============================================================================
# Configuration
# =============================================================================

$ErrorActionPreference = "Stop"
$ProjectName = "cinematch"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir = Join-Path $ScriptDir "logs"
$LogFile = Join-Path $LogDir "rollback-$(Get-Date -Format 'yyyyMMdd-HHmmss').log"

# Ensure log directory exists
if (!(Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

# =============================================================================
# Logging Functions
# =============================================================================

function Write-Log {
    param(
        [Parameter(Mandatory=$true)]
        [ValidateSet('INFO', 'WARN', 'ERROR', 'DEBUG')]
        [string]$Level,
        
        [Parameter(Mandatory=$true)]
        [string]$Message
    )
    
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $LogMessage = "[$Timestamp] [$Level] $Message"
    
    # Console output with colors
    switch ($Level) {
        'INFO'  { Write-Host "[INFO] $Message" -ForegroundColor Green }
        'WARN'  { Write-Host "[WARN] $Message" -ForegroundColor Yellow }
        'ERROR' { Write-Host "[ERROR] $Message" -ForegroundColor Red }
        'DEBUG' { Write-Host "[DEBUG] $Message" -ForegroundColor Cyan }
    }
    
    # File output
    Add-Content -Path $LogFile -Value $LogMessage
}

function Write-Info  { param($Message) Write-Log -Level INFO -Message $Message }
function Write-Warn  { param($Message) Write-Log -Level WARN -Message $Message }
function Write-Err   { param($Message) Write-Log -Level ERROR -Message $Message }
function Write-Debug { param($Message) Write-Log -Level DEBUG -Message $Message }

# =============================================================================
# Helper Functions
# =============================================================================

function Test-Prerequisites {
    Write-Info "Checking prerequisites..."
    
    $Missing = @()
    
    # Check required commands
    @('kubectl', 'helm') | ForEach-Object {
        if (!(Get-Command $_ -ErrorAction SilentlyContinue)) {
            $Missing += $_
        }
    }
    
    if ($Missing.Count -gt 0) {
        Write-Err "Missing required commands: $($Missing -join ', ')"
        exit 1
    }
    
    # Check kubectl connection
    try {
        kubectl cluster-info 2>&1 | Out-Null
    } catch {
        Write-Err "Cannot connect to Kubernetes cluster"
        exit 1
    }
    
    Write-Info "All prerequisites satisfied"
}

function Get-Namespace {
    switch -Regex ($Environment) {
        'dev|development' { return "$ProjectName-dev" }
        'staging|stage'   { return "$ProjectName-staging" }
        'prod|production' { return "$ProjectName-prod" }
        default {
            Write-Err "Unknown environment: $Environment"
            exit 1
        }
    }
}

function Get-ReleaseName {
    return "$ProjectName-$Environment"
}

function Confirm-Action {
    param([string]$Prompt = "Are you sure?")
    
    if ($Force) { return $true }
    
    $Response = Read-Host "$Prompt [y/N]"
    return ($Response -match '^[Yy]')
}

# =============================================================================
# Rollback Functions
# =============================================================================

function Get-HelmRevisions {
    $Namespace = Get-Namespace
    $Release = Get-ReleaseName
    
    Write-Info "Listing Helm revisions for $Release in $Namespace..."
    
    try {
        helm history $Release -n $Namespace --max 10
    } catch {
        Write-Warn "No Helm release found, checking Kubernetes deployments..."
        kubectl rollout history deployment -n $Namespace -l app.kubernetes.io/name=$ProjectName
    }
}

function Get-CurrentRevision {
    $Namespace = Get-Namespace
    $Release = Get-ReleaseName
    
    try {
        $Status = helm status $Release -n $Namespace -o json | ConvertFrom-Json
        return $Status.version
    } catch {
        return 0
    }
}

function Get-PreviousRevision {
    $Current = Get-CurrentRevision
    return [Math]::Max(0, $Current - 1)
}

function Invoke-HelmRollback {
    $Namespace = Get-Namespace
    $Release = Get-ReleaseName
    $TargetRevision = if ($Revision -gt 0) { $Revision } else { Get-PreviousRevision }
    
    Write-Info "Rolling back Helm release $Release to revision $TargetRevision..."
    
    if ($DryRun) {
        Write-Info "[DRY-RUN] Would execute: helm rollback $Release $TargetRevision -n $Namespace"
        helm rollback $Release $TargetRevision -n $Namespace --dry-run
        return $true
    }
    
    # Create rollback record
    $RollbackRecord = @{
        timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
        environment = $Environment
        release = $Release
        from_revision = Get-CurrentRevision
        to_revision = $TargetRevision
        component = $Component
        initiated_by = $env:USERNAME
    }
    
    $RecordPath = Join-Path $LogDir "rollback-record-$(Get-Date -Format 'yyyyMMdd-HHmmss').json"
    $RollbackRecord | ConvertTo-Json | Out-File -FilePath $RecordPath
    
    # Execute rollback
    try {
        helm rollback $Release $TargetRevision -n $Namespace --wait --timeout 10m
        Write-Info "Helm rollback completed successfully"
        return $true
    } catch {
        Write-Err "Helm rollback failed: $_"
        return $false
    }
}

function Invoke-DeploymentRollback {
    param(
        [string]$Deployment,
        [string]$Ns = $null
    )
    
    $Namespace = if ($Ns) { $Ns } else { Get-Namespace }
    $TargetRevision = if ($Revision -gt 0) { $Revision } else { 0 }
    
    Write-Info "Rolling back deployment $Deployment..."
    
    if ($DryRun) {
        if ($TargetRevision -eq 0) {
            Write-Info "[DRY-RUN] Would execute: kubectl rollout undo deployment/$Deployment -n $Namespace"
        } else {
            Write-Info "[DRY-RUN] Would execute: kubectl rollout undo deployment/$Deployment -n $Namespace --to-revision=$TargetRevision"
        }
        return $true
    }
    
    try {
        if ($TargetRevision -eq 0) {
            kubectl rollout undo deployment/$Deployment -n $Namespace
        } else {
            kubectl rollout undo deployment/$Deployment -n $Namespace --to-revision=$TargetRevision
        }
        
        # Wait for rollout
        kubectl rollout status deployment/$Deployment -n $Namespace --timeout=5m
        return $true
    } catch {
        Write-Err "Deployment rollback failed: $_"
        return $false
    }
}

function Invoke-ComponentRollback {
    $Namespace = Get-Namespace
    
    switch ($Component) {
        'ui' {
            return Invoke-DeploymentRollback -Deployment "$ProjectName-ui" -Ns $Namespace
        }
        'api' {
            return Invoke-DeploymentRollback -Deployment "$ProjectName-api" -Ns $Namespace
        }
        'all' {
            Write-Info "Rolling back all components..."
            return Invoke-HelmRollback
        }
        default {
            Write-Err "Unknown component: $Component"
            exit 1
        }
    }
}

# =============================================================================
# Verification Functions
# =============================================================================

function Test-RollbackSuccess {
    $Namespace = Get-Namespace
    
    Write-Info "Verifying rollback..."
    
    # Check pod status
    $Pods = kubectl get pods -n $Namespace -l app.kubernetes.io/name=$ProjectName -o json | ConvertFrom-Json
    
    $UnhealthyPods = $Pods.items | Where-Object { $_.status.phase -ne "Running" }
    if ($UnhealthyPods) {
        Write-Warn "Some pods are not running: $($UnhealthyPods.metadata.name -join ', ')"
        return $false
    }
    
    # Check readiness
    $NotReady = $Pods.items | Where-Object {
        $ReadyCondition = $_.status.conditions | Where-Object { $_.type -eq "Ready" }
        $ReadyCondition.status -ne "True"
    }
    
    if ($NotReady) {
        Write-Warn "Some pods are not ready: $($NotReady.metadata.name -join ', ')"
        return $false
    }
    
    Write-Info "Rollback verification completed successfully"
    return $true
}

# =============================================================================
# Notification Functions
# =============================================================================

function Send-Notification {
    param(
        [string]$Status,
        [string]$Message
    )
    
    # Slack notification (if webhook configured)
    $SlackWebhook = $env:SLACK_WEBHOOK_URL
    if ($SlackWebhook) {
        $Color = switch ($Status) {
            'success' { 'good' }
            'warning' { 'warning' }
            'failure' { 'danger' }
            default   { '#808080' }
        }
        
        $Payload = @{
            attachments = @(@{
                color = $Color
                title = "CineMatch Rollback - $Environment"
                text = $Message
                footer = "Initiated by $env:USERNAME"
                ts = [int](Get-Date -UFormat %s)
            })
        }
        
        try {
            Invoke-RestMethod -Uri $SlackWebhook -Method Post -Body ($Payload | ConvertTo-Json -Depth 10) -ContentType 'application/json'
        } catch {
            Write-Debug "Failed to send Slack notification: $_"
        }
    }
    
    Write-Debug "Notification sent: $Status - $Message"
}

# =============================================================================
# Main
# =============================================================================

function Main {
    Write-Info "Starting CineMatch rollback process..."
    Write-Info "Environment: $Environment"
    Write-Info "Component: $Component"
    Write-Info "Revision: $(if ($Revision -gt 0) { $Revision } else { 'previous' })"
    Write-Info "Dry run: $DryRun"
    
    # Check prerequisites
    Test-Prerequisites
    
    # List current revisions
    Get-HelmRevisions
    
    # Confirm rollback
    Write-Host ""
    if (!(Confirm-Action "Do you want to proceed with the rollback?")) {
        Write-Info "Rollback cancelled"
        exit 0
    }
    
    # Execute rollback
    Write-Info "Executing rollback..."
    Send-Notification -Status "warning" -Message "Rollback initiated for $Component component(s)"
    
    $RollbackSuccess = Invoke-ComponentRollback
    
    if ($RollbackSuccess) {
        Write-Info "Rollback executed successfully"
        
        # Verify rollback
        if (Test-RollbackSuccess) {
            Write-Info "Rollback verified successfully"
            Send-Notification -Status "success" -Message "Rollback completed successfully"
        } else {
            Write-Warn "Rollback verification had warnings"
            Send-Notification -Status "warning" -Message "Rollback completed with warnings"
        }
    } else {
        Write-Err "Rollback failed"
        Send-Notification -Status "failure" -Message "Rollback failed - manual intervention may be required"
        exit 1
    }
    
    Write-Info "Rollback process completed"
    Write-Info "Log file: $LogFile"
}

# Run main
Main
