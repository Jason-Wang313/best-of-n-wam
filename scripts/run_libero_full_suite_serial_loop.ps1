param(
    [string]$Python = "",
    [string]$LiberoSource = "",
    [string]$LiberoConfig = "",
    [int]$MaxAttempts = 40,
    [int]$TasksPerAttempt = 1,
    [double]$MaxHours = 0,
    [switch]$IncludeLibero90,
    [double]$MinAvailableRamGb = 1.5,
    [double]$MinDiskFreeGb = 2.0,
    [double]$SleepBetweenTasks = 5.0,
    [int]$WaitForPreflightSeconds = 3600,
    [int]$PreflightPollSeconds = 60,
    [int]$AttemptPauseSeconds = 300
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Runner = Join-Path $PSScriptRoot "run_libero_full_suite_serial.ps1"
$StatusPath = Join-Path $Repo "results\libero_full_suite_serial\status_summary.json"
$StartedAt = Get-Date

function New-CommonParams {
    $params = @{
        MinAvailableRamGb = $MinAvailableRamGb
        MinDiskFreeGb = $MinDiskFreeGb
        SleepBetweenTasks = $SleepBetweenTasks
    }
    if ($Python) {
        $params.Python = $Python
    }
    if ($LiberoSource) {
        $params.LiberoSource = $LiberoSource
    }
    if ($LiberoConfig) {
        $params.LiberoConfig = $LiberoConfig
    }
    if ($IncludeLibero90) {
        $params.IncludeLibero90 = $true
    }
    return $params
}

function Read-LiberoSummary {
    if (-not (Test-Path -LiteralPath $StatusPath)) {
        return $null
    }
    return Get-Content -LiteralPath $StatusPath -Raw | ConvertFrom-Json
}

function Get-CompletedCount {
    param($Summary)
    if ($null -eq $Summary -or $null -eq $Summary.completed_task_count) {
        return -1
    }
    return [int]$Summary.completed_task_count
}

function Write-ProgressLine {
    param($Prefix, $Summary)
    if ($null -eq $Summary) {
        Write-Host "$Prefix no summary yet"
        return
    }
    $completed = Get-CompletedCount $Summary
    $total = if ($null -ne $Summary.task_count) { $Summary.task_count } else { "?" }
    $ram = "unknown"
    if ($null -ne $Summary.preflight -and $null -ne $Summary.preflight.memory_available_gb) {
        $ram = "{0:N2}GB" -f [double]$Summary.preflight.memory_available_gb
    }
    $next = "none"
    if ($null -ne $Summary.next_pending_task -and $null -ne $Summary.next_pending_task.task_key) {
        $next = $Summary.next_pending_task.task_key
    }
    Write-Host "$Prefix complete=$($Summary.complete) verified=$($Summary.verified) tasks=$completed/$total ram=$ram next=$next"
}

function Invoke-Status {
    $commonParams = New-CommonParams
    $commonParams.Status = $true
    & $Runner @commonParams
    if ($LASTEXITCODE -ne 0) {
        throw "status command failed with exit code $LASTEXITCODE"
    }
    return Read-LiberoSummary
}

if ($TasksPerAttempt -lt 1) {
    throw "TasksPerAttempt must be at least 1"
}
if ($MaxAttempts -lt 0) {
    throw "MaxAttempts must be nonnegative"
}

$summary = Invoke-Status
Write-ProgressLine "[loop] initial" $summary

if ($MaxAttempts -eq 0) {
    exit 0
}

for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
    if ($MaxHours -gt 0 -and ((Get-Date) - $StartedAt).TotalHours -ge $MaxHours) {
        Write-Host "[loop] reached MaxHours=$MaxHours; stopping after status checkpoint"
        break
    }

    if ($null -ne $summary -and $summary.complete -eq $true) {
        Write-Host "[loop] benchmark already complete"
        break
    }

    $before = Get-CompletedCount $summary
    Write-Host "[loop] attempt $attempt/${MaxAttempts}: running up to $TasksPerAttempt task(s)"
    $runParams = New-CommonParams
    $runParams.StopAfterTasks = $TasksPerAttempt
    $runParams.WaitForPreflightSeconds = $WaitForPreflightSeconds
    $runParams.PreflightPollSeconds = $PreflightPollSeconds
    & $Runner @runParams
    if ($LASTEXITCODE -ne 0) {
        throw "runner failed with exit code $LASTEXITCODE"
    }

    $summary = Invoke-Status
    Write-ProgressLine "[loop] after attempt $attempt" $summary
    $after = Get-CompletedCount $summary

    if ($null -ne $summary -and $summary.complete -eq $true) {
        Write-Host "[loop] benchmark complete"
        break
    }

    if ($after -le $before) {
        Write-Host "[loop] no new task completed; pausing $AttemptPauseSeconds seconds before retry"
        Start-Sleep -Seconds $AttemptPauseSeconds
    }
}

$summary = Invoke-Status
Write-ProgressLine "[loop] final" $summary
