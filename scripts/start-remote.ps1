#Requires -Version 5.1
<#
.SYNOPSIS
    Tailscale 다기기 모드로 AI Debate Room 서버를 띄웁니다.

.DESCRIPTION
    손으로 칠 때 빠뜨리기 쉬운 것 둘을 대신합니다.

      1. DEBATE_CONFIG_UI=off
         설정 화면은 .env 를 쓰고 API 키를 다룹니다. 다기기 구성에서 이걸
         끄는 게 차단의 **보장**입니다(루프백 클라이언트 검사는 끄는 걸
         잊었을 때의 두 번째 방어선이지, 기대야 할 장치가 아닙니다).

      2. 리포 루트에서 실행
         pricing_path / db_path / env_path 가 전부 **상대 경로**라, 다른
         폴더에서 띄우면 빈 .env 와 새 DB 를 조용히 만듭니다. 키가 안 읽히는
         채로 서버가 떠서, 원인을 찾기 어려운 쪽으로 실패합니다.

    바인드 주소는 127.0.0.1 그대로 둡니다. `tailscale serve` 가 거기로
    프록시하므로 바꿀 이유가 없고, 바꾸면 소켓이 LAN 에도 열립니다.

.EXAMPLE
    .\scripts\start-remote.ps1
    .\scripts\start-remote.ps1 -Port 8001
#>
[CmdletBinding()]
param(
    [ValidateRange(1, 65535)]
    [int] $Port = 8000
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot

function Restore-EnvVar {
    param([string] $Name, [string] $Value)
    if ([string]::IsNullOrEmpty($Value)) {
        Remove-Item "Env:$Name" -ErrorAction SilentlyContinue
    } else {
        Set-Item "Env:$Name" $Value
    }
}

# 포트가 이미 쓰이고 있으면 uvicorn 은 'Application startup complete' 를 찍은
# **뒤에** bind 오류를 냅니다. 성공한 것처럼 보이므로 먼저 확인하고 범인을
# 지목합니다. (tailscale serve 는 이 포트를 점유하지 않습니다 — tailnet 의
# 443 에서 듣고 127.0.0.1:<Port> 로 넘길 뿐입니다.)
if (Get-Command Get-NetTCPConnection -ErrorAction SilentlyContinue) {
    $busy = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($busy) {
        Write-Host "포트 $Port 이 이미 쓰이고 있습니다:" -ForegroundColor Red
        foreach ($conn in $busy) {
            $proc = Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
            if ($proc) {
                Write-Host ("  PID {0}  {1}  {2}" -f $proc.Id, $proc.ProcessName, $proc.Path)
            } else {
                Write-Host ("  PID {0}  (프로세스 정보 없음)" -f $conn.OwningProcess)
            }
        }
        Write-Host ""
        Write-Host "그 프로세스를 끄거나(Stop-Process -Id <PID>), -Port 로 다른 포트를 주십시오."
        Write-Host "다른 포트를 쓰면 tailscale serve 도 같은 포트로 다시 띄워야 합니다."
        exit 1
    }
}

$prevConfigUi = $env:DEBATE_CONFIG_UI
$prevPort     = $env:DEBATE_PORT
try {
    $env:DEBATE_CONFIG_UI = 'off'
    $env:DEBATE_PORT      = "$Port"

    Push-Location $repoRoot
    try {
        Write-Host "repo    : $repoRoot"
        Write-Host "config  : DEBATE_CONFIG_UI=off  (설정 화면은 404)"
        Write-Host "tailnet : 터널이 아직 없다면 별도 창에서  tailscale serve --bg $Port"
        Write-Host ""
        python -m debate.cli serve
    } finally {
        Pop-Location
    }
} finally {
    # 호출한 세션에 off 가 남으면, 나중에 로컬에서 설정 화면을 열 때 404 가
    # 나는 이유를 찾기 어렵습니다. 스크립트가 끝나면 원래대로 돌립니다.
    Restore-EnvVar -Name 'DEBATE_CONFIG_UI' -Value $prevConfigUi
    Restore-EnvVar -Name 'DEBATE_PORT'      -Value $prevPort
}
