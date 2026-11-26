#Requires -Version 5.1
param([switch]$CheckOnly, [switch]$Prepare, [switch]$Install, [switch]$Elevated)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $PSScriptRoot 'venv\Scripts\python.exe'
$driverRoot = Join-Path $PSScriptRoot 'data\drivers\vb-cable'
$packageRoot = Join-Path $driverRoot 'package'
$archivePath = Join-Path $driverRoot 'VBCABLE_Driver_Pack46.zip'
$archiveUrl = 'https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack46.zip'
$archiveHash = 'BC44E8974191A5A2427D6C979E2DAB08DBB09925F3F435DFA2F0BA3E45B6EB90'

function Get-PhoneCamState {
    if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Installez PhoneCam avec Installer.bat avant les pilotes.' }
    $probe = @'
import json, shutil
import sounddevice as sd
import cv2
from cv2_enumerate_cameras import enumerate_cameras
names=[d['name'] for d in sd.query_devices()]
cameras=[c.name for c in enumerate_cameras(cv2.CAP_DSHOW)]
print(json.dumps({'cable':any(n.startswith(('CABLE Input','CABLE-A Input','CABLE-B Input','CABLE-C Input','CABLE-D Input')) for n in names),'obs':'OBS Virtual Camera' in cameras,'ffmpeg':bool(shutil.which('ffmpeg'))}))
'@
    $json = & $pythonPath -c $probe
    if ($LASTEXITCODE -ne 0) { throw 'Verification des peripheriques impossible.' }
    return ($json | ConvertFrom-Json)
}

function Prepare-VBCable {
    New-Item -ItemType Directory -Path $driverRoot -Force | Out-Null
    if (-not (Test-Path -LiteralPath $archivePath) -or (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash -ne $archiveHash) {
        Write-Host 'Telechargement de VB-CABLE depuis le site officiel...'
        Invoke-WebRequest -UseBasicParsing -Uri $archiveUrl -OutFile ($archivePath + '.part')
        if ((Get-FileHash -LiteralPath ($archivePath + '.part') -Algorithm SHA256).Hash -ne $archiveHash) {
            throw 'Le paquet VB-CABLE a change : empreinte SHA-256 inattendue. Installation arretee.'
        }
        Move-Item -LiteralPath ($archivePath + '.part') -Destination $archivePath -Force
    }
    # Le paquet officiel reste intact ; seuls ses fichiers de donnees sont extraits.
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [IO.Compression.ZipFile]::OpenRead($archivePath)
    try {
        $base = [IO.Path]::GetFullPath($packageRoot).TrimEnd('\') + '\'
        foreach ($entry in $archive.Entries) {
            $target = [IO.Path]::GetFullPath((Join-Path $packageRoot $entry.FullName))
            if (-not $target.StartsWith($base, [StringComparison]::OrdinalIgnoreCase)) { throw 'Chemin invalide dans le paquet.' }
        }
    } finally { $archive.Dispose() }
    Expand-Archive -LiteralPath $archivePath -DestinationPath $packageRoot -Force
    $setup = Join-Path $packageRoot $(if ([Environment]::Is64BitOperatingSystem) { 'VBCABLE_Setup_x64.exe' } else { 'VBCABLE_Setup.exe' })
    $signature = Get-AuthenticodeSignature -LiteralPath $setup
    if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'BUREL|VB-AUDIO') {
        throw 'Signature officielle de VB-CABLE non valide. Aucun programme lance.'
    }
    return $setup
}

try {
    $state = Get-PhoneCamState
    if ($CheckOnly) { $state | ConvertTo-Json -Compress; exit 0 }
    Write-Host ('Camera virtuelle OBS : ' + $state.obs)
    Write-Host ('Micro virtuel VB-CABLE : ' + $state.cable)
    Write-Host ('FFmpeg : ' + $state.ffmpeg)
    if ($Prepare) {
        if (-not $state.cable) { $setup = Prepare-VBCable; Write-Host ('Installateur officiel verifie : ' + $setup) }
        exit 0
    }
    if (-not $Install) { exit 0 }
    if ($state.cable -and $state.obs -and $state.ffmpeg) { Write-Host 'Tous les prerequis sont deja presents.'; exit 0 }
    $isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    if (-not $isAdmin) {
        if ($Elevated) { throw 'Droits administrateur non accordes.' }
        Write-Host 'Windows va demander une autorisation administrateur pour les pilotes.'
        $child = Start-Process -FilePath 'powershell.exe' -Verb RunAs -Wait -PassThru -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'RemoteSigned', '-File', ('"{0}"' -f $PSCommandPath), '-Install', '-Elevated')
        exit $child.ExitCode
    }
    if (-not $state.obs) {
        & winget install --exact --id OBSProject.OBSStudio --source winget --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -ne 0) { throw 'Installation OBS non terminee.' }
    }
    if (-not $state.ffmpeg) {
        & winget install --exact --id Gyan.FFmpeg --source winget --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -ne 0) { throw 'Installation FFmpeg non terminee.' }
    }
    if (-not $state.cable) {
        $setup = Prepare-VBCable
        # Installation personnelle via le programme signe du fabricant. Aucun ajout
        # de certificat, aucune desactivation des protections Windows, aucun reboot force.
        Write-Host 'Installation de VB-CABLE...'
        $process = Start-Process -FilePath $setup -ArgumentList @('-i', '-h') -WorkingDirectory $packageRoot -Wait -PassThru -WindowStyle Hidden
        if ($process.ExitCode -notin @(0, 3010)) { throw ('VB-CABLE a retourne le code ' + $process.ExitCode) }
        Write-Host 'Installateur termine. Redemarrez Windows pour finaliser le pilote audio.'
    }
    $after = Get-PhoneCamState
    $after | ConvertTo-Json -Compress
    if (-not $after.cable) { Write-Host 'VB-CABLE pas encore visible : verification necessaire apres redemarrage.'; exit 3010 }
    Write-Host 'Dans PhoneCam : CABLE Input. Dans Discord / Snapchat : CABLE Output.'
    exit 0
} catch {
    Write-Host ('Installation non terminee : ' + $_.Exception.Message) -ForegroundColor Red
    exit 1
}
