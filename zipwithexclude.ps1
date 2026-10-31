
$ErrorActionPreference = "Stop"

# 1. Source and destination
$source = "C:\sangeeta\git\applied_agentic_ai_program_for_swes\class_03_rag_powered_knowledge_agents_part_2"
$destination = "C:\sangeeta\git\applied_agentic_ai_program_for_swes\RAG_Class03.zip"

# 2. Exclude folders
$excludeFolders = @(
    ".venv",
    "venv",
    "node_modules",
    ".git",
    "__pycache__",
    ".pytest_cache",
    "chroma_db",
    "chroma_data",
    "db",
    "data_db"
)

# 3. Exclude files
$excludeFiles = @(
    ".env",
    ".env.*",
    "*.db",
    "*.sqlite",
    "*.sqlite3",
    "*.pyc",
    "*.log",
    "*.pem",
    "*.key",
    "*credentials*.json",
    "*secret*.json"
)

# 4. Load compression libraries
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

# 5. Validate source
if (-not (Test-Path -LiteralPath $source -PathType Container)) {
    throw "Source folder not found: $source"
}

# 6. Select files, preserving subfolder structure
$files = @(
    Get-ChildItem -LiteralPath $source -Recurse -File -Force |
    Where-Object {
        $relative = $_.FullName.Substring($source.Length).TrimStart('\')
        $parts = $relative -split '\\'

        $excludedFolder = $false

        # Check all directory components (not filename)
        if ($parts.Length -gt 1) {
            foreach ($folder in $parts[0..($parts.Length - 2)]) {
                if ($folder -in $excludeFolders) {
                    $excludedFolder = $true
                    break
                }
            }
        }

        $excludedFile = $false
        foreach ($pattern in $excludeFiles) {
            if ($_.Name -like $pattern) {
                $excludedFile = $true
                break
            }
        }

        -not $excludedFolder -and -not $excludedFile
    }
)

if ($files.Count -eq 0) {
    throw "No files found to ZIP after exclusions."
}

# 7. Create ZIP
$zip = $null

try {
    if (Test-Path -LiteralPath $destination) {
        Remove-Item -LiteralPath $destination -Force
    }

    $zip = [System.IO.Compression.ZipFile]::Open(
        $destination,
        [System.IO.Compression.ZipArchiveMode]::Create
    )

    foreach ($file in $files) {
        $relative = $file.FullName.Substring($source.Length).TrimStart('\')
        $entryName = "class_03_rag_powered_knowledge_agents_part_2/" +
                     ($relative -replace '\\', '/')

        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
            $zip,
            $file.FullName,
            $entryName,
            [System.IO.Compression.CompressionLevel]::Optimal
        ) | Out-Null
    }
}
finally {
    if ($null -ne $zip) {
        $zip.Dispose()
    }
}

# 8. Verify ZIP
$verify = [System.IO.Compression.ZipFile]::OpenRead($destination)

try {
    $count = $verify.Entries.Count
}
finally {
    $verify.Dispose()
}

if ($count -ne $files.Count) {
    throw "ZIP verification failed. Expected $($files.Count), found $count."
}

Write-Host ""
Write-Host "ZIP created successfully!" -ForegroundColor Green
Write-Host "Location: $destination"
Write-Host "Files included: $count"
Write-Host "ZIP size: $([math]::Round((Get-Item $destination).Length / 1MB, 2)) MB"
