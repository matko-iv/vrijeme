$ErrorActionPreference = 'Stop'
$stageRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$targetRoot = [IO.Path]::GetFullPath('C:\Users\Matija\Documents\GitHub\rpn-web')
$baselineRoot = Join-Path $stageRoot 'baseline'
$files = @(
 'frontend/src/lib/game/map-scale.ts',
 'frontend/src/lib/game/organic-terrain.ts',
 'frontend/src/lib/game/organic-cities.ts',
 'frontend/src/lib/game/city-surface.ts',
 'frontend/src/lib/game/settlement-layout.ts',
 'frontend/src/lib/game/map-renderer.ts',
 'frontend/src/lib/game/traffic-layer.ts',
 'frontend/src/lib/game/checkpoint-layer.ts',
 'frontend/src/lib/game/transport.ts',
 'frontend/src/lib/components/map/HexMap.svelte',
 'scripts/prepare-organic-terrain.py',
 'scripts/prepare-organic-details.py',
 'scripts/test-organic-map.cjs'
)
$assetRoot = Join-Path $stageRoot 'frontend/static/tiles/organic'
foreach ($file in Get-ChildItem -LiteralPath $assetRoot -File -Recurse) {
 $files += 'frontend/static/tiles/organic/' + $file.FullName.Substring($assetRoot.Length + 1).Replace('\','/')
}
foreach ($relative in $files) {
 $sourcePath = [IO.Path]::GetFullPath((Join-Path $stageRoot $relative))
 $targetPath = [IO.Path]::GetFullPath((Join-Path $targetRoot $relative))
 if (-not $sourcePath.StartsWith($stageRoot + '\') -or -not $targetPath.StartsWith($targetRoot + '\')) { throw 'Path outside implementation scope' }
 if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) { throw "Missing source: $relative" }
 $baselinePath = Join-Path $baselineRoot $relative
 if ((Test-Path -LiteralPath $targetPath) -and -not (Test-Path -LiteralPath $baselinePath)) {
  New-Item -ItemType Directory -Path (Split-Path $baselinePath) -Force | Out-Null
  Copy-Item -LiteralPath $targetPath -Destination $baselinePath
 }
 New-Item -ItemType Directory -Path (Split-Path $targetPath) -Force | Out-Null
 Copy-Item -LiteralPath $sourcePath -Destination $targetPath -Force
 Write-Output $relative
}
