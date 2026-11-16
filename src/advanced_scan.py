"""Scan all pages of Vinamilk AR 2022 PDF and test Windows.Data.Pdf PNG rendering."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image
from pypdf import PdfReader


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def render_pages_winrt(pdf_path: Path, output_dir: Path, pages_1based: list[int], dpi_scale: float = 1.5) -> list[Path]:
    """Render specific 1-based PDF pages to PNG using built-in Windows.Data.Pdf WinRT API."""
    output_dir.mkdir(parents=True, exist_ok=True)
    abs_pdf = str(pdf_path.resolve())
    abs_out = str(output_dir.resolve())
    page_list_str = ",".join(str(p) for p in pages_1based)

    ps_script = f"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType=WindowsRuntime]
$null = [Windows.Data.Pdf.PdfDocument, Windows.Data.Pdf, ContentType=WindowsRuntime]
$null = [Windows.Storage.Streams.InMemoryRandomAccessStream, Windows.Storage.Streams, ContentType=WindowsRuntime]
$null = [Windows.Data.Pdf.PdfPageRenderOptions, Windows.Data.Pdf, ContentType=WindowsRuntime]

$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {{
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
}})[0]

function Await($WinRtTask, $ResultType) {{
    $asTask = $asTaskGeneric.MakeGenericMethod($ResultType)
    $netTask = $asTask.Invoke($null, @($WinRtTask))
    $netTask.Wait(-1) | Out-Null
    $netTask.Result
}}

$asTaskAction = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {{
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncAction'
}})[0]

function AwaitAction($WinRtTask) {{
    $netTask = $asTaskAction.Invoke($null, @($WinRtTask))
    $netTask.Wait(-1) | Out-Null
}}

$file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync('{abs_pdf}')) ([Windows.Storage.StorageFile])
$pdf = Await ([Windows.Data.Pdf.PdfDocument]::LoadFromFileAsync($file)) ([Windows.Data.Pdf.PdfDocument])

$pages = @({page_list_str})
foreach ($p in $pages) {{
    $idx = [uint32]($p - 1)
    if ($idx -ge $pdf.PageCount) {{ continue }}
    $page = $pdf.GetPage($idx)
    $opts = New-Object Windows.Data.Pdf.PdfPageRenderOptions
    $opts.DestinationWidth = [uint32]([Math]::Round($page.Size.Width * {dpi_scale}))
    $opts.DestinationHeight = [uint32]([Math]::Round($page.Size.Height * {dpi_scale}))
    $outPath = Join-Path '{abs_out}' ('page_' + $p.ToString('000') + '.png')
    $stream = New-Object Windows.Storage.Streams.InMemoryRandomAccessStream
    AwaitAction ($page.RenderToStreamAsync($stream, $opts))
    $reader = New-Object Windows.Storage.Streams.DataReader($stream.GetInputStreamAt(0))
    $size = [uint32]$stream.Size
    $null = Await ($reader.LoadAsync($size)) ([uint32])
    $bytes = New-Object byte[] $size
    $reader.ReadBytes($bytes)
    [System.IO.File]::WriteAllBytes($outPath, $bytes)
    $reader.Dispose()
    $stream.Dispose()
    $page.Dispose()
}}
Write-Output ("Rendered " + $pages.Count + " page(s)")
"""
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"WinRT PDF render failed: {proc.stderr.strip() or proc.stdout.strip()}")
    return [output_dir / f"page_{p:03d}.png" for p in pages_1based]


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan PDF pages and test PNG rendering.")
    parser.add_argument("--pdf", type=Path, default=Path("data/vinamilk_ar_2022.pdf"))
    parser.add_argument("--stats-out", type=Path, default=Path("data/advanced_page_stats.json"))
    parser.add_argument("--test-render-dir", type=Path, default=Path("data/advanced_pages"))
    args = parser.parse_args()

    reader = PdfReader(str(args.pdf))
    stats: list[dict[str, Any]] = []
    for idx, page in enumerate(reader.pages, start=1):
        try:
            raw_text = page.extract_text() or ""
            text = normalize_ws(raw_text)
            err = None
        except Exception as exc:
            text = ""
            err = str(exc)
        try:
            n_imgs = len(page.images)
        except Exception:
            n_imgs = 0
        stats.append(
            {
                "page": idx,
                "char_count": len(text),
                "image_count": n_imgs,
                "preview": text[:140],
                "error": err,
            }
        )

    args.stats_out.parent.mkdir(parents=True, exist_ok=True)
    args.stats_out.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    low_text = [s["page"] for s in stats if s["char_count"] < 50]
    with_imgs = [s["page"] for s in stats if s["image_count"] > 0]
    errors = [s["page"] for s in stats if s["error"]]
    print(f"Total pages: {len(stats)}")
    print(f"Pages with < 50 chars ({len(low_text)}): {low_text}")
    print(f"Pages with embedded images ({len(with_imgs)}): {with_imgs[:25]}{'...' if len(with_imgs) > 25 else ''}")
    print(f"Extraction errors: {errors}")

    rendered = render_pages_winrt(args.pdf, args.test_render_dir, [10], dpi_scale=1.5)
    with Image.open(rendered[0]) as im:
        print(f"Test render {rendered[0]}: size={im.size}, mode={im.mode}")


if __name__ == "__main__":
    main()
