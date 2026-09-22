import argparse
import hashlib
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import tomllib
from pydantic import BaseModel

ROOT: Final = Path(__file__).resolve().parents[1]
OUTPUT: Final = ROOT / "litellm-rust/THIRD_PARTY_NOTICES.txt"
TARGETS: Final = (
    "x86_64-unknown-linux-gnu",
    "aarch64-unknown-linux-gnu",
    "x86_64-apple-darwin",
    "aarch64-apple-darwin",
)
PERMISSIVE_IDENTIFIERS: Final = frozenset(
    {
        "MIT",
        "MIT-0",
        "Apache-2.0",
        "LLVM-exception",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "BSL-1.0",
        "ISC",
        "CC0-1.0",
        "Unlicense",
        "Unicode-3.0",
        "CDLA-Permissive-2.0",
    }
)
SIMD_SOURCE: Final = "https://github.com/Nugine/simd/blob/d74c030d9dc4f3cae02146d1f497ff62726ef09a/LICENSE"
SIMD_SHA256: Final = "71674605ec4c087fe9eb534e3e4f9e26eb2e4aabcd76a29fd156c6a844d44b3d"


class CargoPackage(BaseModel):
    name: str
    version: str
    source: str | None
    manifest_path: str
    license: str | None
    license_file: str | None


class CargoMetadata(BaseModel):
    packages: tuple[CargoPackage, ...]


@dataclass(frozen=True)
class Notice:
    package: str
    source: str
    text: str


def cargo(*args: str) -> str:
    return subprocess.run(
        ("cargo", *args), cwd=ROOT / "litellm-rust", check=True, capture_output=True, text=True
    ).stdout


def selected_names() -> frozenset[str]:
    return frozenset(
        line.split(" (", 1)[0]
        for target in TARGETS
        for line in cargo(
            "tree",
            "--locked",
            "--offline",
            "--target",
            target,
            "--edges",
            "normal,build",
            "--package",
            "litellm-python-bridge",
            "--features",
            "extension-module",
            "--prefix",
            "none",
            "--format",
            "{p}",
        ).splitlines()
    )


def package_notices(package: CargoPackage) -> tuple[Notice, ...]:
    identifiers: Final = frozenset(re.split(r"\s+(?:AND|OR|WITH)\s+|[()/]", package.license or ""))
    assert {identifier.strip() for identifier in identifiers if identifier.strip()} <= PERMISSIVE_IDENTIFIERS, (
        f"Review new native license before packaging: {package.name}: {package.license}"
    )
    assert package.license, f"Missing native license: {package.name}"
    directory: Final = Path(package.manifest_path).parent
    identity: Final = f"{package.name} {package.version} ({package.license})"
    if package.name in {"base64-simd", "vsimd"} and package.version == "0.8.0":
        supplemental: Final = ROOT / "litellm-rust/notices/simd-0.8.0-LICENSE"
        assert hashlib.sha256(supplemental.read_bytes()).hexdigest() == SIMD_SHA256
        return (Notice(identity, SIMD_SOURCE, supplemental.read_text()),)
    files: Final = tuple(
        sorted(
            path
            for path in directory.rglob("*")
            if path.is_file()
            and (
                re.fullmatch(r"(?i)(?:LICENSE|LICENCE|COPYING|NOTICE|COPYRIGHT)(?:[._-].*)?", path.name)
                or "licenses" in {part.lower() for part in path.relative_to(directory).parts[:-1]}
                or (package.license_file is not None and path == directory / package.license_file)
            )
        )
    )
    assert files, f"Missing upstream notices: {identity}"
    return tuple(
        Notice(
            identity,
            f"https://docs.rs/crate/{package.name}/{package.version}/source/{path.relative_to(directory)}",
            path.read_text(),
        )
        for path in files
    )


def render() -> str:
    toolchain: Final = tomllib.loads((ROOT / "rust-toolchain.toml").read_text())["toolchain"]["channel"]
    rust_version: Final = subprocess.check_output(("rustc", "--version"), cwd=ROOT, text=True).split()[1]
    assert rust_version == toolchain, "Native notices require the repository's pinned Rust toolchain"
    sysroot: Final = Path(subprocess.check_output(("rustc", "--print", "sysroot"), cwd=ROOT, text=True).strip())
    library_notice: Final = ROOT / "litellm-rust/notices/RUST-COPYRIGHT-library.html"
    assert library_notice.read_bytes() == (sysroot / "share/doc/rust/COPYRIGHT-library.html").read_bytes(), (
        "Update the standard-library notice from the pinned Rust toolchain"
    )
    selected: Final = selected_names()
    metadata: Final = CargoMetadata.model_validate_json(
        cargo("metadata", "--locked", "--offline", "--format-version", "1")
    )
    packages: Final = tuple(
        sorted(
            (
                package
                for package in metadata.packages
                if package.source is not None and f"{package.name} v{package.version}" in selected
            ),
            key=lambda package: (package.name, package.version),
        )
    )
    assert packages and all(
        package.source == "registry+https://github.com/rust-lang/crates.io-index" for package in packages
    )
    notices: Final = tuple(notice for package in packages for notice in package_notices(package))
    texts: Final = tuple(
        sorted({notice.text for notice in notices}, key=lambda text: hashlib.sha256(text.encode()).hexdigest())
    )
    header: Final = (
        "OpenOrange LiteLLM native dependency notices\n\n"
        "Generated by scripts/generate_openorange_native_notices.py.\n"
        "Cargo.lock SHA-256: " + hashlib.sha256((ROOT / "litellm-rust/Cargo.lock").read_bytes()).hexdigest() + "\n"
        "Scope: litellm-python-bridge, extension-module and default features; normal/build dependencies.\n"
        "Targets: " + ", ".join(TARGETS) + ".\n"
        "This conservative inventory includes build tools and nested upstream notices; not every component is linked on every target.\n"
        "The following texts retain upstream notices and do not replace LiteLLM's own LICENSE.\n\n"
        + "\n".join(f"{package.name} {package.version}: {package.license}" for package in packages)
        + "\n"
    )
    return header + "".join(
        "\n"
        + "=" * 78
        + "\n"
        + "\n".join(sorted(f"{notice.package}\nSource: {notice.source}" for notice in notices if notice.text == text))
        + "\n\n"
        + text.rstrip()
        + "\n"
        for text in texts
    )


if __name__ == "__main__":
    parser: Final = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args: Final = parser.parse_args()
    rendered: Final = render()
    if args.write:
        OUTPUT.write_text(rendered)
    else:
        assert OUTPUT.read_text() == rendered, "Native notices are stale; regenerate with --write"
    sys.stdout.write("Native dependency notices verified\n")
