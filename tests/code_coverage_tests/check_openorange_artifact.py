import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import re
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Final


def restricted_path(path: str) -> bool:
    return any(part in {"enterprise", "litellm_enterprise"} for part in PurePosixPath(path).parts)


def check_wheel(path: Path, license_sha256: str, native_notices_sha256: str, rust_notices_sha256: str) -> None:
    with zipfile.ZipFile(path) as wheel:
        names: Final = tuple(wheel.namelist())
        assert not any(restricted_path(name) for name in names), "Enterprise source in wheel"
        metadata_names: Final = tuple(name for name in names if name.endswith(".dist-info/METADATA"))
        assert len(metadata_names) == 1, "Expected one wheel distribution"
        metadata: Final = wheel.read(metadata_names[0]).decode()
        assert not re.search(r"(?im)^Requires-Dist: litellm[-_]enterprise(?:\W|$)", metadata), (
            "Enterprise dependency in wheel metadata"
        )
        assert "License-Expression: MIT" in metadata, "Missing MIT license metadata"
        assert any(
            name.startswith("litellm/rust_bridge/_native") and name.endswith((".so", ".pyd")) for name in names
        ), "Native extension missing from wheel"
        notices: Final = tuple(name for name in names if ".dist-info/" in name and name.endswith("/LICENSE"))
        assert any(hashlib.sha256(wheel.read(name)).hexdigest() == license_sha256 for name in notices), (
            "Upstream LICENSE missing or changed in wheel"
        )
        native_notices: Final = tuple(
            name for name in names if ".dist-info/" in name and name.endswith("/THIRD_PARTY_NOTICES.txt")
        )
        assert any(hashlib.sha256(wheel.read(name)).hexdigest() == native_notices_sha256 for name in native_notices), (
            "Native dependency notices missing or changed in wheel"
        )
        rust_notices: Final = tuple(name for name in names if name.endswith("/RUST-COPYRIGHT-library.html"))
        assert any(hashlib.sha256(wheel.read(name)).hexdigest() == rust_notices_sha256 for name in rust_notices), (
            "Rust standard-library notices missing or changed in wheel"
        )
    sys.stdout.write(f"OSS package, native extension and upstream notice verified: {path.name}\n")


def check_installed(license_sha256: str, native_notices_sha256: str, rust_notices_sha256: str) -> None:
    names: Final = tuple(
        distribution.metadata["Name"].lower().replace("_", "-") for distribution in importlib.metadata.distributions()
    )
    assert "litellm-enterprise" not in names, "Enterprise distribution installed"
    assert not Path("/app/enterprise").exists(), "Enterprise source copied into runtime"
    assert importlib.util.find_spec("enterprise") is None, "Enterprise source importable"
    assert importlib.util.find_spec("litellm_enterprise") is None, "Enterprise package importable"
    distribution: Final = importlib.metadata.distribution("litellm")
    files: Final = tuple(distribution.files or ())
    assert not any(restricted_path(str(path)) for path in files), "Enterprise source installed with LiteLLM"
    notices: Final = tuple(path for path in files if ".dist-info/" in str(path) and path.name == "LICENSE")
    assert any(
        hashlib.sha256(Path(distribution.locate_file(path)).read_bytes()).hexdigest() == license_sha256
        for path in notices
    ), "Upstream LICENSE missing or changed in installed package"
    native_notices: Final = tuple(
        path for path in files if ".dist-info/" in str(path) and path.name == "THIRD_PARTY_NOTICES.txt"
    )
    assert any(
        hashlib.sha256(Path(distribution.locate_file(path)).read_bytes()).hexdigest() == native_notices_sha256
        for path in native_notices
    ), "Native dependency notices missing or changed in installed package"
    rust_notices: Final = tuple(path for path in files if path.name == "RUST-COPYRIGHT-library.html")
    assert any(
        hashlib.sha256(Path(distribution.locate_file(path)).read_bytes()).hexdigest() == rust_notices_sha256
        for path in rust_notices
    ), "Rust standard-library notices missing or changed in installed package"
    importlib.import_module("litellm.rust_bridge._native")
    importlib.import_module("litellm.proxy.proxy_server")
    sys.stdout.write("OSS runtime imports, native extension and upstream notice verified\n")


if __name__ == "__main__":
    parser: Final = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--license-sha256", required=True)
    parser.add_argument("--native-notices-sha256", required=True)
    parser.add_argument("--rust-notices-sha256", required=True)
    args: Final = parser.parse_args()
    if args.wheel:
        check_wheel(args.wheel, args.license_sha256, args.native_notices_sha256, args.rust_notices_sha256)
    else:
        check_installed(args.license_sha256, args.native_notices_sha256, args.rust_notices_sha256)
