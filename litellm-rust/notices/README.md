# Native distribution notices

The Python wheel statically links Rust dependencies. Their license documents
must accompany that wheel even though those crates are absent from Python
package metadata. `pyproject.toml` includes the following alongside the unchanged
LiteLLM MIT `LICENSE`:

- `../THIRD_PARTY_NOTICES.txt`: full upstream license and notice texts for the
  locked native bridge dependency graph on OpenOrange's Linux and macOS targets.
  The inventory conservatively includes build dependencies and nested notices.
- `RUST-COPYRIGHT-library.html`: the unmodified standard-library copyright and
  license inventory from the pinned Rust toolchain's
  `share/doc/rust/COPYRIGHT-library.html`. Cargo metadata omits these dependencies.

`simd-0.8.0-LICENSE` supplies the MIT text missing from the published
`base64-simd` and `vsimd` 0.8.0 archives. Their `.cargo_vcs_info.json` points to
[this exact upstream source revision](https://github.com/Nugine/simd/blob/d74c030d9dc4f3cae02146d1f497ff62726ef09a/LICENSE).
The notice generator verifies the retained text's SHA-256.

After a native dependency or compiler update, review changed license terms,
refresh the compiler inventory from the newly pinned toolchain if necessary,
then run from the repository root:

```sh
rustup toolchain install --component rust-docs
cargo fetch --locked --manifest-path litellm-rust/Cargo.toml
uv run --no-sync python scripts/generate_openorange_native_notices.py --write
uv run --no-sync python scripts/generate_openorange_native_notices.py
```

Generation uses the locked Cargo graph and cached published source archives.
New license identifiers or missing upstream notice files fail the check and
require review. Keep copyright and notice text intact; this inventory does not
replace upstream licenses or grant permission for enterprise packages. Both the
wheel and installed-image CI checks verify that all three distributed notice
files match their source bytes and that enterprise modules are absent.
