"""The tool execution boundary: what a tool may read, load and write, decided
once for every MCP server instead of inside each of ~45 path-taking tools.

`install(mcp)` wraps the server's @mcp.tool() so every call has its arguments
checked by NAME before the tool runs — the same names mean the same thing on
every agent (path/data_path/... are CSVs read, pkl_path is a model loaded,
out_path is a file written, run_id names a run folder). Checks resolve
symlinks; the tool still gets the value as given (callers cannot create
symlinks, and a rewritten path breaks the champion's relative links).

With a user bound (every gateway request) the rules are:
  read   - own uploads (data/uploads/<user>), own predictions
           (data/predictions/<user>), shared datasets (data/*.csv,
           data/datasets, data/smoke), and a model's reference/forecast CSV
           when that model may be loaded. Data files only (core/datasource).
  load   - .pkl under data/artifacts whose sha256 matches the record written
           when export_model produced it, and which the user owns — or which
           is a registered version / champion (team assets).
  write  - export_model: .pkl under data/artifacts, never over a colleague's.
           Everything else: .csv under the user's predictions or uploads.
No user (tests, local CLI, trusted infra) skips the location rules, but a
pickle still loads only with a matching export record: a crafted pickle is
code execution whoever asks. Production refuses user-less requests in
core/runtime, so "no user" never reaches here from a person.
"""

import functools
import hashlib
import inspect
import json
import os
import re
import time
from pathlib import Path

from . import runtime
from .datasource import DATA_SUFFIXES, is_data_file

INPUT_ARGS = {"path", "data_path", "reference_path", "current_path", "future_exog_path"}
MODEL_ARGS = {"pkl_path"}
OUTPUT_ARGS = {"out_path"}
DIR_ARGS = {"directory"}
PROFILE_ARGS = {"profile_path"}
# Write repo files / shell out: fine for a developer at a terminal, never for
# a model steered by text inside someone's dataset.
DEV_ONLY_TOOLS = {"dvc_track", "generate_ci_workflow"}
_RUN_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_TEAM_MODEL = re.compile(r"-(v\d+|champion)$")


class Refused(ValueError):
    pass


def data_dir() -> Path:
    return Path(
        os.environ.get("AGENTIC_ML_DATA_DIR")
        or Path(__file__).resolve().parents[1] / "data"
    ).resolve()


def user_folder(user: str | None) -> str:
    """A user's folder name — also what the gateway uploads into. Leading dots
    stripped, so no identity can name '..' and land in the parent."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", user or "anonymous").lstrip(".") or "_"


def _inside(path: Path, root: Path) -> bool:
    return path == root or path.is_relative_to(root)


def _record(pkl: Path) -> Path:
    return pkl.with_name(pkl.name + ".sha256.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def trust(pkl: str | Path, owner: str | None = None) -> dict:
    """Record a .pkl as a genuine export. export_model does this through the
    guard; call it by hand only for a file you know export_model produced."""
    pkl = Path(pkl).resolve()
    record = {
        "sha256": sha256(pkl),
        "owner": owner,
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _record(pkl).write_text(json.dumps(record))
    return record


def _export_owner(pkl: Path) -> str | None:
    try:
        return json.loads(_record(pkl).read_text()).get("owner")
    except (OSError, ValueError):
        return None


def _model_access(pkl: Path, user: str) -> None:
    """A user may load a model they exported, or a team one (registered
    version / champion). Raises Refused otherwise."""
    if _TEAM_MODEL.search(pkl.stem):
        return
    if denied := runtime.access_error(_export_owner(pkl), "model"):
        raise Refused(denied)


def check_model(value: str) -> Path:
    pkl = Path(value).resolve()
    if pkl.suffix != ".pkl":
        raise Refused(f"'{value}' is not an exported model (.pkl)")
    if not pkl.is_file():
        raise Refused(f"no exported model at '{value}'")
    user = runtime.user()
    if user:
        if not _inside(pkl, data_dir() / "artifacts"):
            raise Refused(
                f"models load only from {data_dir() / 'artifacts'} — '{value}' is outside it"
            )
        _model_access(pkl, user)
    try:
        expected = json.loads(_record(pkl).read_text())["sha256"]
    except (OSError, ValueError, KeyError):
        raise Refused(
            f"'{value}' has no export record — only files written by export_model are loaded"
        )
    if sha256(pkl) != expected:
        raise Refused(f"'{value}' changed after it was exported — refusing to load it")
    return pkl


def _model_of(csv: Path) -> Path | None:
    for suffix in (".reference.csv", "_forecast.csv", ".monitoring.json"):
        if csv.name.endswith(suffix):
            return csv.with_name(csv.name[: -len(suffix)] + ".pkl")
    return None


def check_input(value: str) -> Path:
    path = Path(value).resolve()
    user = runtime.user()
    if not user:
        return path
    if not is_data_file(path):
        raise Refused(f"'{value}' is not a supported data file ({sorted(DATA_SUFFIXES)})")
    root = data_dir()
    own = [
        root / "uploads" / user_folder(user),
        root / "predictions" / user_folder(user),
        root / "datasets",
        root / "smoke",
    ]
    if any(_inside(path, r) for r in own) or path.parent == root:
        return path
    model = _model_of(path)
    if model and _inside(path, root / "artifacts"):
        check_model(str(model))
        return path
    raise Refused(
        f"'{value}' is outside your data — use a file you uploaded or a shared dataset"
    )


def check_profile(value: str) -> Path:
    path = Path(value).resolve()
    if runtime.user():
        model = _model_of(path)
        if not model or not _inside(path, data_dir() / "artifacts"):
            raise Refused(
                f"'{value}' is not a model's monitoring profile in {data_dir() / 'artifacts'}"
            )
        check_model(str(model))
    return path


def check_directory(value: str) -> Path:
    path = Path(value).resolve()
    if runtime.user() and path != data_dir() / "artifacts":
        raise Refused(f"only {data_dir() / 'artifacts'} can be listed")
    return path


def check_output(value: str, tool: str) -> Path:
    path = Path(value).resolve()
    user = runtime.user()
    if tool == "export_model":
        if path.suffix != ".pkl":
            raise Refused(f"export_model writes a .pkl — got '{value}'")
        if user:
            if not _inside(path, data_dir() / "artifacts"):
                raise Refused(
                    f"models export only to {data_dir() / 'artifacts'} — '{value}' is outside it"
                )
            if path.exists() and (
                denied := runtime.access_error(_export_owner(path), "model file")
            ):
                raise Refused(denied)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    if not user:
        return path
    if path.suffix.lower() not in (".csv", ".parquet"):
        raise Refused(f"'{value}' is not a CSV or parquet file — tools write data outputs only")
    root, folder = data_dir(), user_folder(user)
    if not any(
        _inside(path, r)
        for r in (root / "predictions" / folder, root / "uploads" / folder)
    ):
        raise Refused(
            f"outputs go under {root / 'predictions' / folder} — '{value}' is outside it"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def check_call(tool: str, arguments: dict, outputs: bool = True) -> None:
    """Raise Refused if this call may not run. Empty values are left to the
    tool (most treat "" as "use the default")."""
    if tool in DEV_ONLY_TOOLS and runtime.user():
        raise Refused(f"'{tool}' is a developer tool and is not available to agents")
    for name, value in arguments.items():
        if not isinstance(value, str) or not value:
            continue
        if name == "run_id" and not _RUN_ID.match(value):
            raise Refused(f"'{value}' is not a run id")
        if name in INPUT_ARGS:
            check_input(value)
        elif name in MODEL_ARGS:
            check_model(value)
        elif name in PROFILE_ARGS:
            check_profile(value)
        elif name in DIR_ARGS:
            check_directory(value)
        elif name in OUTPUT_ARGS and outputs:
            check_output(value, tool)


def install(mcp, outputs: bool = True):
    """Guard every tool registered on `mcp` from here on. Call right after
    creating the server, before any other decorator wrapper, so the guard is
    the outermost layer. outputs=False for a server that ignores out_path's
    folder itself (visualization keeps only the file name)."""
    register = mcp.tool

    def tool(*d_args, **d_kwargs):
        wrap = register(*d_args, **d_kwargs)

        def decorate(fn):
            signature = inspect.signature(fn)

            @functools.wraps(fn)  # FastMCP reads the signature through __wrapped__
            def guarded(*args, **kwargs):
                try:
                    check_call(
                        fn.__name__, signature.bind(*args, **kwargs).arguments, outputs
                    )
                except Refused as refused:
                    return json.dumps({"error": f"refused: {refused}"})
                result = fn(*args, **kwargs)
                if fn.__name__ == "export_model":
                    _record_export(result)
                return result

            return wrap(guarded)

        return decorate

    mcp.tool = tool
    return mcp


def _record_export(result) -> None:
    try:
        out = json.loads(result).get("out_path")
    except (TypeError, ValueError, AttributeError):
        return
    if out and Path(out).is_file():
        trust(out, runtime.user())


if __name__ == "__main__":
    # Operator use: bless exports that predate the guard, e.g.
    #   python -m core.toolguard data/artifacts/*.pkl
    import sys

    for arg in sys.argv[1:]:
        print(arg, trust(arg))
