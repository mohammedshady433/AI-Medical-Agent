"""Real process-tree regression for orphaned private GPU workers on Windows."""
import asyncio
import os
import pytest
import json
import psutil
import subprocess
import sys

from app.services.local_runtime import LocalRuntime


async def test_close_stops_owned_tree_but_preserves_unrelated_process(tmp_path):
    try:
        psutil.pids()
    except (PermissionError, psutil.AccessDenied):
        pytest.skip('Host sandbox denies process enumeration; run the real tree test on CI')
    pid_file = tmp_path / "child.json"
    script = tmp_path / "parent.py"
    script.write_text(
        "import subprocess,sys,time,json\nfrom pathlib import Path\n"
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)'])\n"
        "Path(sys.argv[1]).write_text(json.dumps(child.pid))\ntime.sleep(120)\n", encoding="utf8")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    unrelated = psutil.Popen([sys.executable, "-c", "import time; time.sleep(120)"], creationflags=flags)
    runtime = LocalRuntime(None)
    runtime.process = psutil.Popen([sys.executable, str(script), str(pid_file)], creationflags=flags, start_new_session=os.name != "nt")
    parent, child = runtime.process, None
    try:
        for _ in range(100):
            if pid_file.exists():
                child = psutil.Process(json.loads(pid_file.read_text()))
                break
            await asyncio.sleep(.05)
        assert child is not None, "Fixture child did not start"
        await runtime.close()
        assert runtime.process is None
        assert parent.poll() is not None
        assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
        assert unrelated.poll() is None
        await runtime.close()  # Closing twice must not affect a different process.
        assert unrelated.poll() is None
    finally:
        for process in [child, parent, unrelated]:
            if process is not None:
                try:
                    process.kill()
                    process.wait(5)
                except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                    pass


async def test_no_owned_process_leaves_external_runtime_untouched():
    runtime = LocalRuntime(None)
    await runtime.close()
    assert runtime.process is None
