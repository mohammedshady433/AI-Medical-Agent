"""Cooperative cancellation without a background stdin reader or reloading weights."""
import contextlib
import json
from pathlib import Path
import sys
import traceback


class SpeechCancelled(Exception):
    pass


class SpeechWorkerRestart(Exception):
    """An optimized-worker failure requires a fresh worker with source FP32 weights."""


class CancelFlag:
    def __init__(self, filename):
        self.path = Path(filename) if filename else None

    def is_set(self):
        return self.path is not None and self.path.is_file()


def check_cancelled(event):
    if event is not None and event.is_set():
        raise SpeechCancelled()


@contextlib.contextmanager
def cancellable_inference(model, event):
    """Check between decoder steps without changing weights, sampling or vendor code."""
    check_cancelled(event)
    handles = []
    if event is not None:
        for component in (model.t3.tfmr, model.s3gen):
            handles.append(component.register_forward_pre_hook(lambda *_: check_cancelled(event)))
    try:
        yield
        check_cancelled(event)
    finally:
        for handle in handles:
            handle.remove()


def serve(handle, source=None, output=None):
    source, output = source or sys.stdin, output or sys.stdout
    for line in source:
        try:
            request = json.loads(line)
            event = CancelFlag(request.get("cancel_file"))
            with contextlib.redirect_stdout(sys.stderr):
                check_cancelled(event)
                result = handle(request, event)
                check_cancelled(event)
        except SpeechCancelled:
            result = {"cancelled": True}
        except SpeechWorkerRestart:
            traceback.print_exc(file=sys.stderr)
            result = {"retry_without_decoder_graph": True}
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            result = {"error": str(error)[:500]}
        print(json.dumps(result, ensure_ascii=False), file=output, flush=True)
