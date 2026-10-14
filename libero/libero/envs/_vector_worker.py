"""Private fresh-interpreter vector worker; never imports a parent __main__."""
import importlib
import os
import sys
from multiprocessing.connection import Connection
import cloudpickle


def main():
    connections = [Connection(int(fd)) for fd in sys.argv[1].split(',')]
    try:
        module_name, search_path, factories_bytes, share_memory = cloudpickle.load(sys.stdin.buffer)
        sys.path[:] = search_path
        module = importlib.import_module(module_name)
        factories = cloudpickle.loads(factories_bytes)
    except BaseException as exc:
        import traceback
        for connection in connections:
            connection.send(('libero.worker.failure', type(exc).__name__, str(exc), traceback.format_exc()))
            connection.close()
        return
    if len(factories) == 1:
        module._worker(None, connections[0], module.CloudpickleWrapper(factories[0]), share_memory)
    else:
        module._group_worker(connections, factories, share_memory)


if __name__ == '__main__':
    main()
