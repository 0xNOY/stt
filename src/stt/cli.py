import argparse
import os
import sys
import typing


def get_location(module: typing.Any) -> str:
    file_path = getattr(module, "__file__", None)
    if file_path:
        return os.path.dirname(file_path)
    else:
        # namespace package or similar
        return list(getattr(module, "__path__", []))[0]


def setup_environment():
    """
    Check if NVIDIA library paths are in LD_LIBRARY_PATH.
    If not, add them and re-execute the process.
    """
    try:
        import nvidia.cublas.lib
        import nvidia.cudnn.lib
    except ImportError:
        # If these are not installed, we can't do anything.
        # This might happen in non-GPU environments or if deps are missing.
        return

    cublas_path = get_location(nvidia.cublas.lib)
    cudnn_path = get_location(nvidia.cudnn.lib)

    env_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    current_paths = env_ld_path.split(os.pathsep) if env_ld_path else []

    new_paths = []
    if cublas_path not in current_paths:
        new_paths.append(cublas_path)
    if cudnn_path not in current_paths:
        new_paths.append(cudnn_path)

    if new_paths:
        # Prepend new paths
        updated_paths = new_paths + current_paths
        new_ld_path = os.pathsep.join(updated_paths)

        # Update environment variable
        os.environ["LD_LIBRARY_PATH"] = new_ld_path

        # Re-execute the process with the new environment
        # We need to use sys.executable to ensure we run with the same python interpreter
        print(f"Updating LD_LIBRARY_PATH and restarting: {new_paths}")
        try:
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except OSError as e:
            print(f"Failed to re-execute process: {e}")
            sys.exit(1)


def main():
    setup_environment()

    parser = argparse.ArgumentParser(description="STT CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Serve the STT application")
    serve_subparsers = serve_parser.add_subparsers(
        dest="subcommand", help="Serve subcommands"
    )

    # serve http
    serve_subparsers.add_parser("http", help="Start the HTTP node")

    args = parser.parse_args()

    if args.command == "serve":
        if args.subcommand == "http":
            try:
                from stt.http_node import start as start_http_node
            except ImportError as e:
                print(f"Error: Server dependencies are missing. ({e})")
                print("To run the server, please install with the 'server' extra:")
                print("  pip install '.[server]'")
                print("Or if using uvx:")
                print("  uvx --from 'stt[server]' stt serve http")
                sys.exit(1)

            start_http_node()
        else:
            serve_parser.print_help()
    elif args.command == "viz":
        from stt.viz import start

        start()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
