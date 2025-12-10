import argparse
from stt.http_node import start as start_http_node


def main():
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
            start_http_node()
        else:
            serve_parser.print_help()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
