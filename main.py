import sys
import subprocess
import argparse

def run_generator(args):
    cmd = [sys.executable, "src/quote_generator.py"] + args
    subprocess.run(cmd)

def run_learner(args):
    cmd = [sys.executable, "src/learner.py"] + args
    subprocess.run(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quotes Engine Management Script")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Subparser for generate
    gen_parser = subparsers.add_parser("generate", help="Generate structured quotes")
    gen_parser.add_argument("--num_quotes", type=int, default=5)
    gen_parser.add_argument("--rate", action="store_true", help="Rate quotes interactively")
    gen_parser.add_argument("--raw", action="store_true", help="Print only the quote text")

    # Subparser for learn
    learn_parser = subparsers.add_parser("learn", help="Learn from rated quotes")

    args, unknown = parser.parse_known_args()

    if args.command == "generate":
        pass_args = []
        if args.num_quotes: pass_args += ["--num_quotes", str(args.num_quotes)]
        if args.rate: pass_args.append("--rate")
        if args.raw: pass_args.append("--raw")
        pass_args.extend(unknown)
        run_generator(pass_args)
    elif args.command == "learn":
        run_learner(unknown)
    else:
        parser.print_help()
