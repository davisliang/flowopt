"""`flowopt` — run the whole pipeline for one task.

    flowopt --task gsm8k                     # a file under config/task/
    flowopt --task gsm8k designer.rounds=1   # override any config key
"""
import argparse
import functools

from . import analysis, report
from .optimizer import optimize
from .session import Session


def main() -> None:
    """Run analyze → design → rank → report for one task, from the command line.

    Reads the task and any config overrides from argv. Writes the search JSON to
    disk (and optionally a plot), printing the frontier and every finalist's code
    along the way.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="clinical_notes",
                        help="config/task/<name>.yaml (default: clinical_notes)")
    parser.add_argument("--out", default=None, help="where to write the search JSON")
    parser.add_argument("--plot", default=None, help="also save the frontier plot here")
    parser.add_argument("overrides", nargs="*", metavar="key=value",
                        help="config overrides, e.g. designer.rounds=1 runtime.concurrency=4")
    args = parser.parse_args()

    log = functools.partial(print, flush=True)   # subprocess output is block-buffered otherwise
    session = Session.load(args.task, args.overrides)
    cfg = session.cfg

    benchmark = analysis.build_benchmark(cfg, session.client, log=log)
    search = optimize(cfg, benchmark, session.evaluator(benchmark.grader), log=log)

    report.summarize(search, cfg, log=log)
    report.print_code(search, log=log)
    log(f"\nwritten: {report.save(search, cfg, args.out)}")
    if args.plot:
        report.plot(search, path=args.plot)
        log(f"plot:    {args.plot}")


if __name__ == "__main__":
    main()
