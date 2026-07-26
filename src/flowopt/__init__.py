"""flowopt — find the best LLM workflow for a task under a budget.

A workflow is a Python program `solve(question, call_model) -> answer`. A design
agent writes candidates, each is metered and graded on real examples, and the
ones on the accuracy/cost Pareto frontier are what you choose between.

    from flowopt import Session, analysis, optimize, report

    session = Session.load("gsm8k")
    benchmark = analysis.build_benchmark(session.cfg, session.client)
    search = optimize(session.cfg, benchmark, session.evaluator(benchmark.grader))
    report.summarize(search, session.cfg)

The vocabulary, in the order the pipeline uses it:

    Session       one run's config plus the client it calls models through
    TaskAnalysis  what the task is and how an answer should be graded
    Benchmark     that analysis, plus a Grader and the dev/test splits
    Candidate     one proposed workflow program and how it scored
    SplitScore    a candidate's accuracy and cost on one split
    Search        everything tried, and the finalists worth choosing between
"""
from . import analysis, dataset, designer, pareto, report
from .analysis import Benchmark, TaskAnalysis
from .client import ApiResponse, ModelClient
from .config import load_config
from .grading import Grader
from .models import ModelCatalog
from .optimizer import DEV, TEST, Candidate, Search, optimize
from .runtime import ANSWER_SCHEMA, CallMeter, CallRecord, Evaluator, Reply, SplitScore
from .session import Session

__all__ = [
    "ANSWER_SCHEMA", "ApiResponse", "Benchmark", "CallMeter", "CallRecord", "Candidate",
    "DEV", "Evaluator", "Grader", "ModelCatalog", "ModelClient", "Reply", "Search",
    "Session", "SplitScore", "TEST", "TaskAnalysis", "analysis", "dataset", "designer",
    "load_config", "optimize", "pareto", "report",
]
