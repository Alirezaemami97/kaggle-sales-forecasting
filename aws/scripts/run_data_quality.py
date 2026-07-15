"""Evaluate the Glue Data Quality ruleset against the features table.

Registers (or updates) the DQDL ruleset in aws/glue/data_quality_rules.dqdl
against the Glue Catalog table created by create_features_table.sql, starts an
evaluation run, and prints a pass/fail summary per rule — the AWS-native
analogue of the pydantic checks in demand_forecasting/data/schema.py.

DQDL has no comment syntax, so the rule-by-rule rationale lives here instead:
  - IsComplete "id" / "d"        — identity/day keys must never be null
                                   (schema.py: null checks on identity columns)
  - ColumnValues "sales" >= 0    — units sold can never be negative
                                   (schema.py: validate_sales non-negative check)
  - ColumnValues "sell_price" > 0 — every feature row has drop_pre_release
                                   applied, so price is always known and
                                   positive here (schema.py: validate_prices)
  - ColumnValues "d" >= 1 / <= 2000 — sanity bound (M5 spans d_1..d_1969); split
                                   into two comparisons rather than "between",
                                   which hit an ambiguous default-threshold
                                   result in this account and is worth avoiding
  - RowCount > 40000000         — loose floor on the real 46,881,677-row join,
                                   so a badly truncated run fails loudly

    python aws/scripts/run_data_quality.py --role-arn <glue-role-arn> --wait
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import boto3

logger = logging.getLogger(__name__)

RULESET_NAME = "demand-forecasting-features-quality"
DQDL_PATH = Path(__file__).resolve().parent.parent / "glue" / "data_quality_rules.dqdl"
_TERMINAL_STATES = ("SUCCEEDED", "FAILED", "STOPPED", "TIMEOUT")


def create_or_update_ruleset(glue: Any, database: str, table: str) -> None:
    ruleset = DQDL_PATH.read_text(encoding="utf-8")
    target = {"TableName": table, "DatabaseName": database}
    try:
        glue.create_data_quality_ruleset(Name=RULESET_NAME, Ruleset=ruleset, TargetTable=target)
        logger.info("Created ruleset %s", RULESET_NAME)
    except glue.exceptions.AlreadyExistsException:
        glue.update_data_quality_ruleset(Name=RULESET_NAME, Ruleset=ruleset)
        logger.info("Updated existing ruleset %s", RULESET_NAME)


def start_run(glue: Any, database: str, table: str, role_arn: str) -> str:
    resp = glue.start_data_quality_ruleset_evaluation_run(
        DataSource={"GlueTable": {"DatabaseName": database, "TableName": table}},
        Role=role_arn,
        RulesetNames=[RULESET_NAME],
    )
    run_id: str = resp["RunId"]
    logger.info("Started evaluation run %s", run_id)
    return run_id


def wait_for_run(glue: Any, run_id: str) -> dict[str, Any]:
    while True:
        run = glue.get_data_quality_ruleset_evaluation_run(RunId=run_id)
        state = run["Status"]
        logger.info("Run %s: %s", run_id, state)
        if state in _TERMINAL_STATES:
            return dict(run)
        time.sleep(15)


def print_results(glue: Any, run: dict[str, Any]) -> None:
    for result_id in run.get("ResultIds", []):
        result = glue.get_data_quality_result(ResultId=result_id)
        for rule in result.get("RuleResults", []):
            logger.info(
                "[%s] %s — %s", rule["Result"], rule["Name"], rule.get("EvaluationMessage", "")
            )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default="demand_forecasting")
    parser.add_argument("--table", default="features")
    parser.add_argument("--role-arn", required=True, help="Glue service role ARN")
    parser.add_argument(
        "--wait", action="store_true", help="Poll until the run finishes and print results"
    )
    args = parser.parse_args()

    glue = boto3.client("glue")
    create_or_update_ruleset(glue, args.database, args.table)
    run_id = start_run(glue, args.database, args.table, args.role_arn)
    if args.wait:
        run = wait_for_run(glue, run_id)
        print_results(glue, run)


if __name__ == "__main__":
    main()
