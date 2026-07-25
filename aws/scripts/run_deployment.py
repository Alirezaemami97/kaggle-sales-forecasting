"""Phase 8 deployment harness: compare a candidate to the incumbent, promote or roll back.

This is a BATCH system, so shadow/A-B are Batch Transform comparisons, not live
traffic splitting. The default path here is the cheap one — read each version's
backtest WAPE from its registry ModelMetrics (attached by the Phase-6 pipeline)
and promote the candidate only if it beats the incumbent. Promote = Approve;
rollback = Reject the current top so the prior Approved is production again. A
batch job reads latest-Approved at run time, so a rollback is one API call.

    python aws/scripts/run_deployment.py --compare
    python aws/scripts/run_deployment.py --promote     # gated on beating the incumbent
    python aws/scripts/run_deployment.py --rollback
"""

import argparse
import logging
from typing import Any

logger = logging.getLogger(__name__)


def promote_decision(incumbent_wape: float | None, candidate_wape: float | None) -> bool:
    """Promote if there is no incumbent yet (bootstrap the first production) or the
    candidate is at least as good on WAPE (lower is better). A candidate with no
    metric can't be verified, so it is not promoted. Pure — unit-tested in CI."""
    if incumbent_wape is None:
        return True
    if candidate_wape is None:
        return False
    return candidate_wape <= incumbent_wape


def _describe(pkg: dict[str, Any] | None, wape: float | None) -> str:
    if pkg is None:
        return "none"
    return f"v{pkg['ModelPackageVersion']} (wape={wape if wape is not None else 'n/a'})"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", action="store_true", help="Show incumbent vs candidate")
    parser.add_argument("--promote", action="store_true", help="Approve the candidate if it wins")
    parser.add_argument("--rollback", action="store_true", help="Reject the current Approved top")
    parser.add_argument("--force", action="store_true", help="Promote even if it does not win")
    parser.add_argument("--group", help="Model package group (defaults to the LGBM group)")
    args = parser.parse_args()

    import boto3
    import registry
    import sagemaker_compat

    sagemaker_compat.apply()  # Windows SDK infinite-loop fix; see sagemaker_compat.py

    boto_session = boto3.Session()
    sm = boto_session.client("sagemaker")
    s3 = boto_session.client("s3")
    group = args.group or registry.MODEL_PACKAGE_GROUP

    versions = registry.list_versions(sm, group)
    incumbent = registry.latest_with_status(versions, registry.APPROVED)
    candidate = registry.latest_with_status(versions, registry.PENDING)

    if args.rollback:
        if incumbent is None:
            logger.info("Nothing Approved to roll back in %s", group)
            return
        registry.set_approval(sm, incumbent["ModelPackageArn"], registry.REJECTED)
        logger.info(
            "Rolled back v%d; the prior Approved version is production again",
            incumbent["ModelPackageVersion"],
        )
        return

    if candidate is None:
        logger.info("No Pending candidate in %s — nothing to compare or promote", group)
        return

    inc_wape = registry.metric_wape(sm, s3, incumbent["ModelPackageArn"]) if incumbent else None
    cand_wape = registry.metric_wape(sm, s3, candidate["ModelPackageArn"])
    decision = promote_decision(inc_wape, cand_wape)
    logger.info(
        "Incumbent %s | Candidate %s | promote=%s",
        _describe(incumbent, inc_wape), _describe(candidate, cand_wape), decision,
    )

    if args.promote:
        if decision or args.force:
            registry.set_approval(sm, candidate["ModelPackageArn"], registry.APPROVED)
            logger.info("Promoted candidate v%d to Approved", candidate["ModelPackageVersion"])
        else:
            logger.info("Candidate did not beat the incumbent; not promoted (use --force)")


if __name__ == "__main__":
    main()
