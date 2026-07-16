"""Shared error-shape handling for idempotent `create_*` calls.

Every setup script in aws/ must be safe to re-run, which means treating "the
resource is already there" as success. AWS makes that harder than it sounds: the
SageMaker create APIs are NOT consistent about how they signal a duplicate.
Inspecting botocore's own service model:

    CreateModelPackageGroup -> ['ResourceLimitExceeded']
    CreateExperiment        -> ['ResourceLimitExceeded']
    CreateTrial             -> ['ResourceNotFound', 'ResourceLimitExceeded']
    CreateModelPackage      -> ['ConflictException', 'ResourceLimitExceeded']

None of them declare a typed "already exists" error, and in practice a duplicate
surfaces as a generic ValidationException whose *message* is the only signal.
(Other services do raise typed errors — ECR has RepositoryAlreadyExistsException,
S3 has BucketAlreadyOwnedByYou.) So this predicate matches both the typed codes
and the message, and lives in one place rather than being re-guessed per script.
"""

from typing import Any

_ALREADY_EXISTS_CODES = {"ResourceInUse", "ConflictException", "RepositoryAlreadyExistsException"}


def already_exists(exc: Any) -> bool:
    """Whether a create_* call failed only because the resource already exists.

    Pure — takes a botocore ClientError-shaped object, so it is unit-testable in
    CI without AWS or moto.
    """
    error = getattr(exc, "response", {}).get("Error", {})
    code = error.get("Code", "")
    if code in _ALREADY_EXISTS_CODES:
        return True
    return code == "ValidationException" and "already exists" in error.get("Message", "")
