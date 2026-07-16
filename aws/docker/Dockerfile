# Custom SageMaker training image (BYOC) for the LightGBM baseline.
#
# Why not a prebuilt AWS framework image? They pin their own Python: the newest
# SKLearn training image is 1.2-1 on Python 3.9, and this project targets 3.12
# (PEP 604 unions like `TFTConfig | None` are a syntax error on 3.9). Rather than
# contort the portable package to fit a container, we own the container. That is
# the real-world reason teams move to BYOC, and it costs us one Dockerfile.
#
# `sagemaker-training` is the piece that keeps SCRIPT MODE working: it provides
# the `train` entrypoint implementing SageMaker's container contract (fetch
# sourcedir.tar.gz from S3, export SM_CHANNEL_*/SM_MODEL_DIR, invoke the entry
# point named by SAGEMAKER_PROGRAM). So train_entry.py is unchanged from the
# prebuilt-container version — only the base image differs.
#
# Deps are baked in at build time rather than pip-installed per job: faster job
# starts and a reproducible, pinned environment.
FROM python:3.12-slim

# libgomp1 is LightGBM's OpenMP runtime — it will not import without it.
# build-essential is needed to compile any sdist-only wheels during pip install.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Match the versions the portable package is developed against (see pyproject.toml).
# Specifiers MUST be quoted: an unquoted `<5` is a shell redirect, not a version bound.
RUN pip install --no-cache-dir \
    "sagemaker-training>=4.7" \
    "lightgbm>=4.3,<5" \
    "pandas>=2.2,<3" \
    "numpy>=1.26,<2" \
    "pyarrow>=15,<21" \
    "pydantic>=2.6,<3" \
    "pyyaml>=6,<7"

# Unbuffered stdout matters: SageMaker scrapes metric regexes from the log
# stream, and buffered output can be lost entirely if the job fails.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
