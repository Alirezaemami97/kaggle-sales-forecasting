"""Compatibility shim for the SageMaker Python SDK on Windows.

The SDK's `sagemaker._studio._find_config` walks parent directories with
`while path is None and not wd.match("/")` to locate a Studio project config.
On Windows this never terminates: a Windows path never matches the POSIX root
"/", and the parent of a drive root (C:\\) is itself — so the loop spins
forever. It is reached from `create_feature_group()` and `Estimator.fit()` via
`_append_project_tags()`, which hang instead of failing.

We do not run inside SageMaker Studio, so short-circuiting that lookup to return
None is safe (its only effect is adding Studio project tags) and stops the hang.
Call `apply()` once before any SDK call that creates a resource.
"""


def apply() -> None:
    """Disable the SDK's Studio-config directory walk (Windows infinite-loop fix)."""
    import sagemaker._studio

    sagemaker._studio._find_config = lambda *_args, **_kwargs: None
