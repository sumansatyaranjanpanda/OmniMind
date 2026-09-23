"""Retry classification in deploy/oci_provision.py.

An unattended overnight provisioning run died repeatedly and the cause was not
capacity — it was a ConnectTimeout escaping the handler. The trap: the OCI SDK
vendors its own copy of `requests`, so `oci.exceptions.ConnectTimeout` inherits from
the VENDORED RequestException, a different class object from
`oci.exceptions.RequestException`. An `except oci.exceptions.RequestException`
clause therefore never fires for it, even though "RequestException" appears in its
MRO. Both hierarchies do bottom out at OSError.
"""

import oci
import pytest


def _classify(err):
    """Mirror of the classifier in oci_provision.main(), which is a closure."""
    if isinstance(err, oci.exceptions.ServiceError):
        msg = str(err.message).lower()
        if err.status == 429:
            return "rate-limited"
        if err.status in (500, 502, 503) and ("capacity" in msg or "out of host" in msg):
            return "no capacity"
        return None
    if isinstance(err, OSError):
        return "network unreachable"
    return None


def test_the_vendoring_trap_is_real_not_hypothetical():
    """If this ever becomes True, the name-based catch would have been fine."""
    assert not issubclass(oci.exceptions.ConnectTimeout, oci.exceptions.RequestException)
    assert issubclass(oci.exceptions.ConnectTimeout, OSError)


def test_connect_timeout_is_retryable_not_fatal():
    """The exact exception that killed an unattended run."""
    assert _classify(oci.exceptions.ConnectTimeout("timed out")) == "network unreachable"


def test_capacity_shortage_is_retryable():
    err = oci.exceptions.ServiceError(500, "InternalError", {}, "Out of host capacity.")
    assert _classify(err) == "no capacity"


def test_rate_limit_is_retryable_even_though_it_never_mentions_capacity():
    err = oci.exceptions.ServiceError(429, "TooManyRequests", {}, "Too many requests for the user")
    assert _classify(err) == "rate-limited"


@pytest.mark.parametrize("status,code,msg", [
    (401, "NotAuthenticated", "bad key"),
    (404, "NotFound", "image not found"),
    (400, "InvalidParameter", "bad shape config"),
])
def test_real_configuration_errors_still_abort(status, code, msg):
    """Retrying a bad API key for ten hours would hide the actual problem."""
    assert _classify(oci.exceptions.ServiceError(status, code, {}, msg)) is None
