# Copyright (c) 2009, Giampaolo Rodola'. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""Routines common to all posix systems."""

import glob
import os
import sys
import time

from ._common import memoize
from ._common import sdiskusage
from ._common import TimeoutExpired
from ._common import usage_percent
from ._compat import ChildProcessError
from ._compat import FileNotFoundError
from ._compat import InterruptedError
from ._compat import PermissionError
from ._compat import ProcessLookupError
from ._compat import PY3
from ._compat import unicode


__all__ = ["pid_exists", "wait_pid", "disk_usage", "get_terminal_map"]


def pid_exists(pid):
    """Check whether pid exists in the current process table."""
    if pid == 0:
        # According to "man 2 kill" PID 0 has a special meaning:
        # it refers to <<every process in the process group of the
        # calling process>> so we don't want to go any further.
        # If we get here it means this UNIX platform *does* have
        # a process with id 0.
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # EPERM clearly means there's a process to deny access to
        return True
    # According to "man 2 kill" possible error values are
    # (EINVAL, EPERM, ESRCH)
    else:
        return True


def wait_pid(pid, timeout=None, proc_name=None):
    """Wait for process with pid 'pid' to terminate and return its
    exit status code as an integer.

    If pid is not a children of os.getpid() (current process) just
    waits until the process disappears and return None.

    If pid does not exist at all return None immediately.

    Raise TimeoutExpired on timeout expired.
    """

    def check_timeout(delay):
        if timeout is not None:
            if timer() >= stop_at:
                raise TimeoutExpired(timeout, pid=pid, name=proc_name)
        time.sleep(delay)
        return min(delay * 2, 0.04)

    timer = getattr(time, "monotonic", time.time)
    if timeout is not None:

        def waitcall():
            return os.waitpid(pid, os.WNOHANG)

        stop_at = timer() + timeout
    else:

        def waitcall():
            return os.waitpid(pid, 0)

    delay = 0.0001
    while True:
        try:
            retpid, status = waitcall()
        except InterruptedError:
            delay = check_timeout(delay)
        except ChildProcessError:
            # This has two meanings:
            # - pid is not a child of os.getpid() in which case
            #   we keep polling until it's gone
            # - pid never existed in the first place
            # In both cases we'll eventually return None as we
            # can't determine its exit status code.
            while True:
                if pid_exists(pid):
                    delay = check_timeout(delay)
                else:
                    return
        else:
            if retpid == 0:
                # WNOHANG was used, pid is still running
                delay = check_timeout(delay)
                continue
            # process exited due to a signal; return the integer of
            # that signal
            if os.WIFSIGNALED(status):
                return -os.WTERMSIG(status)
            # process exited using exit(2) system call; return the
            # integer exit(2) system call has been called with
            elif os.WIFEXITED(status):
                return os.WEXITSTATUS(status)
            else:
                # should never happen
                raise ValueError("unknown process exit status %r" % status)


def disk_usage(path):
    """Return disk usage associated with path.
    Note: UNIX usually reserves 5% disk space which is not accessible
    by user. In this function "total" and "used" values reflect the
    total and used disk space whereas "free" and "percent" represent
    the "free" and "used percent" user disk space.
    """
    if PY3:
        st = os.statvfs(path)
    else:
        # os.statvfs() does not support unicode on Python 2:
        # - https://github.com/giampaolo/psutil/issues/416
        # - http://bugs.python.org/issue18695
        try:
            st = os.statvfs(path)
        except UnicodeEncodeError:
            if isinstance(path, unicode):
                try:
                    path = path.encode(sys.getfilesystemencoding())
                except UnicodeEncodeError:
                    pass
                st = os.statvfs(path)
            else:
                raise

    # Total space which is only available to root (unless changed
    # at system level).
    total = st.f_blocks * st.f_frsize
    # Remaining free space usable by root.
    avail_to_root = st.f_bfree * st.f_frsize
    # Remaining free space usable by user.
    avail_to_user = st.f_bavail * st.f_frsize
    # Total space being used in general.
    used = total - avail_to_root
    # Total space which is available to user (same as 'total' but
    # for the user).
    total_user = used + avail_to_user
    # User usage percent compared to the total amount of space
    # the user can use. This number would be higher if compared
    # to root's because the user has less space (usually -5%).
    usage_percent_user = usage_percent(used, total_user, round_=1)

    # NB: the percentage is -5% than what shown by df due to
    # reserved blocks that we are currently not considering:
    # https://github.com/giampaolo/psutil/issues/829#issuecomment-223750462
    return sdiskusage(
        total=total, used=used, free=avail_to_user, percent=usage_percent_user
    )


@memoize
def get_terminal_map():
    """Get a map of device-id -> path as a dict.
    Used by Process.terminal()
    """
    ret = {}
    ls = glob.glob("/dev/tty*") + glob.glob("/dev/pts/*")
    for name in ls:
        assert name not in ret, name
        try:
            ret[os.stat(name).st_rdev] = name
        except FileNotFoundError:
            pass
    return ret
