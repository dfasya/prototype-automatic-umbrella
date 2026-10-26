"""Check the connection to the Shelly 2PM Gen4 and nudge the umbrella motor.

The Shelly 2PM Gen4 drives the tubular motor (SL45RM-50/12, 230 V AC) in its
"cover" profile: output O1 turns it one way ("open"), O2 the other ("close").
The Jetson talks to it over Wi-Fi/LAN with the Shelly RPC API (HTTP).

Without ``--move`` the script only reads: it never switches the motor.

    python test_shelly.py --host 192.168.1.50                  # connection + status only
    python test_shelly.py --host 192.168.1.50 --check-auth     # is a password set, is mine right?
    python test_shelly.py --host 192.168.1.50 --move open      # run 1 s one way
    python test_shelly.py --host 192.168.1.50 --move open --back            # forward, then back to the start
    python test_shelly.py --host 192.168.1.50 --move close --back --cycles 3
    python test_shelly.py --host 192.168.1.50 --stop           # stop the motor now

``--back`` makes a round trip that ends where it started: it measures how long
the motor really ran (shorter if a limit switch stopped it), runs the other
way for exactly that long, and, if the Shelly is calibrated, moves to the
exact start position. It then compares the state before and after.

Without hardware, start the simulated Shelly (modules/fake_shelly.py) and use
``--host 127.0.0.1:8080``.

With authentication enabled on the Shelly, pass ``--password`` or set
SHELLY_PASSWORD (the user is always "admin").

Safety: a move is at most 3 s, asks for confirmation (unless ``--yes``), is
watched while it runs, and always ends with Cover.Stop, also after an error.
The duration is sent to the Shelly with the command, so the Shelly stops the
motor itself even if the Jetson loses the connection. Keep clear of the
umbrella while testing.

Stopping: press Ctrl+C at any time. The motor is stopped first (Cover.Stop,
retried; pressing Ctrl+C again does not cancel it), then the final state is
printed. Closing the terminal, ``kill``, or Ctrl+Break do the same. From
another terminal, ``--stop`` is an emergency stop.
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from contextlib import contextmanager

import requests
from requests.auth import HTTPDigestAuth

MAX_DURATION = 3.0       # seconds; enough to see it move (12 rpm = 0.2 turn per second)
POLL_SECONDS = 0.1
MIN_RUNNING_WATTS = 5.0  # below this the motor most likely did not turn
POSITION_TOLERANCE = 1   # % of travel; a calibrated round trip should end this close to the start
MOVING = ("opening", "closing")
STOP_ATTEMPTS = 3        # Cover.Stop is retried this often ...
STOP_TIMEOUT = 1.5       # ... with this many seconds per attempt
# Signals that end the script like Ctrl+C does (motor stopped first), where the OS has them.
STOP_SIGNALS = [getattr(signal, name) for name in ("SIGINT", "SIGTERM", "SIGHUP", "SIGBREAK")
                if hasattr(signal, name)]


class ShellyError(RuntimeError):
    pass


class Shelly:
    """Minimal Shelly Gen2+ RPC client over HTTP."""

    def __init__(self, host: str, password: str | None = None, timeout: float = 3.0):
        self.base = f"http://{host}/rpc"
        self.auth = HTTPDigestAuth("admin", password) if password else None
        self.timeout = timeout
        self.session = requests.Session()
        self.moved = False       # a move command may have been sent during this run
        self.needs_stop = False  # ... and no Cover.Stop has been confirmed since

    def call(self, method: str, timeout: float | None = None, **params):
        timeout = self.timeout if timeout is None else timeout
        try:
            response = self.session.get(f"{self.base}/{method}", params=params,
                                        auth=self.auth, timeout=timeout)
        except requests.Timeout as exc:
            raise ShellyError(f"no answer from {self.base} within {timeout:g} s") from exc
        except requests.RequestException as exc:  # refused, reset, dropped mid-reply, ...
            raise ShellyError(f"cannot reach {self.base}: {exc.__class__.__name__}") from exc
        if response.status_code == 401:
            raise ShellyError("authentication required or wrong password "
                              "(use --password or SHELLY_PASSWORD; the user is 'admin'; see --check-auth)")
        try:
            body = response.json()
        except ValueError:
            raise ShellyError(f"{method}: HTTP {response.status_code}, not JSON: {response.text[:200]!r}")
        if response.status_code != 200 or (isinstance(body, dict) and "code" in body and "message" in body):
            raise ShellyError(f"{method}: error {body.get('code')}: {body.get('message')}")
        return body


def log(message: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}" if message else "", flush=True)


# ---------------------------------------------------------------- stopping --

def _as_ctrl_c(signum, frame):
    raise KeyboardInterrupt(f"signal {signum}")


def install_stop_signals() -> None:
    """Make kill, closing the terminal and Ctrl+Break behave like Ctrl+C.

    Without this they end Python at once, skipping the code that stops the motor.
    """
    for sig in STOP_SIGNALS:
        if sig != signal.SIGINT:
            signal.signal(sig, _as_ctrl_c)


@contextmanager
def uninterruptible():
    """Hold back Ctrl+C and stop signals so a stop command cannot be cancelled halfway."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def hold(signum, frame):
        log("  ... stopping the motor first, please wait")

    previous = {sig: signal.getsignal(sig) for sig in STOP_SIGNALS}
    for sig in STOP_SIGNALS:
        signal.signal(sig, hold)
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def stop_motor(shelly: Shelly, duration: float | None = None) -> bool:
    """Send Cover.Stop, retrying; Ctrl+C cannot interrupt it. True once the Shelly confirms.

    Safe to call at any time: stopping a motor that is not running does nothing.
    """
    with uninterruptible():
        for attempt in range(1, STOP_ATTEMPTS + 1):
            try:
                shelly.call("Cover.Stop", timeout=STOP_TIMEOUT, id=0)
                shelly.needs_stop = False
                log("Cover.Stop sent")
                return True
            except ShellyError as exc:
                log(f"  WARNING: Cover.Stop attempt {attempt}/{STOP_ATTEMPTS} failed: {exc}")
    if duration:
        log(f"  The Shelly stops by itself after the {duration:g} s it was given.")
    else:
        log("  The Shelly stops by itself at its maximum run time (maxtime_open / maxtime_close).")
    log("  If the motor keeps running, cut its power.")
    return False


def describe_status(st: dict) -> str:
    pos = st.get("current_pos")
    parts = [f"state={st.get('state')}",
             f"pos={'?' if pos is None else f'{pos}%'}",
             f"power={st.get('apower', 0):.1f} W",
             f"current={st.get('current', 0):.2f} A",
             f"voltage={st.get('voltage', 0):.0f} V"]
    if st.get("temperature", {}).get("tC") is not None:
        parts.append(f"temp={st['temperature']['tC']:.0f} C")
    if st.get("errors"):
        parts.append(f"ERRORS={st['errors']}")
    return "  ".join(parts)


# ------------------------------------------------------------------ checks --

def check_auth(host: str, password: str | None, timeout: float) -> int:
    """Whether the Shelly asks for a password, and whether the given one works.

    The password cannot be read back from the device; it can only be tested.
    """
    info = Shelly(host, None, timeout).call("Shelly.GetDeviceInfo")  # never needs a password
    log(f"Device {info.get('id')} ({info.get('app')}, firmware {info.get('ver')})")
    if not info.get("auth_en"):
        log("Authentication is OFF: no password needed. Anyone on this network can control it;")
        log("consider setting one (Shelly app or web UI, Settings > Authentication).")
        return 0
    log(f"Authentication is ON (user 'admin', realm {info.get('auth_domain')}).")
    if not password:
        log("No password given: pass --password or set SHELLY_PASSWORD to test one.")
        return 1
    try:
        Shelly(host, password, timeout).call("Sys.GetConfig")  # protected call
    except ShellyError as exc:
        if "authentication" in str(exc):
            log("Password is WRONG.")
            log("It cannot be read from the device. If it is lost: set a new one from the Shelly app")
            log("(if the device is still linked to your account), or factory-reset the Shelly with its")
            log("button (see the Shelly manual) and set it up again.")
            return 1
        raise
    log("Password is CORRECT.")
    return 0


def check_connection(shelly: Shelly, set_profile: bool) -> dict:
    """Device info, profile check, cover config and status. Returns the status."""
    log(f"Connecting to {shelly.base} ...")
    t0 = time.perf_counter()
    info = shelly.call("Shelly.GetDeviceInfo")
    log(f"Connected in {(time.perf_counter() - t0) * 1000:.0f} ms")
    for key in ("id", "model", "app", "gen", "ver", "profile", "auth_en"):
        log(f"  {key:8} {info.get(key)}")
    if "2PM" not in str(info.get("app", "")).upper() and "2PM" not in str(info.get("model", "")).upper():
        log("  WARNING: this does not look like a Shelly 2PM")

    if info.get("profile") != "cover":
        if not set_profile:
            raise ShellyError(
                f"the device is in the '{info.get('profile')}' profile; the motor needs 'cover'. "
                "Set it in the Shelly app/web UI, or re-run with --set-cover-profile "
                "(the device reboots and its schedules and webhooks are deleted)")
        log("Switching to the cover profile; the device reboots ...")
        shelly.call("Shelly.SetProfile", name="cover")
        raise ShellyError("profile changed; wait ~20 s for the reboot, then run this again")

    config = shelly.call("Cover.GetConfig", id=0)
    log("Cover config:")
    for key in ("name", "in_mode", "maxtime_open", "maxtime_close", "power_limit",
                "obstruction_detection", "safety_switch"):
        if key in config:
            log(f"  {key:22} {config[key]}")

    status = shelly.call("Cover.GetStatus", id=0)
    log("Cover status:")
    log(f"  {describe_status(status)}")
    log(f"  calibrated (position control): {status.get('pos_control')}")
    return status


# ------------------------------------------------------------------ moving --

def _watch(shelly: Shelly, t0: float, deadline: float) -> tuple[float, float, dict]:
    """Poll until the cover stops; returns (peak watts, seconds it ran, last status)."""
    peak, ran, st = 0.0, 0.0, {}
    while time.perf_counter() < deadline:
        st = shelly.call("Cover.GetStatus", id=0)
        now = time.perf_counter() - t0
        peak = max(peak, float(st.get("apower") or 0.0))
        log(f"  t={now:4.1f}s  {describe_status(st)}")
        if st.get("errors"):
            log("  errors reported, stopping")
            break
        if st.get("state") in MOVING:
            ran = now
        elif now > 0.3:
            break
        time.sleep(POLL_SECONDS)
    else:
        log("  still moving after the expected time, stopping")
    return peak, ran, st


def run_move(shelly: Shelly, direction: str, duration: float) -> tuple[float, float]:
    """Run the motor for ``duration`` s, watching it; returns (peak watts, seconds it ran).

    The Shelly times the move itself, so a full run lasts ``duration``; a shorter
    one means it stopped early (limit switch, obstruction, error).
    """
    method = "Cover.Open" if direction == "open" else "Cover.Close"
    log(f"{method} for {duration:g} s ...")
    # Marked before sending: if Ctrl+C lands right after the command went out,
    # the motor is running and must still be stopped below.
    shelly.moved = shelly.needs_stop = True
    try:
        shelly.call(method, id=0, duration=duration)
        t0 = time.perf_counter()
        peak, ran, _ = _watch(shelly, t0, t0 + duration + 3.0)
    except KeyboardInterrupt:
        log("  interrupted")
        raise
    finally:
        stop_motor(shelly, duration)
    full = ran >= duration - 3 * POLL_SECONDS
    ran = duration if full else round(ran + POLL_SECONDS / 2, 1)  # polls see the stop late by ~half a poll
    st = shelly.call("Cover.GetStatus", id=0)
    log(f"After the move: {describe_status(st)}")
    log(f"  ran {'the full' if full else 'only'} {ran:g} s of {duration:g} s, peak power {peak:.1f} W")
    if not full:
        log(f"  stopped early: a limit switch was probably reached (state={st.get('state')})")
    if peak < MIN_RUNNING_WATTS:
        log("  WARNING: almost no power drawn, so the motor probably did not turn. Check:")
        log("    - wiring: motor up/down wires on O1/O2, neutral on N, L and N supplied to the Shelly")
        log("    - the motor's limit switch for this direction may already be reached: try the other way")
        log("    - a motor with a built-in radio receiver (3 wires: L, N, PE) cannot be driven")
        log("      by the 2PM's two outputs; it needs a 4-wire (two direction wires) motor")
        ran = 0.0
    else:
        log("  the motor drew power, so it was driven. Check by eye that it turned the expected way;")
        log("  if 'open' rolls the wrong way, swap the motor's O1/O2 wires or set 'swap' in the Shelly app")
    return peak, ran


def go_to_position(shelly: Shelly, pos: int) -> dict:
    log(f"Cover.GoToPosition pos={pos}% (exact return, calibrated) ...")
    shelly.moved = shelly.needs_stop = True  # before sending, as in run_move
    try:
        shelly.call("Cover.GoToPosition", id=0, pos=pos)
        t0 = time.perf_counter()
        _watch(shelly, t0, t0 + MAX_DURATION + 3.0)
    finally:
        stop_motor(shelly)  # GoToPosition has no duration: never leave it running
    return shelly.call("Cover.GetStatus", id=0)


def round_trip(shelly: Shelly, direction: str, duration: float, cycle: int, cycles: int) -> bool:
    """Forward then back so the cover ends where it started; True if it did."""
    other = "close" if direction == "open" else "open"
    start = shelly.call("Cover.GetStatus", id=0)
    log("")
    log(f"=== Round trip {cycle}/{cycles}: {direction} {duration:g} s, then {other} back ===")
    log(f"Start: {describe_status(start)}")

    _, ran = run_move(shelly, direction, duration)
    if ran == 0:
        log("Nothing moved, so there is nothing to undo.")
        end = shelly.call("Cover.GetStatus", id=0)
    else:
        time.sleep(1.0)  # let the motor come to rest before reversing
        log(f"Returning: {other} for the {ran:g} s it actually ran")
        run_move(shelly, other, ran)
        end = shelly.call("Cover.GetStatus", id=0)
        if start.get("pos_control") and start.get("current_pos") is not None:
            off = abs((end.get("current_pos") or 0) - start["current_pos"])
            if off > POSITION_TOLERANCE:
                time.sleep(1.0)
                end = go_to_position(shelly, int(start["current_pos"]))

    log("Before vs after:")
    same = True
    for key in ("state", "current_pos"):
        a, b = start.get(key), end.get(key)
        if key == "current_pos" and a is not None and b is not None:
            ok = abs(a - b) <= POSITION_TOLERANCE
        else:
            ok = a == b
        same &= ok
        log(f"  {key:12} {str(a):>8} -> {str(b):<8} {'same' if ok else 'DIFFERENT'}")
    if not start.get("pos_control"):
        log("  The Shelly is not calibrated, so it cannot report the position: the return is by time.")
        log("  Check by eye; for an exact return, calibrate (Shelly app > Calibrate, or Cover.Calibrate).")
    log("Round trip OK: back in the start state." if same else
        "Round trip did NOT end in the start state (see above).")
    return same


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--host", required=True,
                        help="Shelly IP address or hostname[:port], e.g. 192.168.1.50 or 127.0.0.1:8080 "
                             "(required, no default)")
    parser.add_argument("--password", default=os.environ.get("SHELLY_PASSWORD"),
                        help="Shelly password if authentication is on, user 'admin' "
                             "(default: $SHELLY_PASSWORD, else none)")
    parser.add_argument("--timeout", type=float, default=3.0, help="seconds per request (default: 3)")
    parser.add_argument("--check-auth", action="store_true", default=False,
                        help="only check whether a password is set and whether --password is right "
                             "(default: off)")
    parser.add_argument("--move", choices=["open", "close"], default=None,
                        help="run the motor this way (default: none, read-only)")
    parser.add_argument("--duration", type=float, default=1.0,
                        help=f"seconds to run, 0.1 to {MAX_DURATION:g} (default: 1)")
    parser.add_argument("--back", action="store_true", default=False,
                        help="round trip: after the move, return to the start state (default: off)")
    parser.add_argument("--cycles", type=int, default=1,
                        help="round trips to make with --back, 1 to 10 (default: 1)")
    parser.add_argument("--stop", action="store_true", default=False,
                        help="emergency stop: only send Cover.Stop (default: off)")
    parser.add_argument("--yes", action="store_true", default=False,
                        help="do not ask for confirmation before moving (default: off, asks)")
    parser.add_argument("--set-cover-profile", action="store_true", default=False,
                        help="switch the Shelly to the cover profile if needed; reboots it (default: off)")
    args = parser.parse_args()
    install_stop_signals()
    if not 0.1 <= args.duration <= MAX_DURATION:
        parser.error(f"--duration must be between 0.1 and {MAX_DURATION:g} s")
    if not 1 <= args.cycles <= 10:
        parser.error("--cycles must be between 1 and 10")
    if args.cycles > 1 and not args.back:
        parser.error("--cycles needs --back")

    shelly = Shelly(args.host, args.password, args.timeout)
    try:
        if args.check_auth:
            return check_auth(args.host, args.password, args.timeout)
        if args.stop:
            stopped = stop_motor(shelly)
            log(describe_status(shelly.call("Cover.GetStatus", id=0)))
            return 0 if stopped else 2

        status = check_connection(shelly, args.set_cover_profile)
        if not args.move:
            log("Connection OK. Read-only run; add --move open|close to nudge the motor.")
            return 0

        if status.get("state") in MOVING + ("calibrating",):
            raise ShellyError(f"the cover is {status['state']}; wait or run with --stop first")
        if status.get("errors"):
            raise ShellyError(f"the Shelly reports errors {status['errors']}; fix those first")
        if not args.yes:
            plan = f"{args.move} for {args.duration:g} s"
            if args.back:
                plan += f" and back to the start, {args.cycles}x"
            answer = input(f"Motor will run: {plan}. Keep clear of the umbrella. Type 'yes' to continue: ")
            if answer.strip().lower() != "yes":
                log("Cancelled, nothing moved.")
                return 1

        if not args.back:
            run_move(shelly, args.move, args.duration)
            log("Done.")
            return 0
        results = []
        for cycle in range(1, args.cycles + 1):
            results.append(round_trip(shelly, args.move, args.duration, cycle, args.cycles))
            if cycle < args.cycles:
                time.sleep(1.0)
        log("")
        log(f"Done: {sum(results)}/{len(results)} round trips ended in the start state.")
        return 0 if all(results) else 3
    except ShellyError as exc:
        log(f"FAILED: {exc}")
        if shelly.needs_stop:
            stop_motor(shelly)
        return 2
    except KeyboardInterrupt:
        log("Stopped by user.")
        if shelly.needs_stop:
            stop_motor(shelly)
        if shelly.moved:
            report_final_state(shelly)
        return 130
    except BaseException:
        if shelly.needs_stop:  # unexpected error: never leave the motor running
            stop_motor(shelly)
        raise


def report_final_state(shelly: Shelly) -> None:
    """Where the umbrella ended up after an interrupted run."""
    with uninterruptible():
        try:
            st = shelly.call("Cover.GetStatus", timeout=STOP_TIMEOUT, id=0)
        except ShellyError as exc:
            log(f"Final state unknown ({exc}); check the umbrella by eye.")
            return
    log(f"Final state: {describe_status(st)}")
    if st.get("state") in MOVING:
        log("  WARNING: still reported as moving; run with --stop, or cut the motor's power.")
    else:
        log("  The run was interrupted, so the umbrella may not be back where it started.")
        log("  Move it back with --move open|close (or the Shelly app) when you are ready.")


if __name__ == "__main__":
    sys.exit(main())
