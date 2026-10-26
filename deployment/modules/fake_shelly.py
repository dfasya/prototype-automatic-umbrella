"""Simulated Shelly 2PM Gen4 with a tubular motor, to test without hardware.

Answers the same HTTP RPC calls as the real device (Shelly.GetDeviceInfo,
Sys.GetConfig, Cover.GetConfig/GetStatus/Open/Close/Stop/GoToPosition) and
simulates the motor: position 0 (closed) to 100 (open), travel time per
direction, limit switches at both ends, power drawn only while turning.

    python modules/fake_shelly.py                          # http://127.0.0.1:8080, starts at 50 %
    python modules/fake_shelly.py --password s3cret        # with authentication (user admin)
    python modules/fake_shelly.py --start-pos 98           # near the open limit: a move hits the switch
    python modules/fake_shelly.py --uncalibrated           # no position reporting, like a new device
    python modules/fake_shelly.py --open-seconds 20 --close-seconds 18   # faster down than up

Then, in another terminal:

    python test_shelly.py --host 127.0.0.1:8080 --move open --back

Every RPC call and every motor start/stop is printed.

Stopping: press Ctrl+C (or close the terminal / ``kill`` it). The simulated
motor is stopped where it is, like a power cut, its final position is
printed, and the port is released, so the simulator can be restarted at once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

DEVICE_ID = "shelly2pmg4-000000fake00"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class Motor:
    """Position over time; the state is advanced whenever it is read."""

    def __init__(self, pos: float, open_s: float, close_s: float, watts: float, calibrated: bool):
        self.pos, self.open_s, self.close_s, self.watts = pos, open_s, close_s, watts
        self.calibrated = calibrated
        self.state = "open" if pos >= 100 else "closed" if pos <= 0 else "stopped"
        self.direction = 0          # +1 opening, -1 closing
        self.until = 0.0            # when the commanded duration ends
        self.target = None          # GoToPosition target
        self.last = time.monotonic()
        self.last_direction = None
        self.lock = threading.Lock()

    def _advance(self) -> None:
        now = time.monotonic()
        if self.direction:
            end = min(now, self.until)
            rate = 100 / (self.open_s if self.direction > 0 else self.close_s)  # % per second
            new = self.pos + self.direction * rate * max(end - self.last, 0.0)
            if self.target is not None and (new - self.target) * self.direction >= 0:
                self._halt(self.target, "stopped", "reached target")
            elif self.direction > 0 and new >= 100:
                self._halt(100.0, "open", "open limit switch")
            elif self.direction < 0 and new <= 0:
                self._halt(0.0, "closed", "close limit switch")
            else:
                self.pos = new
                if now >= self.until:
                    self._halt(new, "stopped", "duration over")
        self.last = now

    def _halt(self, pos: float, state: str, why: str) -> None:
        self.pos, self.state, self.direction, self.target = pos, state, 0, None
        print(f"    motor stops at {self.pos:.1f}% ({why})", flush=True)

    def move(self, direction: int, duration: float | None, target: float | None = None) -> None:
        with self.lock:
            self._advance()
            if (direction > 0 and self.pos >= 100) or (direction < 0 and self.pos <= 0):
                print("    already at the limit switch: motor does not turn", flush=True)
                return
            full = self.open_s if direction > 0 else self.close_s
            self.direction, self.target = direction, target
            self.state = "opening" if direction > 0 else "closing"
            self.until = time.monotonic() + (duration if duration is not None else full * 1.1)
            self.last_direction = "open" if direction > 0 else "close"
            print(f"    motor starts {self.state} from {self.pos:.1f}%", flush=True)

    def stop(self, why: str = "Cover.Stop") -> None:
        with self.lock:
            self._advance()
            if self.direction:
                self._halt(self.pos, "stopped", why)

    def status(self) -> dict:
        with self.lock:
            self._advance()
            moving = bool(self.direction)
            return {
                "id": 0, "source": "http", "state": self.state,
                "apower": self.watts if moving else 0.0, "voltage": 230.1,
                "current": round(self.watts / 230, 3) if moving else 0.0, "pf": 0.95 if moving else 0.0,
                "freq": 50.0, "temperature": {"tC": 38.5, "tF": 101.3},
                "pos_control": self.calibrated,
                "current_pos": round(self.pos) if self.calibrated else None,
                "last_direction": self.last_direction, "errors": [],
            }


def make_handler(motor: Motor, password: str | None):
    nonce = str(int(time.time()))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, code: int, body, headers: dict | None = None) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def authorized(self) -> bool:
            header = self.headers.get("Authorization", "")
            if not header.startswith("Digest "):
                return False
            fields = {k.strip(): v.strip().strip('"') for k, v in
                      (part.split("=", 1) for part in header[7:].split(","))}
            ha1 = sha256(f"admin:{DEVICE_ID}:{password}")
            ha2 = sha256(f"GET:{fields.get('uri')}")
            expected = sha256(f"{ha1}:{fields.get('nonce')}:{fields.get('nc')}:"
                              f"{fields.get('cnonce')}:auth:{ha2}")
            return fields.get("algorithm") == "SHA-256" and fields.get("response") == expected

        def do_GET(self):
            url = urlparse(self.path)
            method = url.path.rsplit("/", 1)[-1]
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            if password and method != "Shelly.GetDeviceInfo" and not self.authorized():
                challenge = f'Digest qop="auth", realm="{DEVICE_ID}", nonce="{nonce}", algorithm=SHA-256'
                return self.reply(401, {"code": 401, "message": "unauthorized"},
                                  {"WWW-Authenticate": challenge})
            if method != "Cover.GetStatus":
                print(f"RPC {method} {q}", flush=True)
            if method == "Shelly.GetDeviceInfo":
                return self.reply(200, {
                    "id": DEVICE_ID, "mac": "000000FAKE00", "model": "S4SW-002P16EU (simulated)",
                    "gen": 4, "fw_id": "simulated", "ver": "1.6.2", "app": "S2PMG4",
                    "profile": "cover", "auth_en": bool(password),
                    "auth_domain": DEVICE_ID if password else None})
            if method == "Sys.GetConfig":
                return self.reply(200, {"device": {"name": "umbrella (simulated)"}})
            if method == "Cover.GetConfig":
                return self.reply(200, {
                    "id": 0, "name": "umbrella", "in_mode": "dual",
                    "maxtime_open": motor.open_s * 1.1, "maxtime_close": motor.close_s * 1.1,
                    "power_limit": 2800, "obstruction_detection": {"enable": False},
                    "safety_switch": {"enable": False}})
            if method == "Cover.GetStatus":
                return self.reply(200, motor.status())
            if method in ("Cover.Open", "Cover.Close"):
                duration = float(q["duration"]) if "duration" in q else None
                motor.move(1 if method == "Cover.Open" else -1, duration)
                return self.reply(200, None)
            if method == "Cover.GoToPosition":
                if not motor.calibrated:
                    return self.reply(400, {"code": -103, "message": "Cover not calibrated"})
                target = float(q["pos"])
                current = motor.status()["current_pos"]
                if target != current:
                    motor.move(1 if target > current else -1, None, target)
                return self.reply(200, None)
            if method == "Cover.Stop":
                motor.stop()
                return self.reply(200, None)
            return self.reply(404, {"code": 404, "message": f"No handler for {method}"})

    return Handler


class Server(ThreadingHTTPServer):
    daemon_threads = True  # open requests never keep the process alive after Ctrl+C
    # On Windows, SO_REUSEADDR lets a second simulator bind the same port silently,
    # and requests then reach either one. Elsewhere it only allows a quick restart.
    allow_reuse_address = os.name != "nt"


def _as_ctrl_c(signum, frame):
    raise KeyboardInterrupt(f"signal {signum}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--port", type=int, default=8080, help="port on 127.0.0.1 (default: 8080)")
    parser.add_argument("--password", default=None,
                        help="require this password, user admin (default: none, no authentication)")
    parser.add_argument("--start-pos", type=float, default=50.0, help="start position, 0-100 %% (default: 50)")
    parser.add_argument("--open-seconds", type=float, default=20.0,
                        help="full travel time opening, s (default: 20)")
    parser.add_argument("--close-seconds", type=float, default=20.0,
                        help="full travel time closing, s (default: 20)")
    parser.add_argument("--watts", type=float, default=120.0,
                        help="power while turning, 0 = dead motor (default: 120)")
    parser.add_argument("--uncalibrated", action="store_true", default=False,
                        help="do not report the position, like a new device (default: off, calibrated)")
    args = parser.parse_args()

    # kill / closing the terminal / Ctrl+Break: shut down like Ctrl+C instead of dying mid-move.
    for name in ("SIGTERM", "SIGHUP", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _as_ctrl_c)

    motor = Motor(min(max(args.start_pos, 0.0), 100.0), args.open_seconds, args.close_seconds,
                  args.watts, not args.uncalibrated)
    try:
        server = Server(("127.0.0.1", args.port), make_handler(motor, args.password))
    except OSError as exc:
        sys.exit(f"Cannot use port {args.port} ({exc.strerror or exc}). Is another simulator still "
                 f"running? Stop it, or start this one with --port {args.port + 1}.")
    print(f"Simulated Shelly 2PM Gen4 at http://127.0.0.1:{args.port}  "
          f"(position {motor.pos:.0f}%, {'calibrated' if motor.calibrated else 'uncalibrated'}, "
          f"{'password required' if args.password else 'no password'})  Ctrl+C to stop", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\nShutting down ...", flush=True)
    finally:
        motor.stop("simulator stopped: power cut")
        server.server_close()
        print(f"Simulator stopped. Final position {motor.pos:.1f}% ({motor.status()['state']}); "
              f"port {args.port} released.", flush=True)


if __name__ == "__main__":
    main()
