# Test report: `deployment/` at `f942476`

Dimas Maulana Fasya · 26 October 2026
Branch: `dfasya/prototype-automatic-umbrella` → `dimas-improvements`

You said the deployment code had not been tested yet, so I ran it. Everything
below was done without hardware, using `modules/fake_shelly.py` and the
committed evaluation clips.

Three findings, thirteen behaviours that work as intended, and one thing I
could not test. Every claim quotes the line that produced it, with its
timestamp, so you can check any of them against your own run.

Environment: Windows 11, Command Prompt, conda `weather_env`, Python 3.9,
pandas 2.3.3, numpy 2.0.2. `pythermalcomfort` is **not** installed and was not
needed — `clips.csv` already carries `utci`, so nothing had to recompute it.

---

## 1. Model replay

`python -m modules.inference` over all 60 clips reproduces your README exactly:
F1 0.599 against your 0.60, persistence 0.891 against 0.89, `false_open` 0.067
against 0.07, heat recall 0.108 against 0.11. No drift between your machine and
mine.

The per-category table is worth more than the headline, though.

| | rows | open_share | precision | recall | F1 | false_open |
|---|---|---|---|---|---|---|
| **all** model | 11,880 | 0.387 | 0.818 | 0.472 | 0.599 | 0.067 |
| persistence | 11,880 | 0.387 | 0.888 | 0.894 | 0.891 | 0.071 |
| **rain** model | 3,960 | 0.343 | 0.701 | 0.803 | 0.749 | 0.179 |
| persistence | 3,960 | 0.343 | 0.776 | 0.784 | 0.780 | 0.118 |
| **heat** model | 4,320 | 0.543 | 0.930 | 0.108 | 0.194 | 0.010 |
| persistence | 4,320 | 0.543 | 0.910 | 0.918 | 0.914 | 0.107 |
| **heat+rain** model | 900 | 1.000 | 1.000 | 0.920 | 0.958 | 0.000 |
| persistence | 900 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |

Counting positives per category (`rows × open_share`) and the ones the model
caught (`× recall`):

| Category | Should be OPEN | Caught | Missed |
|---|---|---|---|
| rain | 1,358 | 1,090 | 268 |
| heat | 2,346 | 253 | 2,093 |
| heat+rain | 900 | 828 | 72 |
| **total** | **~4,600** | **2,172** | **2,432** |

**Heat accounts for 2,093 of the 2,432 misses — 86 %.** Everything else is
close to parity.

On rain the model actually beats persistence on recall, 0.803 against 0.784. It
pays for that with precision, 0.701 against 0.776, and with false opens, 0.179
against 0.118: roughly six extra false-open points bought two points of rain
recall. For an umbrella that trade is defensible — opening when it turns out
dry is cheap, failing to open in rain is not.

On heat, recall 0.108 with precision 0.930 is the signature of a decision
threshold far too high for that regime, a heat signal that is barely present in
the features, or both.

So "persistence still wins" is true but misleading. The rain path works. The
heat path does not, and it is where essentially all the loss lives.

`heat+rain` cannot be compared at all: `open_share` is 1.000, so persistence
scores a perfect 1.000 by never changing its mind. 900 rows of no information.

One more number: **moves per 24 h, model 16.8 against label 23.1.** The model
moves less than the ground truth. Easy on the motor, but it is the same
under-reaction showing up from a different angle.

---

## 2. Finding 1 — an uncalibrated round trip reports success it cannot verify

**How to reproduce**

```bash
python modules/fake_shelly.py --uncalibrated --close-seconds 15   # terminal 1
python test_shelly.py --host 127.0.0.1:8080 --move open --back    # terminal 2
```

**What `test_shelly.py` printed** (09:22:57)

```
current_pos      None -> None     same
The Shelly is not calibrated, so it cannot report the position: the return is by time.
Check by eye; for an exact return, calibrate (Shelly app > Calibrate, or Cover.Calibrate).
Round trip OK: back in the start state.

Done: 1/1 round trips ended in the start state.
```

**What the simulator printed for the same run**

```
motor starts opening from 50.0%  -> stops at 55.0% (duration over)
motor starts closing from 55.0%  -> stops at 48.3% (duration over)
Simulator stopped. Final position 48.3% (stopped)
```

The cover ended at 48.3 %, not 50 %. One second of closing travels 6.7 % while
one second of opening travels 5 %, so the timed return overshoots.

**Why it matters.** The drift compounds. At `--cycles 3` it is about 5 %; used
daily it is fabric creeping down while the operator is told everything is fine.
And the trigger condition is not exotic — a tubular awning motor is normally
faster in one direction because gravity helps it.

`None -> None  same` is the root of it. Two unknowns are not evidence of
equality.

**Why I think this is an oversight rather than a design choice.** Your Ctrl+C
path handles exactly the same uncertainty correctly (09:38:19):

```
The run was interrupted, so the umbrella may not be back where it started.
Move it back with --move open|close (or the Shelly app) when you are ready.
```

That is the right wording. The uncalibrated path just never got it.

**Suggested fix.** When `calibrated` is false, do not print "back in the start
state" and do not exit 0. Something like "returned by time; position not
verified" with a distinct exit code would match what the script actually knows.

---

## 3. Finding 2 — power overrides position, but only sometimes

Two runs produced the same evidence — the position changed, no power was
measured — and the script did opposite things.

**Run A, `--watts 0`** (09:24:16). The move ran its full duration.

```
ran the full 1 s of 1 s, peak power 0.0 W
WARNING: almost no power drawn, so the motor probably did not turn. Check:
  ...
Nothing moved, so there is nothing to undo.
Before vs after:
  current_pos        50 -> 55       DIFFERENT
Round trip did NOT end in the start state (see above).
```

"Nothing moved" and `50 -> 55 DIFFERENT` are three lines apart. The script
concluded nothing had moved from power alone, skipped the return, and then its
own position check disagreed. The cover was left displaced.

**Run B, external stop mid-move** (09:57:42). The move was cut short.

```
ran only 0.1 s of 3 s, peak power 0.0 W
WARNING: almost no power drawn, so the motor probably did not turn. Check:
  ...
Cover.GoToPosition pos=52% (exact return, calibrated) ...
current_pos        52 -> 52       same
Round trip OK: back in the start state.
```

Same zero-power reading, but here the script recovered the position.

The difference is which branch runs — "move completed" versus "move truncated"
— not what the evidence says. Part of Run A is an artefact of the simulator
(`--watts 0` models a motor that turns while drawing nothing, which is not
physical). But the precedence rule is real: the script had position data in
both cases and only used it in one.

**Suggested fix.** When the Shelly reports a position and that position has
changed, let position decide whether to undo. Keep the power reading as a
warning about wiring, not as the thing that decides whether to move the cover
back.

---

## 4. Finding 3 — a stop from another client is diagnosed as a wiring fault

**How to reproduce.** Start a long run in terminal 2, then from terminal 3:

```bash
python test_shelly.py --host 127.0.0.1:8080 --stop
```

**What terminal 2 printed** (09:57:42)

```
ran only 0.1 s of 3 s, peak power 0.0 W
stopped early: a limit switch was probably reached (state=stopped)
WARNING: almost no power drawn, so the motor probably did not turn. Check:
  - wiring: motor up/down wires on O1/O2, neutral on N, L and N supplied to the Shelly
  - the motor's limit switch for this direction may already be reached: try the other way
  - a motor with a built-in radio receiver (3 wires: L, N, PE) cannot be driven
    by the 2PM's two outputs; it needs a 4-wire (two direction wires) motor
```

None of those applied. The motor was stopped by another client on the network.

The recovery was correct — `GoToPosition` brought it back and the final verdict
was honest. Only the explanation is wrong.

**Why it matters in the field.** Someone standing by the canopy sees something
wrong and hits the emergency stop. The operator at the laptop is then told to
check the mains wiring and suspect the motor type. That is a real afternoon
lost, on an installation that is fine.

**The discriminator is already in your own output.** Compare a genuine limit
switch (09:01:07) with the external stop (09:57:42):

```
09:01:07   state=open       <- limit switch
09:57:42   state=stopped    <- stopped by someone else
```

A limit switch leaves the cover in `open` or `closed`. An external `Cover.Stop`
leaves it in `stopped`. The script prints both values; it just does not use the
difference. Adding a third case — "the move was stopped early with
`state=stopped`, so something else sent a stop" — would cover it.

---

## 5. Documentation: the Windows path

`deployment/README.md` assumes a bash shell in two places:

```bash
python test_shelly.py --host 192.168.1.50 --check-auth --password 'secret'
export SHELLY_PASSWORD='secret'
```

In Command Prompt the single quotes become part of the password, so it is
rejected, and `export` does not exist. The equivalents are:

```bat
python test_shelly.py --host 192.168.1.50 --check-auth --password secret
set SHELLY_PASSWORD=secret
```

I lost a few minutes to the first one. Worth a line in the README, since the
Jetson side is Linux but whoever tests from a laptop may not be.

---

## 6. What works

Thirteen behaviours, all confirmed:

- The read-only check never moves the motor, and says so.
- A symmetric round trip returns to the start and reports `same`.
- A limit switch is handled correctly: `ran only 0.3 s of 1 s`, the return uses
  the time the motor actually turned, and the cover lands back on 98 % rather
  than overshooting to 95 %.
- Asymmetric travel is corrected: close overshoots to 48 %, `GoToPosition`
  brings it to 50 %, verdict `same`.
- Without a password: `FAILED: authentication required or wrong password`, no
  traceback, and the message names the fix.
- `--check-auth --password secret` confirms the password is correct.
- Uncalibrated: no `GoToPosition` is attempted, the return is by time, and the
  caveat is printed (the verdict line is Finding 1).
- A motor drawing no power is caught, with a warning that names the radio-motor
  case explicitly.
- Ctrl+C stops the motor before exiting, prints the final state, warns that the
  cover may not be back, and exits 130. Confirmed twice, mid-open and
  mid-close.
- A second simulator on a busy port is refused with a message that names the
  remedy.
- Pasting a command into the `yes` prompt cancels safely: "Cancelled, nothing
  moved."
- `--stop` sent while nothing is moving is a harmless no-op. That matters:
  in a real emergency someone will press it repeatedly.
- `--stop` sent mid-move is detected (`ran only 0.1 s of 3 s`) and recovered
  from, and the run continues to `Done: 10/10 round trips ended in the start
  state`.

The duration is also sent with every move command, which I checked in the
simulator log:

```
RPC Cover.Open {'id': '0', 'duration': '3.0'}
```

So the Shelly stops the motor by itself if the Jetson dies mid-move. That is
the right design and it is worth keeping prominent in the README.

---

## 7. Not tested

- Anything with real hardware. The Shelly has not been bought yet.
- `extract_evaluation.py`, which needs `labeled_dataset/` regenerated.
- Whether `test_shelly.py` notices an external stop *during* the per-0.1 s
  status polling rather than at the end of the move. The observed behaviour is
  consistent with detection at the end.

---

## 8. Three things missing from `deployment/`

Not defects, just gaps I checked for while reading:

**No wind rule.** Wind reaches the decision only through UTCI, where it cools,
so higher wind makes a heat opening less likely. For rain it does nothing at
all: `p_rain` is 1 whenever the detector is wet, whatever the wind. A
thunderstorm is wet and windy at once. I have written
`modules/model/wind_safety.py` on my branch — it sits after `apply_hysteresis`,
can only force CLOSED, and does not touch `hmm.py`.

**No S2 duty-cycle protection.** The only limit is the Shelly's own
`maxtime_open` / `maxtime_close`, which `test_shelly.py` reads and reports. The
SL45RM-50/12 is rated S2 4 min; there is no cooldown and no minimum gap between
commands anywhere in the code.

**Nothing connects `Decision` to the motor.** `Cover.Open` and `Cover.Close`
appear only in `fake_shelly.py` and `test_shelly.py` — the simulator and the
manual test tool. `inference.py` produces `Decision` objects and stops there. I
have `canopy_driver.py` for this: change detection, `RUN_LIMIT_S=240`,
`COOLDOWN_S=600`, `MIN_COMMAND_GAP_S=60`.

Noticed while testing, worth repeating: one asymmetric round trip issues three
motor commands in four seconds (`Open`, `Close`, `GoToPosition`). Fine for a
test, but it is exactly what the command spacing above exists to prevent in
normal running.

---

## 9. One question for you

`deployment/README.md` says:

> A tubular motor with a built-in radio receiver only has L, N and PE and
> cannot be driven by the 2PM's outputs; check the motor's label or wires.

Our motor's nameplate reads **"Radio with manual tubular motor"**, model
SL45RM-50/12. That is the case your own warning describes.

I am going to count the conductors in the motor cable this week. Four wires and
the plan stands; three and the Shelly 2PM cannot drive it at all and we need a
different approach. I will report back either way.

---

Reproduced on Windows 11, `weather_env`, against `deployment/` at commit
`f942476`. Timestamps refer to the terminal session of 26 October 2026, from
08:53 to 09:59. Model figures in section 1 come from
`python -m modules.inference` over the committed 60 clips; figures attributed to
`deployment/README.md` are quoted from it.
