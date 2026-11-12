"""ML Forecasting Pipeline by Matija Ivanovic™ — portable setup wizard.

A standard Windows Installer-style wizard (Welcome, Run Options, Ready,
Progress, Completed) that runs the whole Budva forecast pipeline from the
drive this file lives on: checks/repairs the bundled Python runtime, fetches
fresh station and model data, runs the rain backtest, trains every model and
saves the result to <drive>/models/<stamp>/. Nothing is installed on the host
PC; every path derives from this file's location (<root>/app/gui/).

    pythonw pipeline_gui.py [--dry-run] [--page N] [--demo] [--screenshot PATH]
"""
import ctypes
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

PRODUCT = "ML Forecasting Pipeline by Matija Ivanovic™"
SHORT = "ML Forecasting Pipeline"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.environ.get("MLFP_ROOT") or os.path.dirname(os.path.dirname(HERE)))
REPO = os.path.join(ROOT, "app", "repo")
ASSETS = os.path.join(HERE, "assets")
PYTHON = os.path.join(ROOT, "runtime", "python", "python.exe")
if not os.path.exists(PYTHON):          # development checkout
    PYTHON = sys.executable.replace("pythonw.exe", "python.exe")
LOGS = os.path.join(ROOT, "logs")
WORK_TMP = os.path.join(ROOT, "work", "tmp")
MODELS_OUT = os.path.join(ROOT, "models")
TASKS = os.path.join(HERE, "pipeline_tasks.py")
FACE = "SystemButtonFace"

STAGES = [
    {"key": "prereq", "name": "Checking prerequisites", "weight": 2, "critical": True,
     "cmds": [[TASKS, "prereq"], [os.path.join(REPO, "forecast_48h_v3.py"), "--check-device"]]},
    {"key": "station", "name": "Downloading station data", "weight": 4, "critical": False,
     "cmds": [[os.path.join(REPO, "wu_update.py")]]},
    {"key": "models", "name": "Downloading model data", "weight": 12, "critical": False,
     "cmds": [[os.path.join(REPO, "update_model_csvs.py")],
              [os.path.join(REPO, "fetch_previous_runs.py")],
              [os.path.join(REPO, "fetch_aux_history.py")]]},
    {"key": "backtest", "name": "Running the rain feature backtest", "weight": 18, "critical": False,
     "cmds": [[os.path.join(REPO, "analysis_output", "rain_v3_backtest.py")]]},
    {"key": "train", "name": "Training models", "weight": 150, "critical": True,
     "cmds": [[os.path.join(REPO, "forecast_48h_v3.py")]]},
    {"key": "verify", "name": "Running tests", "weight": 1, "critical": False,
     "cmds": [[os.path.join(REPO, "test_forecast_core.py")]]},
    {"key": "export", "name": "Saving the model to this drive", "weight": 1, "critical": True,
     "cmds": [[TASKS, "export"]]},
]
PROGRESS_RE = re.compile(r"^##(PROGRESS|TIMING) (\{.*\})\s*$")
SAVED_RE = re.compile(r"^Saved model: (.+?) \(")


def _fmt_dur(sec):
    sec = int(max(sec, 0))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class Runner:
    """Runs the stages in a worker thread and reports through a queue."""

    def __init__(self, on_line):
        self.q = queue.Queue()
        self.on_line = on_line
        self.proc = None
        self.cancelled = False
        self.running = False
        self.states = {s["key"]: "pending" for s in STAGES}
        self.stage_frac = {s["key"]: 0.0 for s in STAGES}
        self.current = None
        self.run_started = None
        self.train_step = None
        self.pipe_step = None
        self.saved_to = None
        self.tests_passed = None
        self.device = "auto"
        self.trials = "30"
        self.dry_run = False
        self.run_log_path = None

    # -- environment for the child processes
    def env(self):
        env = os.environ.copy()
        for k in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "VIRTUAL_ENV", "CONDA_PREFIX"):
            env.pop(k, None)
        os.makedirs(WORK_TMP, exist_ok=True)
        env.update({
            "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1", "PYTHONNOUSERSITE": "1",
            "PYTHONIOENCODING": "utf-8", "TEMP": WORK_TMP, "TMP": WORK_TMP,
            "MPLCONFIGDIR": os.path.join(ROOT, "work", "mpl"),
            "FC_PROGRESS": "1", "FC_DEVICE": self.device, "FC_TRIALS": self.trials,
            "MLFP_ROOT": ROOT, "MLFP_RUN_LOG": self.run_log_path or "",
        })
        if self.tests_passed is not None:
            env["MLFP_TESTS_PASSED"] = "yes" if self.tests_passed else "no"
        cfg = os.path.join(ROOT, "config.json")
        if os.path.exists(cfg):
            try:
                with open(cfg, encoding="utf-8") as f:
                    for k, v in json.load(f).items():
                        if isinstance(v, str) and v:
                            env[k] = v
            except (OSError, ValueError) as exc:
                self.q.put(("line", f"config.json ignored ({exc})"))
        return env

    def start(self):
        os.makedirs(LOGS, exist_ok=True)
        self.run_log_path = os.path.join(LOGS, time.strftime("run_%Y%m%d_%H%M%S.log"))
        self.running = True
        self.cancelled = False
        self.run_started = time.time()
        threading.Thread(target=self._worker, daemon=True).start()

    def cancel(self):
        self.cancelled = True
        p = self.proc
        if p and p.poll() is None:
            subprocess.call(["taskkill", "/PID", str(p.pid), "/T", "/F"],
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _worker(self):
        for st in STAGES:
            key = st["key"]
            if self.cancelled:
                self.q.put(("state", key, "skipped"))
                continue
            cmds = st["cmds"]
            if self.dry_run and key in ("station", "models", "backtest", "train"):
                cmds = [[os.path.join(REPO, "forecast_48h_v3.py"), "--check-device"]]
            self.q.put(("stage", key))
            t0 = time.time()
            ok_all = True
            for cmd in cmds:
                rc = self._run([PYTHON, "-u", *cmd])
                if self.cancelled:
                    break
                if rc != 0:
                    ok_all = False
                    if st["critical"] and key != "train":
                        break
            if self.cancelled:
                state = "failed"
            elif ok_all:
                state = "done"
            elif key == "train" and self._train_saved_since(t0):
                state = "warn"   # models saved; only the live forecast at the end failed
                self.q.put(("line", "Models were trained and saved; only the live forecast failed."))
            elif st["critical"]:
                state = "failed"
            else:
                state = "warn"
            if key == "verify":
                self.tests_passed = ok_all
            self.q.put(("state", key, state))
            if state == "failed":
                self.cancelled = True
        self.q.put(("finished",))

    def _train_saved_since(self, t0):
        path = os.path.join(REPO, "trained_models_v2", "timings.json")
        return os.path.exists(path) and os.path.getmtime(path) >= t0

    def _run(self, cmd):
        shown = " ".join(os.path.relpath(c, ROOT) if os.path.isabs(c) and c.startswith(ROOT) else c
                         for c in cmd[2:])
        self.q.put(("line", f"> python {shown}"))
        try:
            self.proc = subprocess.Popen(
                cmd, cwd=REPO, env=self.env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace", bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError as exc:
            self.q.put(("line", f"Could not start: {exc}"))
            return 127
        for line in self.proc.stdout:
            self.q.put(("raw", line.rstrip("\r\n")))
        rc = self.proc.wait()
        self.q.put(("line", f"  exit code {rc}"))
        return rc

    # -- progress model
    def overall_frac(self):
        total = sum(s["weight"] for s in STAGES)
        acc = 0.0
        for s in STAGES:
            st = self.states[s["key"]]
            if st in ("done", "warn", "failed", "skipped"):
                acc += s["weight"]
            elif st == "running":
                acc += s["weight"] * self.stage_frac[s["key"]]
        return acc / total

    def eta(self):
        if not self.running or not self.run_started:
            return None
        frac = self.overall_frac()
        spent = time.time() - self.run_started
        if frac < 0.03 or spent < 60:
            return None
        return spent * (1 - frac) / frac

    def apply_progress(self, d):
        stage, step, total, label = d.get("stage"), d.get("step"), d.get("total"), d.get("label")
        if not step or not total:
            return
        if stage == "pipeline":
            self.pipe_step = (step, total, label)
            if self.current == "train":
                frac = 0.03 * min(step, 3) / 3 if step <= 3 else 0.93 + 0.07 * (step - 4) / 3
                self.stage_frac["train"] = max(self.stage_frac["train"], frac)
        elif stage == "train":
            self.train_step = (step, total, label)
            if self.current == "train":
                self.stage_frac["train"] = max(self.stage_frac["train"],
                                               0.03 + 0.90 * (step - 1) / total)
        elif stage == "backtest" and self.current == "backtest":
            self.stage_frac["backtest"] = (step - 1) / total


class Wizard:
    def __init__(self, root, args):
        self.root = root
        self.args = args
        self.scale = max(1.0, root.winfo_fpixels("1i") / 96.0)
        self.runner = Runner(self._log)
        self.runner.dry_run = "--dry-run" in args
        self.page = 0
        self.finish_open = tk.BooleanVar(value=True)
        self.device_var = tk.StringVar(value="auto")
        self.trials_var = tk.StringVar(value="30")
        self.dry_var = tk.BooleanVar(value=self.runner.dry_run)
        self.status_var = tk.StringVar(value="")
        self.detail_var = tk.StringVar(value="")
        self.time_var = tk.StringVar(value="")
        self.result_ok = None

        u = self.u
        root.title(f"{PRODUCT} Setup")
        root.configure(bg=FACE)
        root.resizable(False, False)
        try:
            root.iconbitmap(os.path.join(ASSETS, "app.ico"))
        except tk.TclError:
            pass
        w, h = u(500), u(390)
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 3}")

        style = ttk.Style(root)
        for theme in ("vista", "xpnative", "winnative"):
            if theme in style.theme_names():
                style.theme_use(theme)
                break
        self.f_normal = ("Segoe UI", 9)
        self.f_bold = ("Segoe UI", 9, "bold")
        self.f_welcome = ("Segoe UI", 12, "bold")
        style.configure("White.TCheckbutton", background="white")
        style.configure("TButton", padding=(u(4), u(1)))

        self._make_images()
        self.body = tk.Frame(root, bg=FACE)
        self.body.place(x=0, y=0, width=w, height=h - u(49))
        ttk.Separator(root, orient="horizontal").place(x=0, y=h - u(49), width=w)
        bw, bh = u(75), u(23)
        by = h - u(36)
        self.btn_back = ttk.Button(root, text="< Back", command=self.back)
        self.btn_next = ttk.Button(root, text="Next >", command=self.next)
        self.btn_cancel = ttk.Button(root, text="Cancel", command=self.cancel)
        self.btn_back.place(x=w - u(11) - bw * 3 - u(12), y=by, width=bw, height=bh)
        self.btn_next.place(x=w - u(11) - bw * 2 - u(12), y=by, width=bw, height=bh)
        self.btn_cancel.place(x=w - u(11) - bw, y=by, width=bw, height=bh)
        root.bind("<Return>", lambda e: self.btn_next.instate(["!disabled"]) and self.next())
        root.bind("<Escape>", lambda e: self.cancel())
        root.protocol("WM_DELETE_WINDOW", self.cancel)

        self.log_lines = []
        self.pages = [self._page_welcome, self._page_options, self._page_ready,
                      self._page_progress, self._page_finish]
        start_page = 0
        if "--page" in args:
            start_page = int(args[args.index("--page") + 1])
        if "--demo" in args:
            self._demo()
        self.show(start_page)
        self.root.after(80, self._poll)
        self.root.after(1000, self._tick)
        if "--screenshot" in args:
            path = args[args.index("--screenshot") + 1]
            self.root.after(1500, lambda: self._screenshot(path))
        # unattended test run: --autorun [--finish-shot PATH]
        self.finish_shot = (args[args.index("--finish-shot") + 1]
                            if "--finish-shot" in args else None)
        if "--autorun" in args:
            self.root.after(800, lambda: (self.show(2), self.next()))

    def u(self, v):
        return int(round(v * self.scale))

    # ------------------------------------------------------------ artwork
    def _make_images(self):
        u = self.u
        src = Image.open(os.path.join(ASSETS, "pressure_map.png")).convert("RGB")
        W, H = src.size
        # Dialog bitmap (left of Welcome/Completed): the low and its fronts.
        dw, dh = u(164), u(390) - u(49)
        cw = int(H * dw / dh)
        cx = int(W * 0.62)
        side = src.crop((cx - cw // 2, 0, cx + cw // 2, H)).resize((dw, dh), Image.LANCZOS)
        self.img_side = ImageTk.PhotoImage(side)
        # Banner bitmap (top of interior pages): map strip fading into white.
        bw, bh = u(500), u(58)
        strip_w = u(190)
        ch = int(W * 0.30 * bh / strip_w)
        cy = int(H * (1 - 0.42))
        strip = src.crop((int(W * 0.50), cy - ch // 2, int(W * 0.80), cy + ch // 2)).resize(
            (strip_w, bh), Image.LANCZOS)
        banner = Image.new("RGB", (bw, bh), "white")
        mask = Image.linear_gradient("L").rotate(90).resize((strip_w, bh))
        mask = mask.point(lambda v: min(255, int(v * 1.6)))
        banner.paste(strip, (bw - strip_w, 0), mask.transpose(Image.FLIP_LEFT_RIGHT))
        self.img_banner = ImageTk.PhotoImage(banner)

    # ------------------------------------------------------------ page plumbing
    def show(self, idx):
        self.page = idx
        for child in self.body.winfo_children():
            child.destroy()
        self.pages[idx]()
        self._buttons()

    def _buttons(self):
        p = self.page
        running = self.runner.running
        self.btn_back.state(["!disabled"] if p in (1, 2) else ["disabled"])
        if p == 2:
            self.btn_next.configure(text="Run")
        elif p == 4:
            self.btn_next.configure(text="Finish")
        else:
            self.btn_next.configure(text="Next >")
        self.btn_next.state(["disabled"] if p == 3 else ["!disabled"])
        self.btn_cancel.state(["disabled"] if p == 4 or (p == 3 and not running) else ["!disabled"])
        self.btn_next.focus_set()

    def _white_page_with_side(self):
        u = self.u
        f = tk.Frame(self.body, bg="white")
        f.place(x=0, y=0, relwidth=1, relheight=1)
        tk.Label(f, image=self.img_side, bd=0, bg="white").place(x=0, y=0)
        return f, u(164) + u(16)

    def _interior(self, title, subtitle):
        u = self.u
        banner = tk.Frame(self.body, bg="white")
        banner.place(x=0, y=0, relwidth=1, height=u(58))
        tk.Label(banner, image=self.img_banner, bd=0, bg="white").place(x=0, y=0)
        tk.Label(banner, text=title, font=self.f_bold, bg="white", anchor="w").place(
            x=u(15), y=u(8))
        tk.Label(banner, text=subtitle, font=self.f_normal, bg="white", anchor="w").place(
            x=u(25), y=u(28))
        ttk.Separator(self.body, orient="horizontal").place(x=0, y=u(58), relwidth=1)
        content = tk.Frame(self.body, bg=FACE)
        content.place(x=u(20), y=u(70), relwidth=1, width=-u(40), relheight=1, height=-u(78))
        return content

    def _text(self, parent, text, bg, wrap, font=None, **place):
        lbl = tk.Label(parent, text=text, font=font or self.f_normal, bg=bg, justify="left",
                       anchor="nw", wraplength=wrap)
        lbl.place(**place)
        return lbl

    # ------------------------------------------------------------ pages
    def _page_welcome(self):
        u = self.u
        f, x = self._white_page_with_side()
        wrap = u(500) - x - u(20)
        self._text(f, f"Welcome to the {SHORT} Setup Wizard", "white", wrap,
                   self.f_welcome, x=x, y=u(16))
        self._text(f, "by Matija Ivanovic™", "white", wrap, x=x, y=u(64))
        self._text(f, ("The Setup Wizard prepares this computer, downloads fresh weather "
                       "data for Budva, trains the forecast models and saves the new model "
                       "to this drive.\n\nNothing is installed on this computer. Python and "
                       "every package run from the drive, and all files are written there.\n\n"
                       "Click Next to continue, or Cancel to exit the Setup Wizard."),
                   "white", wrap, x=x, y=u(100))

    def _page_options(self):
        u = self.u
        c = self._interior("Run Options", "Choose how the models are trained.")
        dev = ttk.LabelFrame(c, text="Training device")
        dev.place(x=0, y=0, relwidth=1, height=u(84))
        for i, (txt, val) in enumerate((
                ("Automatic: use the NVIDIA GPU when one is available", "auto"),
                ("GPU only: stop if no usable GPU is found", "cuda"),
                ("CPU only", "cpu"))):
            ttk.Radiobutton(dev, text=txt, value=val, variable=self.device_var).place(
                x=u(10), y=u(2) + i * u(20))
        tri = ttk.LabelFrame(c, text="Optuna trials per target")
        tri.place(x=0, y=u(92), relwidth=1, height=u(46))
        for i, (txt, val) in enumerate((("15 (quick)", "15"), ("30 (recommended)", "30"),
                                        ("50 (thorough)", "50"))):
            ttk.Radiobutton(tri, text=txt, value=val, variable=self.trials_var).place(
                x=u(10) + i * u(140), y=u(2))
        ttk.Checkbutton(c, text="Dry run: check the device instead of downloading data and training",
                        variable=self.dry_var).place(x=0, y=u(146))
        self._text(c, "The new model is saved to:", FACE, u(440), x=0, y=u(172))
        e = ttk.Entry(c)
        e.insert(0, MODELS_OUT + os.sep)
        e.state(["readonly"])
        e.place(x=0, y=u(190), relwidth=1, height=u(22))

    def _page_ready(self):
        u = self.u
        c = self._interior(f"Ready to Run {SHORT}", "The Setup Wizard is ready to begin.")
        dev = {"auto": "automatic (GPU when available)", "cuda": "GPU only", "cpu": "CPU only"}
        mode = "dry run (device check only)" if self.dry_var.get() else "full run"
        self._text(c, ("Click Run to begin. Click Back to review or change the settings, or "
                       "Cancel to exit the wizard."), FACE, u(450), x=0, y=0)
        summary = (f"Mode:\t\t{mode}\n"
                   f"Device:\t\t{dev[self.device_var.get()]}\n"
                   f"Optuna trials:\t{self.trials_var.get()} per target\n"
                   f"Model folder:\t{MODELS_OUT}")
        self._text(c, summary, FACE, u(450), x=0, y=u(40))
        self._text(c, ("A full run takes about 2 to 3 hours on a GPU and several hours on a "
                       "CPU. Keep this computer connected to power and the internet; it will "
                       "not go to sleep while the wizard runs."), FACE, u(450), x=0, y=u(130))

    def _page_progress(self):
        u = self.u
        c = self._interior(f"Running {SHORT}",
                           "Please wait while the wizard trains the models.")
        self._text(c, "Status:", FACE, u(80), x=0, y=0)
        tk.Label(c, textvariable=self.status_var, font=self.f_normal, bg=FACE, anchor="w").place(
            x=u(48), y=0, relwidth=1, width=-u(48))
        self.pbar = ttk.Progressbar(c, orient="horizontal", mode="determinate", maximum=1000)
        self.pbar.place(x=0, y=u(22), relwidth=1, height=u(17))
        tk.Label(c, textvariable=self.detail_var, font=self.f_normal, bg=FACE, anchor="w").place(
            x=0, y=u(44), relwidth=1)
        tk.Label(c, textvariable=self.time_var, font=self.f_normal, bg=FACE, anchor="w",
                 fg="#444444").place(x=0, y=u(62), relwidth=1)
        box = tk.Frame(c, bg="#A0A0A0", bd=0)
        box.place(x=0, y=u(86), relwidth=1, relheight=1, height=-u(86))
        self.logw = tk.Text(box, font=("Consolas", 8), bg="white", fg="#202020", relief="flat",
                            bd=0, wrap="none", padx=u(4), pady=u(2), highlightthickness=0)
        sb = ttk.Scrollbar(box, orient="vertical", command=self.logw.yview)
        self.logw.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y", padx=(0, 1), pady=1)
        self.logw.pack(side="left", fill="both", expand=True, padx=(1, 0), pady=1)
        self.logw.tag_configure("err", foreground="#B00020")
        for line, tag in self.log_lines[-400:]:
            self.logw.insert("end", line + "\n", tag)
        self.logw.see("end")
        self.logw.configure(state="disabled")
        self._refresh_progress()

    def _page_finish(self):
        u = self.u
        f, x = self._white_page_with_side()
        wrap = u(500) - x - u(20)
        if self.result_ok:
            title = f"Completed the {SHORT} Setup Wizard"
            where = self.runner.saved_to or MODELS_OUT
            body = (f"The models were trained and saved to this drive:\n\n{where}\n\n"
                    f"MANIFEST.json in that folder lists the metrics, the data range and "
                    f"the timings. The full log is in the logs folder.\n\nClick Finish to "
                    f"exit the Setup Wizard.")
            warn = [s["name"] for s in STAGES if self.runner.states[s["key"]] == "warn"]
            if warn:
                body += ("\n\nCompleted with warnings: " + ", ".join(warn).lower() +
                         ". See the log for details.")
        else:
            title = f"{SHORT} Setup Wizard ended prematurely"
            failed = next((s["name"] for s in STAGES if self.runner.states[s["key"]] == "failed"),
                          None)
            body = ("The run was cancelled or stopped before the model was saved. "
                    + (f"It stopped while {failed.lower()}. " if failed else "")
                    + f"Nothing on this computer was changed.\n\nThe log is in:\n"
                    + f"{self.runner.run_log_path or LOGS}\n\nClick Finish to exit the Setup Wizard.")
        self._text(f, title, "white", wrap, self.f_welcome, x=x, y=u(16))
        self._text(f, body, "white", wrap, x=x, y=u(80))
        label = "Open the models folder" if self.result_ok else "Open the log"
        ttk.Checkbutton(f, text=label, variable=self.finish_open,
                        style="White.TCheckbutton").place(x=x, y=u(282))

    # ------------------------------------------------------------ navigation
    def back(self):
        if self.page in (1, 2):
            self.show(self.page - 1)

    def next(self):
        if self.page in (0, 1):
            self.show(self.page + 1)
        elif self.page == 2:
            r = self.runner
            r.device, r.trials, r.dry_run = (self.device_var.get(), self.trials_var.get(),
                                             self.dry_var.get())
            self._keep_awake(True)
            r.start()
            self._log(f"{PRODUCT}", None)
            self._log(f"Folder: {ROOT}   Python: {PYTHON}", None)
            self._log(f"Device: {r.device}   Optuna trials: {r.trials}   "
                      f"{'dry run' if r.dry_run else 'full run'}", None)
            self.show(3)
        elif self.page == 4:
            if self.finish_open.get():
                target = MODELS_OUT if self.result_ok else (self.runner.run_log_path or LOGS)
                if self.result_ok and self.runner.saved_to and os.path.isdir(self.runner.saved_to):
                    target = self.runner.saved_to
                try:
                    os.startfile(target)
                except OSError:
                    pass
            self.root.destroy()

    def cancel(self):
        if self.page == 4:
            self.root.destroy()
            return
        if self.runner.running:
            if messagebox.askyesno(f"{PRODUCT} Setup",
                                   "Are you sure you want to cancel the run?\n\n"
                                   "Stages that already finished are kept.",
                                   icon="warning", parent=self.root):
                self.status_var.set("Cancelling...")
                self.runner.cancel()
            return
        if messagebox.askyesno(f"{PRODUCT} Setup",
                               "Are you sure you want to cancel the Setup Wizard?",
                               parent=self.root):
            self.root.destroy()

    # ------------------------------------------------------------ run updates
    def _log(self, line, tag=None):
        self.log_lines.append((line, tag))
        if len(self.log_lines) > 4000:
            self.log_lines = self.log_lines[-3000:]
        if self.runner.run_log_path:
            try:
                with open(self.runner.run_log_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError:
                pass
        w = getattr(self, "logw", None)
        if self.page == 3 and w is not None and w.winfo_exists():
            w.configure(state="normal")
            at_end = w.yview()[1] > 0.995
            w.insert("end", line + "\n", tag)
            if int(w.index("end-1c").split(".")[0]) > 5000:
                w.delete("1.0", "1000.0")
            if at_end:
                w.see("end")
            w.configure(state="disabled")

    def _handle_output(self, line):
        m = PROGRESS_RE.match(line)
        if m:
            try:
                data = json.loads(m.group(2))
            except ValueError:
                data = None
            if data and m.group(1) == "PROGRESS":
                self.runner.apply_progress(data)
                return
            if data and m.group(1) == "TIMING":
                self._log(f"   [{data.get('stage')}: {data.get('seconds')} s]")
                return
        sm = SAVED_RE.match(line)
        if sm:
            self.runner.saved_to = sm.group(1)
        low = line.lower()
        bad = "traceback" in low or "error" in low or "fail" in low or "neuspj" in low
        self._log(line, "err" if bad else None)

    def _poll(self):
        r = self.runner
        changed = False
        try:
            for _ in range(500):
                item = r.q.get_nowait()
                changed = True
                kind = item[0]
                if kind == "raw":
                    self._handle_output(item[1])
                elif kind == "line":
                    self._log(item[1])
                elif kind == "stage":
                    r.current = item[1]
                    r.states[item[1]] = "running"
                    r.train_step = r.pipe_step = None
                    name = next(s["name"] for s in STAGES if s["key"] == item[1])
                    self._log("")
                    self._log(f"== {name} ==")
                elif kind == "state":
                    r.states[item[1]] = item[2]
                    r.stage_frac[item[1]] = 1.0
                elif kind == "finished":
                    r.running = False
                    r.current = None
                    self._keep_awake(False)
                    self.result_ok = all(r.states[s["key"]] in ("done", "warn") for s in STAGES)
                    self._log("")
                    self._log("Run finished." if self.result_ok else "Run stopped.",
                              None if self.result_ok else "err")
                    self.show(4)
                    if self.finish_shot:
                        self.root.after(1500, lambda: self._screenshot(self.finish_shot))
        except queue.Empty:
            pass
        if changed and self.page == 3:
            self._refresh_progress()
        self.root.after(100, self._poll)

    def _refresh_progress(self):
        r = self.runner
        if not hasattr(self, "pbar") or not self.pbar.winfo_exists():
            return
        self.pbar["value"] = int(r.overall_frac() * 1000)
        if r.current:
            name = next(s["name"] for s in STAGES if s["key"] == r.current)
            idx = [s["key"] for s in STAGES].index(r.current) + 1
            self.status_var.set(f"{name}...  (step {idx} of {len(STAGES)})")
        detail = ""
        if r.current == "train":
            if r.train_step:
                detail = f"Target {r.train_step[0]} of {r.train_step[1]}: {r.train_step[2]}"
            elif r.pipe_step:
                detail = f"Pipeline step {r.pipe_step[0]} of {r.pipe_step[1]}: {r.pipe_step[2]}"
        elif r.current == "backtest" and r.pipe_step is None:
            detail = "Comparing rain classifier feature sets"
        self.detail_var.set(detail)

    def _tick(self):
        r = self.runner
        if self.page == 3 and r.run_started:
            eta = r.eta()
            txt = f"Time elapsed: {_fmt_dur(time.time() - r.run_started)}"
            if eta is not None:
                txt += f"      Time remaining: about {_fmt_dur(eta)}"
            self.time_var.set(txt)
            self._refresh_progress()
        self.root.after(1000, self._tick)

    def _keep_awake(self, on):
        try:
            flags = 0x80000000 | (0x00000001 if on else 0)
            ctypes.windll.kernel32.SetThreadExecutionState(flags)
        except (AttributeError, OSError):
            pass

    # ------------------------------------------------------------ demo / screenshot
    def _demo(self):
        r = self.runner
        for k, v in (("prereq", "done"), ("station", "done"), ("models", "warn"),
                     ("backtest", "done"), ("train", "running")):
            r.states[k] = v
        r.stage_frac["train"] = 0.47
        r.current = "train"
        r.run_started = time.time() - 4622
        r.running = True
        r.train_step = (5, 9, "Udari vjetra")
        self.result_ok = True
        r.saved_to = os.path.join(MODELS_OUT, "20261111_0412")
        for line in ("== Training models ==", "[3/6] Treniranje XGBoost modela...",
                     "  Feature-a za treniranje: 1318 (Optuna trials: 30)",
                     "    Feature selection (wind_gusts_10m): 1318 -> 402 features",
                     "    Optuna (wind_gusts_10m): best MAE=1.0412 (depth=6, lr=0.0431, pruned=11/30)",
                     "   [wind_gusts_10m.optuna: 412.6 s]",
                     "    Lead report (wind_gusts_10m): mae_lead27=1.211, mae_lead51=1.348",
                     "  Udari vjetra   MAE: 1.032m/s | best: 1.402 (ITALIAME) | ens: 1.288"):
            self.log_lines.append((line, None))
        self.time_var.set("Time elapsed: 1:17:02      Time remaining: about 1:26:40")

    def _screenshot(self, path):
        try:
            from PIL import ImageGrab
            self.root.update()
            x, y = self.root.winfo_rootx(), self.root.winfo_rooty()
            w, h = self.root.winfo_width(), self.root.winfo_height()
            ImageGrab.grab(bbox=(x, y, x + w, y + h), all_screens=True).save(path)
        finally:
            self.root.destroy()


def main():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "MatijaIvanovic.MLForecastingPipeline")
    except (AttributeError, OSError):
        pass
    root = tk.Tk()
    Wizard(root, sys.argv[1:])
    root.mainloop()


if __name__ == "__main__":
    main()
