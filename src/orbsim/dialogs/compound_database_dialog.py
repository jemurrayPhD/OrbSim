from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import threading

from PySide6 import QtCore, QtWidgets

from orbsim.chem import compound_db

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tools import build_compound_db


@dataclass
class BuildJob:
    seed_path: Path
    mode: str
    limit: int | None = None


class CompoundDbWorker(QtCore.QObject):
    progress = QtCore.Signal(int, int)
    log = QtCore.Signal(str)
    finished = QtCore.Signal(bool)

    def __init__(self, job: BuildJob, output_path: Path) -> None:
        super().__init__()
        self._job = job
        self._output_path = output_path
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    @QtCore.Slot()
    def run(self) -> None:
        try:
            reporter = build_compound_db.BuildReporter(
                log=self.log.emit,
                progress=self.progress.emit,
            )
            success = build_compound_db.build_db(
                self._job.seed_path,
                self._output_path,
                mode=self._job.mode,
                limit=self._job.limit,
                reporter=reporter,
                cancel_event=self._cancel_event,
            )
            self.finished.emit(success)
        except Exception as exc:
            self.log.emit(f"Error: {exc}")
            self.finished.emit(False)


class CompoundDatabaseDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Compound Database")
        self.setMinimumSize(680, 520)
        self._thread: QtCore.QThread | None = None
        self._worker: CompoundDbWorker | None = None

        self._build_ui()
        self._refresh_status()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        status_group = QtWidgets.QGroupBox("Status")
        status_layout = QtWidgets.QFormLayout(status_group)
        self.db_path_edit = QtWidgets.QLineEdit()
        self.db_path_edit.setReadOnly(True)
        status_layout.addRow("DB path", self.db_path_edit)
        self.count_label = QtWidgets.QLabel("0")
        status_layout.addRow("Compound count", self.count_label)
        self.last_built_label = QtWidgets.QLabel("—")
        status_layout.addRow("Last built", self.last_built_label)
        layout.addWidget(status_group)

        actions_group = QtWidgets.QGroupBox("Actions")
        actions_layout = QtWidgets.QVBoxLayout(actions_group)
        self.rebuild_button = QtWidgets.QPushButton("Rebuild from seed_compounds.csv")
        self.rebuild_button.clicked.connect(self._rebuild)
        actions_layout.addWidget(self.rebuild_button)

        top_row = QtWidgets.QHBoxLayout()
        top_row.addWidget(QtWidgets.QLabel("Expand: Top-N popular/common"))
        self.topn_spin = QtWidgets.QSpinBox()
        self.topn_spin.setRange(50, 5000)
        self.topn_spin.setValue(500)
        top_row.addWidget(self.topn_spin)
        self.topn_button = QtWidgets.QPushButton("Expand")
        self.topn_button.clicked.connect(self._expand_topn)
        top_row.addWidget(self.topn_button)
        actions_layout.addLayout(top_row)

        curriculum_row = QtWidgets.QHBoxLayout()
        curriculum_row.addWidget(QtWidgets.QLabel("Expand: Curriculum set"))
        self.curriculum_combo = QtWidgets.QComboBox()
        self.curriculum_combo.addItems(["Core", "Extended"])
        curriculum_row.addWidget(self.curriculum_combo)
        self.curriculum_button = QtWidgets.QPushButton("Expand")
        self.curriculum_button.clicked.connect(self._expand_curriculum)
        curriculum_row.addWidget(self.curriculum_button)
        actions_layout.addLayout(curriculum_row)

        layout.addWidget(actions_group)

        progress_group = QtWidgets.QGroupBox("Progress")
        progress_layout = QtWidgets.QVBoxLayout(progress_group)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        self.log_view = QtWidgets.QTextEdit()
        self.log_view.setReadOnly(True)
        progress_layout.addWidget(self.log_view, 1)
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel_job)
        progress_layout.addWidget(self.cancel_button, 0, QtCore.Qt.AlignmentFlag.AlignRight)
        layout.addWidget(progress_group, 1)

        close_row = QtWidgets.QHBoxLayout()
        close_row.addStretch()
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_row.addWidget(close_btn)
        layout.addLayout(close_row)

    def _refresh_status(self) -> None:
        path = compound_db.get_db_path()
        self.db_path_edit.setText(str(path))
        self.count_label.setText(str(compound_db.get_compound_count()))
        last_built = compound_db.get_last_built() or "—"
        self.last_built_label.setText(last_built)

    def _seed_path(self, filename: str) -> Path:
        return ROOT / "tools" / filename

    def _start_job(self, job: BuildJob) -> None:
        if self._thread and self._thread.isRunning():
            return
        self.log_view.append(f"Starting {job.mode} ({job.seed_path.name})…")
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self._set_actions_enabled(False)
        self.cancel_button.setEnabled(True)

        self._thread = QtCore.QThread(self)
        self._worker = CompoundDbWorker(job, compound_db.get_db_path())
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.log.connect(self._append_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.finished.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)
        self._thread.start()

    def _rebuild(self) -> None:
        job = BuildJob(self._seed_path("seed_compounds.csv"), mode="rebuild")
        self._start_job(job)

    def _expand_topn(self) -> None:
        limit = self.topn_spin.value()
        job = BuildJob(self._seed_path("seed_top_popular.csv"), mode="append", limit=limit)
        self._start_job(job)

    def _expand_curriculum(self) -> None:
        name = self.curriculum_combo.currentText().lower()
        filename = "seed_curriculum_core.csv" if name == "core" else "seed_curriculum_extended.csv"
        job = BuildJob(self._seed_path(filename), mode="append")
        self._start_job(job)

    def _cancel_job(self) -> None:
        if self._worker:
            self._worker.cancel()
            self._append_log("Cancel requested…")

    def _on_progress(self, current: int, total: int) -> None:
        if total <= 0:
            return
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)

    def _append_log(self, message: str) -> None:
        self.log_view.append(message)

    def _on_finished(self, success: bool) -> None:
        if success:
            self._append_log("Completed.")
        else:
            self._append_log("Stopped.")
        self._set_actions_enabled(True)
        self.cancel_button.setEnabled(False)
        self._refresh_status()

    def _cleanup_thread(self) -> None:
        if self._worker:
            self._worker.deleteLater()
        self._worker = None
        self._thread = None

    def _set_actions_enabled(self, enabled: bool) -> None:
        for widget in (
            self.rebuild_button,
            self.topn_button,
            self.curriculum_button,
            self.topn_spin,
            self.curriculum_combo,
        ):
            widget.setEnabled(enabled)
