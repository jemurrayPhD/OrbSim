from __future__ import annotations

import random

from PySide6 import QtCore, QtGui, QtWidgets

from orbsim.chem.formula_format import format_formula_from_string
from orbsim.nomenclature import load_practice_pool, load_tutorial_content


class ChemicalNomenclatureDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Chemical Nomenclature")
        self.setMinimumSize(760, 560)
        self._pool = load_practice_pool()
        self._current_entry: dict | None = None
        self._score_correct = 0
        self._score_incorrect = 0
        self._attempts = 0
        self._max_attempts = 2
        self._awaiting_next = False
        self._build_ui()
        self._load_tutorial()
        self._new_prompt()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        tabs = QtWidgets.QTabWidget(self)
        tabs.addTab(self._build_tutorial_tab(), "Tutorial")
        tabs.addTab(self._build_practice_tab(), "Practice")
        layout.addWidget(tabs)

        close_row = QtWidgets.QHBoxLayout()
        close_row.addStretch()
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_row.addWidget(close_btn)
        layout.addLayout(close_row)

    def _build_tutorial_tab(self) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        wrapper = QtWidgets.QVBoxLayout(container)
        layout = QtWidgets.QHBoxLayout()

        self.topic_list = QtWidgets.QListWidget()
        self.topic_list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        layout.addWidget(self.topic_list, 1)

        self.topic_content = QtWidgets.QTextEdit()
        self.topic_content.setReadOnly(True)
        layout.addWidget(self.topic_content, 3)

        self.tutorial_footer = QtWidgets.QLabel(
            "References: IUPAC inorganic nomenclature guidance and standard general chemistry conventions."
        )
        self.tutorial_footer.setWordWrap(True)

        wrapper.addLayout(layout)
        wrapper.addWidget(self.tutorial_footer)
        return container

    def _build_practice_tab(self) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)

        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel("Mode"))
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Formula → Name", "Name → Formula"])
        self.mode_combo.currentIndexChanged.connect(self._new_prompt)
        mode_row.addWidget(self.mode_combo)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        filters_group = QtWidgets.QGroupBox("Filters")
        filters_layout = QtWidgets.QGridLayout(filters_group)
        self.filter_acids = QtWidgets.QCheckBox("Acids/Bases")
        self.filter_ionic = QtWidgets.QCheckBox("Ionic")
        self.filter_molecular = QtWidgets.QCheckBox("Molecular")
        self.filter_polyatomic = QtWidgets.QCheckBox("Polyatomic ions")
        self.filter_hydrates = QtWidgets.QCheckBox("Hydrates")
        self.filter_transition = QtWidgets.QCheckBox("Transition metals (roman numerals)")
        for checkbox in (
            self.filter_acids,
            self.filter_ionic,
            self.filter_molecular,
            self.filter_polyatomic,
            self.filter_hydrates,
            self.filter_transition,
        ):
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self._new_prompt)
        filters_layout.addWidget(self.filter_acids, 0, 0)
        filters_layout.addWidget(self.filter_ionic, 0, 1)
        filters_layout.addWidget(self.filter_molecular, 1, 0)
        filters_layout.addWidget(self.filter_polyatomic, 1, 1)
        filters_layout.addWidget(self.filter_hydrates, 2, 0)
        filters_layout.addWidget(self.filter_transition, 2, 1)
        layout.addWidget(filters_group)

        phase_group = QtWidgets.QGroupBox("Phases")
        phase_layout = QtWidgets.QHBoxLayout(phase_group)
        self.include_phases = QtWidgets.QCheckBox("Include phases")
        self.include_phases.stateChanged.connect(self._toggle_phase_filters)
        phase_layout.addWidget(self.include_phases)
        self.phase_aq = QtWidgets.QCheckBox("aq")
        self.phase_g = QtWidgets.QCheckBox("g")
        self.phase_l = QtWidgets.QCheckBox("l")
        self.phase_s = QtWidgets.QCheckBox("s")
        for cb in (self.phase_aq, self.phase_g, self.phase_l, self.phase_s):
            cb.setChecked(True)
            cb.setEnabled(False)
            cb.stateChanged.connect(self._new_prompt)
            phase_layout.addWidget(cb)
        layout.addWidget(phase_group)

        self.card = QtWidgets.QFrame()
        self.card.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.card.setObjectName("practiceCard")
        card_layout = QtWidgets.QVBoxLayout(self.card)
        score_row = QtWidgets.QHBoxLayout()
        self.score_label = QtWidgets.QLabel("Score: 0/0 (0%)")
        score_row.addWidget(self.score_label)
        score_row.addStretch()
        self.reset_score_button = QtWidgets.QPushButton("Reset score")
        self.reset_score_button.clicked.connect(self._reset_score)
        score_row.addWidget(self.reset_score_button)
        card_layout.addLayout(score_row)
        self.prompt_label = QtWidgets.QLabel("—")
        prompt_font = self.prompt_label.font()
        prompt_font.setPointSize(prompt_font.pointSize() + 4)
        prompt_font.setBold(True)
        self.prompt_label.setFont(prompt_font)
        self.prompt_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.prompt_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        card_layout.addWidget(self.prompt_label)

        self.answer_input = QtWidgets.QLineEdit()
        self.answer_input.returnPressed.connect(self._handle_enter)
        card_layout.addWidget(self.answer_input)

        button_row = QtWidgets.QHBoxLayout()
        self.check_button = QtWidgets.QPushButton("Check")
        self.check_button.clicked.connect(self._check_answer)
        self.reveal_button = QtWidgets.QPushButton("Reveal")
        self.reveal_button.clicked.connect(self._reveal_answer)
        self.new_button = QtWidgets.QPushButton("New")
        self.new_button.clicked.connect(self._new_prompt)
        button_row.addWidget(self.check_button)
        button_row.addWidget(self.reveal_button)
        button_row.addWidget(self.new_button)
        button_row.addStretch()
        self.allow_retries = QtWidgets.QCheckBox("Allow unlimited retries")
        self.allow_retries.setChecked(True)
        self.allow_retries.stateChanged.connect(self._sync_attempts)
        button_row.addWidget(self.allow_retries)
        card_layout.addLayout(button_row)

        self.hint_label = QtWidgets.QLabel("")
        self.hint_label.setWordWrap(True)
        self.hint_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        card_layout.addWidget(self.hint_label)

        layout.addWidget(self.card, 1)
        self._neutral_color = self.palette().color(QtGui.QPalette.ColorRole.Base)
        self._setup_reveal_effect()
        return container

    def _load_tutorial(self) -> None:
        content = load_tutorial_content()
        self.topic_list.clear()
        titles: list[str] = []
        for key, payload in content.items():
            title = payload.get("title", key)
            titles.append(title)
            item = QtWidgets.QListWidgetItem(title)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, key)
            self.topic_list.addItem(item)
        if titles:
            metrics = self.topic_list.fontMetrics()
            max_width = max(metrics.horizontalAdvance(title) for title in titles)
            self.topic_list.setMinimumWidth(max_width + 32)
        self.topic_list.currentItemChanged.connect(self._on_topic_changed)
        if self.topic_list.count() > 0:
            self.topic_list.setCurrentRow(0)

    def _on_topic_changed(self, current: QtWidgets.QListWidgetItem) -> None:
        if not current:
            return
        key = current.data(QtCore.Qt.ItemDataRole.UserRole)
        content = load_tutorial_content().get(key, {})
        self.topic_content.setHtml(content.get("body", ""))

    def _toggle_phase_filters(self) -> None:
        enabled = self.include_phases.isChecked()
        for cb in (self.phase_aq, self.phase_g, self.phase_l, self.phase_s):
            cb.setEnabled(enabled)
        self._new_prompt()

    def _filtered_pool(self) -> list[dict]:
        active_categories = set()
        if self.filter_acids.isChecked():
            active_categories.add("Acids/Bases")
        if self.filter_ionic.isChecked():
            active_categories.add("Ionic")
        if self.filter_molecular.isChecked():
            active_categories.add("Molecular")
        if self.filter_polyatomic.isChecked():
            active_categories.add("Polyatomic ions")
        if self.filter_hydrates.isChecked():
            active_categories.add("Hydrates")
        if self.filter_transition.isChecked():
            active_categories.add("Transition metals")
            active_categories.add("Transition metals (roman numerals)")

        allowed_phases = set()
        if self.include_phases.isChecked():
            if self.phase_aq.isChecked():
                allowed_phases.add("aq")
            if self.phase_g.isChecked():
                allowed_phases.add("g")
            if self.phase_l.isChecked():
                allowed_phases.add("l")
            if self.phase_s.isChecked():
                allowed_phases.add("s")

        filtered = []
        for entry in self._pool:
            category = entry.get("category", "")
            if active_categories and category not in active_categories:
                continue
            formula = str(entry.get("formula", ""))
            phase = None
            for token in ("aq", "g", "l", "s"):
                if f"({token})" in formula:
                    phase = token
                    break
            if phase and not self.include_phases.isChecked():
                continue
            if phase and allowed_phases and phase not in allowed_phases:
                continue
            filtered.append(entry)
        return filtered

    def _new_prompt(self) -> None:
        options = self._filtered_pool()
        if not options:
            self.prompt_label.setText("No matching entries.")
            self.answer_input.setEnabled(False)
            self.check_button.setEnabled(False)
            self.reveal_button.setEnabled(False)
            return
        self._current_entry = random.choice(options)
        mode = self.mode_combo.currentText()
        if mode.startswith("Formula"):
            raw = self._current_entry.get("formula", "—")
            prompt = format_formula_from_string(raw).rich if raw else "—"
        else:
            prompt = self._current_entry.get("primary_name", "—")
        self.prompt_label.setText(prompt)
        self.answer_input.setText("")
        self.answer_input.setEnabled(True)
        self.check_button.setEnabled(True)
        self.reveal_button.setEnabled(True)
        self.hint_label.setText("")
        self._attempts = 0
        self._awaiting_next = False
        self._animate_card(self._neutral_color, self._neutral_color)
        self.answer_input.setFocus()

    def _normalize(self, text: str) -> str:
        return "".join(text.lower().strip().split())

    def _check_answer(self) -> None:
        if not self._current_entry:
            return
        if self._awaiting_next:
            self._new_prompt()
            return
        mode = self.mode_combo.currentText()
        user = self._normalize(self.answer_input.text())
        correct = False
        if mode.startswith("Formula"):
            answers = [self._current_entry.get("primary_name", "")] + self._current_entry.get("accepted_answers", [])
            correct = user in {self._normalize(a) for a in answers if a}
        else:
            correct = user == self._normalize(self._current_entry.get("formula", ""))
        if correct:
            self.hint_label.setText("Correct!")
            self._animate_card(QtGui.QColor("#22c55e"), self._neutral_color)
            self._score_correct += 1
            self._awaiting_next = True
            self._update_score_label()
            self.answer_input.setFocus()
        else:
            self._attempts += 1
            hint = self._current_entry.get("hint", "Try again.")
            self.hint_label.setText(f"Incorrect. {hint}")
            self._animate_card(QtGui.QColor("#ef4444"), self._neutral_color)
            if not self.allow_retries.isChecked() and self._attempts >= self._max_attempts:
                self._score_incorrect += 1
                self._awaiting_next = True
                self._update_score_label()
                self.answer_input.setEnabled(False)
            else:
                self.answer_input.setFocus()

    def _reveal_answer(self) -> None:
        if not self._current_entry:
            return
        mode = self.mode_combo.currentText()
        if mode.startswith("Formula"):
            answer = self._current_entry.get("primary_name", "")
        else:
            raw = self._current_entry.get("formula", "")
            answer = format_formula_from_string(raw).rich if raw else ""
        hint = self._current_entry.get("hint", "")
        self.hint_label.setText(f"Answer: {answer}. {hint}")
        self._animate_reveal()
        if not self._awaiting_next:
            self._score_incorrect += 1
            self._awaiting_next = True
            self._update_score_label()

    def _setup_reveal_effect(self) -> None:
        self._reveal_effect = QtWidgets.QGraphicsOpacityEffect(self.hint_label)
        self.hint_label.setGraphicsEffect(self._reveal_effect)
        self._reveal_effect.setOpacity(1.0)

    def _animate_reveal(self) -> None:
        anim = QtCore.QPropertyAnimation(self._reveal_effect, b"opacity", self)
        anim.setDuration(350)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.start(QtCore.QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)

    def _animate_card(self, start: QtGui.QColor, end: QtGui.QColor) -> None:
        anim = QtCore.QVariantAnimation(self)
        anim.setDuration(400)
        anim.setStartValue(start)
        anim.setEndValue(end)
        anim.valueChanged.connect(self._update_card_color)
        anim.start(QtCore.QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)

    def _update_card_color(self, value: QtGui.QColor) -> None:
        color = value.name()
        self.card.setStyleSheet(f"#practiceCard {{ background-color: {color}; border-radius: 8px; }}")

    def _handle_enter(self) -> None:
        if self._awaiting_next:
            self._new_prompt()
            return
        self._check_answer()

    def _update_score_label(self) -> None:
        total = self._score_correct + self._score_incorrect
        pct = int(round((self._score_correct / total) * 100)) if total else 0
        self.score_label.setText(f"Score: {self._score_correct}/{total} ({pct}%)")

    def _reset_score(self) -> None:
        self._score_correct = 0
        self._score_incorrect = 0
        self._update_score_label()

    def _sync_attempts(self) -> None:
        if self.allow_retries.isChecked():
            self._attempts = 0
