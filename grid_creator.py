import tkinter as tk
from tkinter import ttk, messagebox
import sys

# ── Configuration ────────────────────────────────────────────────────────────
DEFAULT_ROWS = 8
DEFAULT_COLS = 10
CELL_SIZE    = 52        # px per cell
DOT_RADIUS   = 8         # dot radius inside each cell
OUTPUT_FILE  = "bit.txt"

COLOR_BG      = "#1a1a2e"
COLOR_PANEL   = "#16213e"
COLOR_CELL    = "#0f3460"
COLOR_BORDER  = "#1a1a2e"
COLOR_GREEN   = "#00d084"
COLOR_RED     = "#ff4757"
COLOR_DOT_OFF = "#2a4a6b"   # unclicked dot
COLOR_TEXT    = "#e0e0e0"
COLOR_BTN     = "#e94560"
COLOR_BTN_H   = "#ff6b81"

# ── Main Application ──────────────────────────────────────────────────────────
class GridApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Grid Clicker")
        self.root.configure(bg=COLOR_BG)
        self.root.resizable(False, False)

        self.rows = DEFAULT_ROWS
        self.cols = DEFAULT_COLS
        self.grid_state = []   # 2-D list of 0/1
        self.cell_ids   = []   # canvas oval IDs

        self._build_ui()

    # ── UI Construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        # ── top panel: dimension inputs ──
        panel = tk.Frame(self.root, bg=COLOR_PANEL, pady=10, padx=16)
        panel.pack(fill="x")

        lbl_style = dict(bg=COLOR_PANEL, fg=COLOR_TEXT,
                         font=("Courier New", 11, "bold"))

        tk.Label(panel, text="ROWS", **lbl_style).grid(row=0, column=0, padx=(0,4))
        self.row_var = tk.IntVar(value=self.rows)
        self.row_spin = tk.Spinbox(panel, from_=1, to=40, width=4,
                                   textvariable=self.row_var,
                                   font=("Courier New", 11),
                                   bg=COLOR_CELL, fg=COLOR_GREEN,
                                   buttonbackground=COLOR_CELL,
                                   relief="flat", insertbackground=COLOR_GREEN)
        self.row_spin.grid(row=0, column=1, padx=(0, 16))

        tk.Label(panel, text="COLS", **lbl_style).grid(row=0, column=2, padx=(0,4))
        self.col_var = tk.IntVar(value=self.cols)
        self.col_spin = tk.Spinbox(panel, from_=1, to=40, width=4,
                                   textvariable=self.col_var,
                                   font=("Courier New", 11),
                                   bg=COLOR_CELL, fg=COLOR_GREEN,
                                   buttonbackground=COLOR_CELL,
                                   relief="flat", insertbackground=COLOR_GREEN)
        self.col_spin.grid(row=0, column=3, padx=(0, 16))

        apply_btn = tk.Button(panel, text="⟳  REBUILD", command=self._rebuild_grid,
                              bg=COLOR_CELL, fg=COLOR_GREEN,
                              font=("Courier New", 10, "bold"),
                              relief="flat", padx=10, cursor="hand2",
                              activebackground=COLOR_GREEN, activeforeground=COLOR_BG)
        apply_btn.grid(row=0, column=4, padx=(0, 24))

        # ── status label ──
        self.status_var = tk.StringVar(value="Click cells to toggle · green = 0 · red = 1")
        tk.Label(panel, textvariable=self.status_var,
                 bg=COLOR_PANEL, fg=COLOR_TEXT,
                 font=("Courier New", 9)).grid(row=0, column=5, padx=(8, 0))

        # ── canvas frame ──
        self.canvas_frame = tk.Frame(self.root, bg=COLOR_BG)
        self.canvas_frame.pack(padx=20, pady=(12, 0))

        self.canvas = tk.Canvas(self.canvas_frame, bg=COLOR_BG,
                                highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._on_click)

        # ── bottom: reset + save ──
        bottom = tk.Frame(self.root, bg=COLOR_BG, pady=14)
        bottom.pack()

        reset_btn = tk.Button(bottom, text="↺  RESET",
                              command=self._reset_grid,
                              bg=COLOR_PANEL, fg=COLOR_TEXT,
                              font=("Courier New", 11, "bold"),
                              relief="flat", padx=16, pady=6, cursor="hand2",
                              activebackground=COLOR_CELL, activeforeground=COLOR_TEXT)
        reset_btn.pack(side="left", padx=8)

        save_btn = tk.Button(bottom, text="⬛  SAVE & EXIT",
                             command=self._save_and_exit,
                             bg=COLOR_BTN, fg="white",
                             font=("Courier New", 11, "bold"),
                             relief="flat", padx=16, pady=6, cursor="hand2",
                             activebackground=COLOR_BTN_H, activeforeground="white")
        save_btn.pack(side="left", padx=8)

        self._rebuild_grid()

    # ── Grid Construction ─────────────────────────────────────────────────────
    def _rebuild_grid(self):
        try:
            self.rows = max(1, min(40, self.row_var.get()))
            self.cols = max(1, min(40, self.col_var.get()))
        except Exception:
            return

        # reset state
        self.grid_state = [[0] * self.cols for _ in range(self.rows)]
        self.cell_ids   = [[None] * self.cols for _ in range(self.rows)]

        w = self.cols * CELL_SIZE + 2
        h = self.rows * CELL_SIZE + 2
        self.canvas.config(width=w, height=h)
        self.canvas.delete("all")

        for r in range(self.rows):
            for c in range(self.cols):
                x0 = c * CELL_SIZE + 1
                y0 = r * CELL_SIZE + 1
                x1 = x0 + CELL_SIZE - 2
                y1 = y0 + CELL_SIZE - 2

                # cell background
                self.canvas.create_rectangle(x0, y0, x1, y1,
                                             fill=COLOR_CELL,
                                             outline=COLOR_BORDER, width=2)
                # dot
                cx = x0 + CELL_SIZE // 2
                cy = y0 + CELL_SIZE // 2
                oid = self.canvas.create_oval(cx - DOT_RADIUS, cy - DOT_RADIUS,
                                              cx + DOT_RADIUS, cy + DOT_RADIUS,
                                              fill=COLOR_DOT_OFF,
                                              outline="", tags=f"dot_{r}_{c}")
                self.cell_ids[r][c] = oid

        self._update_status()

    # ── Interaction ───────────────────────────────────────────────────────────
    def _on_click(self, event):
        c = event.x // CELL_SIZE
        r = event.y // CELL_SIZE
        if 0 <= r < self.rows and 0 <= c < self.cols:
            self.grid_state[r][c] ^= 1          # toggle
            self._refresh_cell(r, c)
            self._update_status()

    def _refresh_cell(self, r, c):
        color = COLOR_RED if self.grid_state[r][c] else COLOR_DOT_OFF
        self.canvas.itemconfig(self.cell_ids[r][c], fill=color)
        # subtle glow on active dot
        if self.grid_state[r][c]:
            self.canvas.itemconfig(self.cell_ids[r][c], outline=COLOR_RED, width=2)
        else:
            self.canvas.itemconfig(self.cell_ids[r][c], outline="", width=0)

    def _reset_grid(self):
        for r in range(self.rows):
            for c in range(self.cols):
                self.grid_state[r][c] = 0
                self._refresh_cell(r, c)
        self._update_status()

    def _update_status(self):
        total   = self.rows * self.cols
        clicked = sum(self.grid_state[r][c]
                      for r in range(self.rows)
                      for c in range(self.cols))
        self.status_var.set(
            f"{self.rows}×{self.cols} grid  ·  {clicked}/{total} cells active"
        )

    # ── Output ────────────────────────────────────────────────────────────────
    def _format_output(self):
        rows_str = []
        for r in range(self.rows):
            inner = ",".join(str(self.grid_state[r][c]) for c in range(self.cols))
            rows_str.append("{" + inner + "}")
        return ",".join(rows_str)

    def _save_and_exit(self):
        output = self._format_output()
        try:
            with open(OUTPUT_FILE, "w") as f:
                f.write(output + "\n")
            messagebox.showinfo(
                "Saved",
                f"Grid saved to '{OUTPUT_FILE}'.\n\n{output[:120]}{'…' if len(output)>120 else ''}"
            )
        except OSError as e:
            messagebox.showerror("Error", f"Could not write file:\n{e}")
            return
        self.root.destroy()
        sys.exit(0)


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = GridApp(root)
    root.mainloop()
