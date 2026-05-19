# Version 3.0
# MAXDIFF ANALYSIS TOOL - desktop GUI

# Pure analysis primitives live in the maxdiff package (since 3.0).
# This file keeps only the matplotlib plotting helpers, segment-level
# pipeline, file I/O, and the customtkinter GUI.

import queue
import threading
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox, ttk

import customtkinter as ctk
import pandas as pd

from maxdiff import (
    HAS_NUMPYRO,
    DataFormatDetector,
    HierarchicalBayesMaxDiff,
    bootstrap_analysis,
    calculate_correlation_matrix,
    calculate_display_statistics,
    calculate_observed_percentages,
    calculate_scores_no_ci,
    check_errors,
    format_display_report,
    process_color_input,
    read_tabular_file,
)
from maxdiff.plotting import (
    plot_correlation_matrix,
    plot_display_balance,
    plot_observed_percentages,
    plot_scores,
    save_dataframe,
    save_plot,
)

# Set appearance mode and color theme
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")


def save_results(
    results,
    base_dir,
    prefix="",
    sample_size=None,
    positive_color="#F4B400",
    negative_color="#5B9BD5",
    error_bar_color="#a1a1a1",
    zero_line_color="red",
    anchor_item_color="#a1a1a1",
    anchor_item_error_color="#a1a1a1",
    segment_info=None,
    anchor_item=None,
    output_terms=("Most", "Least"),
    include_ci=True,
):

    # Map internal result types to cleaner output names
    filename_map = {
        "bootstrap_results": "net_scores",
        "scores": "net_scores",
        "hb_utilities": "hb_utilities",
        "observed_percentages": "selection_frequencies",
    }

    title_map = {
        "bootstrap_results": "MaxDiff Net Scores",
        "scores": "MaxDiff Net Scores",
        "hb_utilities": "HB Utilities",
        "observed_percentages": "Selection Frequencies",
    }

    for result_type, data in results.items():
        # Get clean filename
        clean_name = filename_map.get(result_type, result_type)
        filename = (
            f"{prefix}_{clean_name}_n{sample_size}"
            if prefix
            else f"overall_{clean_name}_n{sample_size}"
        )

        save_dataframe(data, filename, base_dir)

        if result_type in ["bootstrap_results", "scores", "hb_utilities"]:
            title = title_map.get(result_type, "MaxDiff Scores")
            if result_type == "scores" and not include_ci:
                title += " (No CI)"
            if segment_info:
                title += f", {segment_info}"
            fig = plot_scores(
                data,
                positive_color,
                negative_color,
                error_bar_color,
                zero_line_color,
                anchor_item_color,
                anchor_item_error_color,
                title=title,
                sample_size=sample_size,
                anchor_item=anchor_item,
                include_ci=include_ci,
            )
            save_plot(fig, f"{filename}_plot", base_dir)

        elif result_type == "observed_percentages":
            title = title_map.get(result_type, "Selection Frequencies")
            if segment_info:
                title += f", {segment_info}"
            if sample_size:
                title += f" (n={sample_size})"
            fig = plot_observed_percentages(data, title, output_terms)
            save_plot(fig, f"{filename}_plot", base_dir)


def segment_maxdiff_analysis(
    maxdiff_df,
    segment_df,
    pos_col,
    neg_col,
    output_terms,
    n_iterations=10000,
    anchor_item=None,
    include_ci=True,
    progress_callback=None,
    log_callback=None,
    analysis_method="count",
    hb_settings=None,
):
    results = {}
    segment_columns = [col for col in segment_df.columns if col != "Response ID"]
    attribute_columns = [col for col in maxdiff_df.columns if col.startswith("Attribute")]
    unique_attributes = pd.unique(maxdiff_df[attribute_columns].values.ravel("K"))
    unique_attributes = unique_attributes[~pd.isnull(unique_attributes)]
    attr_to_index = {attr: i for i, attr in enumerate(unique_attributes)}
    merged_df = pd.merge(
        maxdiff_df, segment_df[["Response ID"] + segment_columns], on="Response ID", how="left"
    )

    total_segments = sum(len(merged_df[col].dropna().unique()) for col in segment_columns)
    current_segment = 0

    for column in segment_columns:
        if log_callback:
            log_callback(f"Processing segment column: {column}")
        column_results = {}

        for segment in merged_df[column].dropna().unique():
            if log_callback:
                log_callback(f"  Processing segment: {segment}")
            segment_data = merged_df[merged_df[column] == segment]

            if len(segment_data) == 0:
                continue

            sample_size = segment_data["Response ID"].nunique()

            # Calculate display stats for segment
            seg_display_stats, seg_balance = calculate_display_statistics(
                segment_data, attribute_columns, pos_col, neg_col
            )

            # Run appropriate analysis
            if analysis_method == "hb" and HAS_NUMPYRO:
                try:
                    hb_model = HierarchicalBayesMaxDiff(
                        n_iterations=hb_settings.get("iterations", 5000),
                        n_warmup=hb_settings.get("warmup", 2500),
                        n_chains=hb_settings.get("chains", 4),
                    )

                    def seg_progress(p):
                        if progress_callback:
                            progress_callback((current_segment + p) / total_segments)

                    segment_results = hb_model.fit(
                        segment_data,
                        attribute_columns,
                        pos_col,
                        neg_col,
                        progress_callback=seg_progress,
                        log_callback=None,
                    )
                    result_key = "hb_utilities"
                except Exception as e:
                    if log_callback:
                        log_callback(f"    HB failed for segment, falling back to count: {e}")
                    segment_results = calculate_scores_no_ci(
                        segment_data, attribute_columns, pos_col, neg_col
                    )
                    result_key = "scores"
            elif include_ci:

                def seg_progress(p):
                    if progress_callback:
                        progress_callback((current_segment + p) / total_segments)

                segment_results = bootstrap_analysis(
                    segment_data,
                    attribute_columns,
                    unique_attributes,
                    attr_to_index,
                    pos_col,
                    neg_col,
                    n_iterations,
                    seg_progress,
                )
                result_key = "bootstrap_results"
            else:
                segment_results = calculate_scores_no_ci(
                    segment_data, attribute_columns, pos_col, neg_col
                )
                result_key = "scores"
                if progress_callback:
                    progress_callback((current_segment + 1) / total_segments)

            segment_observed = calculate_observed_percentages(
                segment_data, attribute_columns, pos_col, neg_col, output_terms
            )
            segment_corr = calculate_correlation_matrix(
                segment_data, attribute_columns, pos_col, neg_col
            )

            column_results[segment] = {
                "sample_size": sample_size,
                result_key: segment_results,
                "observed_percentages": segment_observed,
                "correlation_matrix": segment_corr,
                "display_statistics": seg_display_stats,
                "balance_metrics": seg_balance,
            }
            current_segment += 1

        if column_results:
            results[column] = column_results

    return results


def process_segment_results(
    segment_results, base_dir, colors, output_terms, anchor_item=None, include_ci=True
):
    for column, column_results in segment_results.items():
        for segment, segment_data in column_results.items():
            safe_column = "".join(c if c.isalnum() else "_" for c in str(column))
            safe_segment = "".join(c if c.isalnum() else "_" for c in str(segment))
            prefix = f"{safe_column}_{safe_segment}"
            sample_size = segment_data["sample_size"]
            segment_info = f"{column}: {segment}"

            # Find the result key
            score_key = None
            for key in ["hb_utilities", "bootstrap_results", "scores"]:
                if key in segment_data:
                    score_key = key
                    break

            if score_key:
                save_results(
                    {
                        score_key: segment_data[score_key],
                        "observed_percentages": segment_data["observed_percentages"],
                    },
                    base_dir,
                    prefix,
                    sample_size,
                    **colors,
                    segment_info=segment_info,
                    anchor_item=anchor_item,
                    output_terms=output_terms,
                    include_ci=include_ci,
                )

            # Save display statistics for segment
            save_dataframe(
                segment_data["display_statistics"],
                f"{prefix}_display_statistics_n{sample_size}",
                base_dir,
            )

            # Save display balance plot for segment
            balance_fig = plot_display_balance(
                segment_data["display_statistics"],
                title=f"Display Balance - {segment_info} (n={sample_size})",
                output_terms=output_terms,
            )
            save_plot(balance_fig, f"{prefix}_display_balance_n{sample_size}", base_dir)

            corr_matrix = segment_data["correlation_matrix"]
            save_dataframe(
                corr_matrix,
                f"{prefix}_correlation_matrix_n{sample_size}",
                base_dir,
                include_index=True,
            )
            corr_fig = plot_correlation_matrix(
                corr_matrix, f"Correlation Matrix, {segment_info} (n={sample_size})"
            )
            save_plot(corr_fig, f"{prefix}_correlation_matrix_n{sample_size}_plot", base_dir)


# ============================================================================
# GUI COMPONENTS
# ============================================================================


class ColorButton(ctk.CTkFrame):
    def __init__(self, master, label, default_color, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.color = default_color
        self.label = ctk.CTkLabel(self, text=label, width=150, anchor="w")
        self.label.pack(side="left", padx=(0, 10))
        self.color_preview = ctk.CTkButton(
            self,
            width=40,
            height=28,
            text="",
            fg_color=default_color,
            hover_color=default_color,
            border_width=2,
            border_color="#999999",
            command=self.pick_color,
        )
        self.color_preview.pack(side="left", padx=(0, 10))
        self.color_entry = ctk.CTkEntry(self, width=100, placeholder_text="#RRGGBB")
        self.color_entry.insert(0, default_color)
        self.color_entry.pack(side="left")
        self.color_entry.bind("<Return>", self.update_from_entry)
        self.color_entry.bind("<FocusOut>", self.update_from_entry)

    def pick_color(self):
        color = colorchooser.askcolor(color=self.color, title="Choose Color")
        if color[1]:
            self.set_color(color[1])

    def update_from_entry(self, event=None):
        color = process_color_input(self.color_entry.get())
        if color:
            self.set_color(color)
        else:
            self.color_entry.delete(0, "end")
            self.color_entry.insert(0, self.color)

    def set_color(self, color):
        self.color = color
        self.color_preview.configure(fg_color=color, hover_color=color)
        self.color_entry.delete(0, "end")
        self.color_entry.insert(0, color)

    def get_color(self):
        return self.color


class CollapsibleFrame(ctk.CTkFrame):
    def __init__(self, master, title, expanded=False, **kwargs):
        super().__init__(master, **kwargs)
        self.expanded = expanded
        self.header = ctk.CTkFrame(self, fg_color="transparent")
        self.header.pack(fill="x", padx=10, pady=(10, 5))
        self.toggle_btn = ctk.CTkButton(
            self.header,
            text="▼" if expanded else "▶",
            width=30,
            height=28,
            command=self.toggle,
            fg_color="transparent",
            text_color=("gray20", "gray80"),
            hover_color=("gray80", "gray30"),
        )
        self.toggle_btn.pack(side="left")
        self.title_label = ctk.CTkLabel(
            self.header, text=title, font=ctk.CTkFont(size=14, weight="bold")
        )
        self.title_label.pack(side="left", padx=5)
        self.content = ctk.CTkFrame(self, fg_color="transparent")
        if expanded:
            self.content.pack(fill="x", padx=15, pady=(0, 10))

    def toggle(self):
        if self.expanded:
            self.content.pack_forget()
            self.toggle_btn.configure(text="▶")
        else:
            self.content.pack(fill="x", padx=15, pady=(0, 10))
            self.toggle_btn.configure(text="▼")
        self.expanded = not self.expanded

    def get_content_frame(self):
        return self.content


class DataPreviewWindow(ctk.CTkToplevel):
    def __init__(self, master, df, detected_format, format_message, on_convert_callback):
        super().__init__(master)
        self.title("Data Preview & Conversion")
        self.geometry("1000x700")
        self.df = df
        self.detected_format = detected_format
        self.on_convert = on_convert_callback
        self.create_widgets(format_message)
        self.transient(master)
        self.grab_set()

    def create_widgets(self, format_message):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=(20, 10))
        ctk.CTkLabel(
            header,
            text="📋 Data Preview & Format Conversion",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).pack(anchor="w")

        detection_frame = ctk.CTkFrame(self)
        detection_frame.pack(fill="x", padx=20, pady=10)
        status_color = "green" if self.detected_format == "ready" else "orange"
        ctk.CTkLabel(
            detection_frame,
            text=f"🔍 {format_message}",
            font=ctk.CTkFont(size=14),
            text_color=status_color,
        ).pack(padx=15, pady=10, anchor="w")

        desc_text = DataFormatDetector.get_format_description(self.detected_format)
        ctk.CTkLabel(
            detection_frame,
            text=desc_text,
            font=ctk.CTkFont(size=12),
            wraplength=900,
            justify="left",
        ).pack(padx=15, pady=(0, 10), anchor="w")

        preview_frame = ctk.CTkFrame(self)
        preview_frame.pack(fill="both", expand=True, padx=20, pady=10)
        ctk.CTkLabel(
            preview_frame,
            text="Data Preview (first 10 rows):",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(anchor="w", padx=10, pady=(10, 5))

        tree_frame = ctk.CTkFrame(preview_frame)
        tree_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        y_scroll = ttk.Scrollbar(tree_frame, orient="vertical")
        x_scroll = ttk.Scrollbar(tree_frame, orient="horizontal")
        self.tree = ttk.Treeview(
            tree_frame, yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set
        )
        y_scroll.config(command=self.tree.yview)
        x_scroll.config(command=self.tree.xview)
        y_scroll.pack(side="right", fill="y")
        x_scroll.pack(side="bottom", fill="x")
        self.tree.pack(fill="both", expand=True)

        self.tree["columns"] = list(self.df.columns)
        self.tree["show"] = "headings"
        for col in self.df.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, minwidth=50)
        for _, row in self.df.head(10).iterrows():
            self.tree.insert("", "end", values=list(row))

        if self.detected_format != "ready":
            self.create_conversion_options()

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=(10, 20))

        if self.detected_format == "ready":
            ctk.CTkButton(
                btn_frame, text="✓ Use This Data", width=150, command=self.use_directly
            ).pack(side="right", padx=5)
        else:
            ctk.CTkButton(
                btn_frame, text="🔄 Convert Data", width=150, command=self.convert_data
            ).pack(side="right", padx=5)

        ctk.CTkButton(
            btn_frame, text="Cancel", width=100, fg_color="gray", command=self.destroy
        ).pack(side="right", padx=5)
        ctk.CTkButton(
            btn_frame, text="📖 Column Mapper", width=150, command=self.open_column_mapper
        ).pack(side="left", padx=5)

    def create_conversion_options(self):
        options_frame = ctk.CTkFrame(self)
        options_frame.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(
            options_frame, text="⚙️ Conversion Settings", font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=10, pady=(10, 5))

        settings_grid = ctk.CTkFrame(options_frame, fg_color="transparent")
        settings_grid.pack(fill="x", padx=10, pady=10)

        if self.detected_format == "qualtrics_wide":
            row1 = ctk.CTkFrame(settings_grid, fg_color="transparent")
            row1.pack(fill="x", pady=2)
            ctk.CTkLabel(row1, text="Task prefix:", width=200, anchor="w").pack(side="left")
            self.task_prefix_entry = ctk.CTkEntry(row1, width=100)
            self.task_prefix_entry.insert(0, "Q")
            self.task_prefix_entry.pack(side="left")

            row2 = ctk.CTkFrame(settings_grid, fg_color="transparent")
            row2.pack(fill="x", pady=2)
            ctk.CTkLabel(row2, text="'Best' suffix:", width=200, anchor="w").pack(side="left")
            self.best_suffix_entry = ctk.CTkEntry(row2, width=100)
            self.best_suffix_entry.insert(0, "Best")
            self.best_suffix_entry.pack(side="left")

            row3 = ctk.CTkFrame(settings_grid, fg_color="transparent")
            row3.pack(fill="x", pady=2)
            ctk.CTkLabel(row3, text="'Worst' suffix:", width=200, anchor="w").pack(side="left")
            self.worst_suffix_entry = ctk.CTkEntry(row3, width=100)
            self.worst_suffix_entry.insert(0, "Worst")
            self.worst_suffix_entry.pack(side="left")

            row4 = ctk.CTkFrame(settings_grid, fg_color="transparent")
            row4.pack(fill="x", pady=2)
            ctk.CTkLabel(row4, text="ID column:", width=200, anchor="w").pack(side="left")
            self.id_col_menu = ctk.CTkComboBox(row4, values=list(self.df.columns), width=200)
            id_candidates = [
                c for c in self.df.columns if "response" in c.lower() or "id" in c.lower()
            ]
            if id_candidates:
                self.id_col_menu.set(id_candidates[0])
            self.id_col_menu.pack(side="left")

        elif self.detected_format == "long":
            cols = list(self.df.columns)
            for label, attr in [
                ("Response ID column:", "long_id_col"),
                ("Task column:", "long_task_col"),
                ("Item column:", "long_item_col"),
                ("Selection column:", "long_selection_col"),
            ]:
                row = ctk.CTkFrame(settings_grid, fg_color="transparent")
                row.pack(fill="x", pady=2)
                ctk.CTkLabel(row, text=label, width=200, anchor="w").pack(side="left")
                combo = ctk.CTkComboBox(row, values=cols, width=200)
                combo.pack(side="left")
                setattr(self, attr, combo)

    def use_directly(self):
        self.on_convert(self.df)
        self.destroy()

    def convert_data(self):
        try:
            if self.detected_format == "qualtrics_wide":
                converted_df = DataFormatDetector.convert_qualtrics_wide(
                    self.df,
                    self.task_prefix_entry.get(),
                    self.best_suffix_entry.get(),
                    self.worst_suffix_entry.get(),
                    self.id_col_menu.get(),
                )
            elif self.detected_format == "long":
                converted_df = DataFormatDetector.convert_long_format(
                    self.df,
                    self.long_id_col.get(),
                    self.long_task_col.get(),
                    self.long_item_col.get(),
                    self.long_selection_col.get(),
                )
            else:
                messagebox.showwarning("Warning", "Please use the Column Mapper")
                return
            self.show_converted_preview(converted_df)
        except Exception as e:
            messagebox.showerror("Error", f"Conversion failed:\n{e!s}")

    def show_converted_preview(self, converted_df):
        preview_win = ctk.CTkToplevel(self)
        preview_win.title("Converted Data Preview")
        preview_win.geometry("900x500")

        ctk.CTkLabel(
            preview_win, text="✓ Converted Data", font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        ctk.CTkLabel(
            preview_win,
            text=f"Shape: {converted_df.shape[0]} rows × {converted_df.shape[1]} columns",
        ).pack()

        tree_frame = ctk.CTkFrame(preview_win)
        tree_frame.pack(fill="both", expand=True, padx=20, pady=10)

        y_scroll = ttk.Scrollbar(tree_frame, orient="vertical")
        x_scroll = ttk.Scrollbar(tree_frame, orient="horizontal")
        tree = ttk.Treeview(tree_frame, yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        y_scroll.config(command=tree.yview)
        x_scroll.config(command=tree.xview)
        y_scroll.pack(side="right", fill="y")
        x_scroll.pack(side="bottom", fill="x")
        tree.pack(fill="both", expand=True)

        tree["columns"] = list(converted_df.columns)
        tree["show"] = "headings"
        for col in converted_df.columns:
            tree.heading(col, text=col)
            tree.column(col, width=120)
        for _, row in converted_df.head(20).iterrows():
            tree.insert("", "end", values=list(row))

        btn_frame = ctk.CTkFrame(preview_win, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=10)

        def use_converted():
            self.on_convert(converted_df)
            preview_win.destroy()
            self.destroy()

        def save_converted():
            filepath = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                title="Save Converted Data",
            )
            if filepath:
                converted_df.to_csv(filepath, index=False)
                messagebox.showinfo("Saved", f"Saved to:\n{filepath}")

        ctk.CTkButton(btn_frame, text="💾 Save CSV", width=120, command=save_converted).pack(
            side="left", padx=5
        )
        ctk.CTkButton(btn_frame, text="✓ Use for Analysis", width=150, command=use_converted).pack(
            side="right", padx=5
        )
        ctk.CTkButton(
            btn_frame, text="Cancel", width=100, fg_color="gray", command=preview_win.destroy
        ).pack(side="right", padx=5)

    def open_column_mapper(self):
        ColumnMapperWindow(self, self.df, self.on_convert)


class ColumnMapperWindow(ctk.CTkToplevel):
    def __init__(self, master, df, on_convert_callback):
        super().__init__(master)
        self.title("Manual Column Mapper")
        self.geometry("700x600")
        self.df = df
        self.on_convert = on_convert_callback
        self.create_widgets()
        self.transient(master)
        self.grab_set()

    def create_widgets(self):
        ctk.CTkLabel(
            self, text="🔧 Manual Column Mapping", font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=20)

        map_frame = ctk.CTkScrollableFrame(self, height=400)
        map_frame.pack(fill="both", expand=True, padx=20, pady=10)

        cols = ["(None)"] + list(self.df.columns)
        self.mappings = {}

        ctk.CTkLabel(map_frame, text="Required:", font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", pady=(10, 5)
        )

        for label, key in [
            ("Response ID:", "response_id"),
            ("Most/Best:", "most"),
            ("Least/Worst:", "least"),
        ]:
            row = ctk.CTkFrame(map_frame, fg_color="transparent")
            row.pack(fill="x", pady=3)
            ctk.CTkLabel(row, text=label, width=150, anchor="w").pack(side="left")
            combo = ctk.CTkComboBox(row, values=cols, width=250)
            combo.pack(side="left")
            self.mappings[key] = combo

        ctk.CTkLabel(map_frame, text="\nAttributes:", font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", pady=(10, 5)
        )

        for i in range(1, 8):
            row = ctk.CTkFrame(map_frame, fg_color="transparent")
            row.pack(fill="x", pady=3)
            ctk.CTkLabel(row, text=f"Attribute{i}:", width=150, anchor="w").pack(side="left")
            combo = ctk.CTkComboBox(row, values=cols, width=250)
            combo.pack(side="left")
            self.mappings[f"attr{i}"] = combo

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=20)
        ctk.CTkButton(
            btn_frame, text="Cancel", width=100, fg_color="gray", command=self.destroy
        ).pack(side="right", padx=5)
        ctk.CTkButton(btn_frame, text="✓ Apply", width=150, command=self.apply_mapping).pack(
            side="right", padx=5
        )

    def apply_mapping(self):
        try:
            id_col = self.mappings["response_id"].get()
            most_col = self.mappings["most"].get()
            least_col = self.mappings["least"].get()

            if "(None)" in [id_col, most_col, least_col]:
                messagebox.showwarning("Warning", "Map all required columns")
                return

            attr_cols = [
                self.mappings[f"attr{i}"].get()
                for i in range(1, 8)
                if self.mappings[f"attr{i}"].get() != "(None)"
            ]

            if len(attr_cols) < 3:
                messagebox.showwarning("Warning", "Need at least 3 attribute columns")
                return

            new_df = pd.DataFrame({"Response ID": self.df[id_col]})
            for i, col in enumerate(attr_cols, 1):
                new_df[f"Attribute{i}"] = self.df[col]
            new_df["Most"] = self.df[most_col]
            new_df["Least"] = self.df[least_col]

            self.on_convert(new_df)
            self.master.destroy()
            self.destroy()
        except Exception as e:
            messagebox.showerror("Error", str(e))


# ============================================================================
# MAIN APPLICATION
# ============================================================================


class MaxDiffGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("MaxDiff Analysis Tool")
        self.geometry("750x900")
        self.minsize(700, 750)

        self.maxdiff_file = None
        self.maxdiff_df = None
        self.segment_file = None
        self.analysis_thread = None
        self.message_queue = queue.Queue()

        self.create_widgets()
        self.check_queue()

    def create_widgets(self):
        self.main_frame = ctk.CTkScrollableFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Header
        header = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        header.pack(fill="x", pady=(0, 20))
        ctk.CTkLabel(
            header, text="📊 MaxDiff Analysis Tool", font=ctk.CTkFont(size=28, weight="bold")
        ).pack()
        ctk.CTkLabel(header, text="By Carl J", font=ctk.CTkFont(size=14), text_color="gray").pack(
            pady=(5, 0)
        )

        # Help section
        help_frame = CollapsibleFrame(self.main_frame, "📖 Data Format Guide")
        help_frame.pack(fill="x", pady=(0, 15))
        help_content = help_frame.get_content_frame()

        instructions = """REQUIRED FORMAT: Response ID | Attribute1 | Attribute2 | Attribute3 | ... | Most | Least

- Response ID: Participant identifier (can repeat for multiple tasks)
- Attribute columns: Items shown in each choice task (minimum 3)
- Most/Best: The item selected as most preferred
- Least/Worst: The item selected as least preferred

Use "Browse" to auto-detect format and convert if needed."""

        ctk.CTkLabel(
            help_content,
            text=instructions,
            font=ctk.CTkFont(size=12),
            justify="left",
            wraplength=650,
        ).pack(pady=5, anchor="w")

        btn_row = ctk.CTkFrame(help_content, fg_color="transparent")
        btn_row.pack(fill="x", pady=5)
        ctk.CTkButton(
            btn_row, text="📥 Example MaxDiff CSV", width=180, command=self.generate_example_maxdiff
        ).pack(side="left", padx=5)
        ctk.CTkButton(
            btn_row, text="📥 Example Segment CSV", width=180, command=self.generate_example_segment
        ).pack(side="left", padx=5)

        # Data files section
        files_frame = ctk.CTkFrame(self.main_frame)
        files_frame.pack(fill="x", pady=(0, 15))

        ctk.CTkLabel(
            files_frame, text="📁 Data Files", font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", padx=15, pady=(15, 10))

        maxdiff_row = ctk.CTkFrame(files_frame, fg_color="transparent")
        maxdiff_row.pack(fill="x", padx=15, pady=5)
        ctk.CTkLabel(maxdiff_row, text="MaxDiff Data:", width=100, anchor="w").pack(side="left")
        self.maxdiff_entry = ctk.CTkEntry(maxdiff_row, width=300, state="disabled")
        self.maxdiff_entry.pack(side="left", padx=(0, 10))
        ctk.CTkButton(
            maxdiff_row,
            text="Browse",
            width=100,
            command=self.load_and_preview_maxdiff,
            fg_color="#2d8a4e",
            hover_color="#236b3c",
        ).pack(side="left")
        self.maxdiff_status = ctk.CTkLabel(
            maxdiff_row, text="⚠️ Required", text_color="orange", width=80
        )
        self.maxdiff_status.pack(side="left", padx=10)

        self.data_info_label = ctk.CTkLabel(
            files_frame, text="", font=ctk.CTkFont(size=12), text_color="gray"
        )
        self.data_info_label.pack(anchor="w", padx=15, pady=(0, 5))

        segment_row = ctk.CTkFrame(files_frame, fg_color="transparent")
        segment_row.pack(fill="x", padx=15, pady=(5, 15))
        ctk.CTkLabel(segment_row, text="Segment Data:", width=100, anchor="w").pack(side="left")
        self.segment_entry = ctk.CTkEntry(
            segment_row, width=300, state="disabled", placeholder_text="Optional"
        )
        self.segment_entry.pack(side="left", padx=(0, 10))
        ctk.CTkButton(segment_row, text="Browse", width=100, command=self.browse_segment).pack(
            side="left"
        )
        self.segment_status = ctk.CTkLabel(
            segment_row, text="Optional", text_color="gray", width=80
        )
        self.segment_status.pack(side="left", padx=10)
        ctk.CTkButton(
            segment_row, text="✕", width=30, fg_color="gray", command=self.clear_segment
        ).pack(side="left", padx=5)

        # Analysis Options (collapsible, default collapsed)
        options_collapsible = CollapsibleFrame(
            self.main_frame, "⚙️ Analysis Options", expanded=False
        )
        options_collapsible.pack(fill="x", pady=(0, 15))
        options_content = options_collapsible.get_content_frame()

        # Output Labels (inside Analysis Options)
        term_section = ctk.CTkFrame(options_content, fg_color="transparent")
        term_section.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(term_section, text="Output Labels:", font=ctk.CTkFont(size=14)).pack(
            anchor="w", pady=(0, 5)
        )
        self.output_term_var = ctk.StringVar(value="Best/Worst")
        ctk.CTkSegmentedButton(
            term_section,
            values=["Most/Least", "Best/Worst"],
            variable=self.output_term_var,
            width=250,
        ).pack(anchor="w")

        # Analysis Method (inside Analysis Options)
        method_section = ctk.CTkFrame(options_content, fg_color="transparent")
        method_section.pack(fill="x", pady=(10, 10))
        ctk.CTkLabel(method_section, text="Analysis Method:", font=ctk.CTkFont(size=14)).pack(
            anchor="w", pady=(0, 5)
        )

        self.analysis_method_var = ctk.StringVar(value="count")

        self.count_radio = ctk.CTkRadioButton(
            method_section,
            text="Count-Based Analysis (Fast, Recommended)",
            variable=self.analysis_method_var,
            value="count",
            command=self.update_method_options,
        )
        self.count_radio.pack(anchor="w", pady=2)

        hb_text = "Hierarchical Bayes (Slow, ~5-20 min)"
        if not HAS_NUMPYRO:
            hb_text += " [Unavailable]"

        self.hb_radio = ctk.CTkRadioButton(
            method_section,
            text=hb_text,
            variable=self.analysis_method_var,
            value="hb",
            command=self.update_method_options,
        )
        self.hb_radio.pack(anchor="w", pady=2)

        if not HAS_NUMPYRO:
            self.hb_radio.configure(state="disabled")

        self.method_desc_label = ctk.CTkLabel(
            method_section,
            text="Simple count-based scores with optional bootstrap confidence intervals.",
            font=ctk.CTkFont(size=11),
            text_color="gray",
            wraplength=650,
        )
        self.method_desc_label.pack(anchor="w", pady=(5, 0))

        # Count-based options
        self.count_options_frame = ctk.CTkFrame(options_content, fg_color="transparent")
        self.count_options_frame.pack(fill="x", pady=5)

        ci_row = ctk.CTkFrame(self.count_options_frame, fg_color="transparent")
        ci_row.pack(fill="x", pady=2)
        self.include_ci_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            ci_row,
            text="Include 95% Confidence Intervals (Bootstrap)",
            variable=self.include_ci_var,
            command=self.toggle_ci_options,
            font=ctk.CTkFont(size=14),
        ).pack(side="left")

        iter_row = ctk.CTkFrame(self.count_options_frame, fg_color="transparent")
        iter_row.pack(fill="x", pady=2)
        ctk.CTkLabel(iter_row, text="Bootstrap Iterations:", width=150, anchor="w").pack(
            side="left"
        )
        self.iterations_var = ctk.StringVar(value="10000")
        self.iterations_entry = ctk.CTkEntry(iter_row, width=120, textvariable=self.iterations_var)
        self.iterations_entry.pack(side="left")

        # HB options
        self.hb_options_frame = ctk.CTkFrame(options_content, fg_color="transparent")
        self.hb_options_frame.pack(fill="x", pady=5)

        hb_iter_row = ctk.CTkFrame(self.hb_options_frame, fg_color="transparent")
        hb_iter_row.pack(fill="x", pady=2)
        ctk.CTkLabel(hb_iter_row, text="MCMC Iterations:", width=150, anchor="w").pack(side="left")
        self.hb_iterations_var = ctk.StringVar(value="5000")
        self.hb_iterations_entry = ctk.CTkEntry(
            hb_iter_row, width=120, textvariable=self.hb_iterations_var
        )
        self.hb_iterations_entry.pack(side="left")
        ctk.CTkLabel(
            hb_iter_row,
            text="(per chain, after warmup)",
            font=ctk.CTkFont(size=11),
            text_color="gray",
        ).pack(side="left", padx=10)

        hb_chains_row = ctk.CTkFrame(self.hb_options_frame, fg_color="transparent")
        hb_chains_row.pack(fill="x", pady=2)
        ctk.CTkLabel(hb_chains_row, text="MCMC Chains:", width=150, anchor="w").pack(side="left")
        self.hb_chains_var = ctk.StringVar(value="4")
        self.hb_chains_entry = ctk.CTkEntry(
            hb_chains_row, width=120, textvariable=self.hb_chains_var
        )
        self.hb_chains_entry.pack(side="left")

        hb_save_row = ctk.CTkFrame(self.hb_options_frame, fg_color="transparent")
        hb_save_row.pack(fill="x", pady=2)
        self.save_individual_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            hb_save_row,
            text="Save Individual-Level Utilities",
            variable=self.save_individual_var,
            font=ctk.CTkFont(size=14),
        ).pack(side="left")

        # Anchor item (common)
        anchor_row = ctk.CTkFrame(options_content, fg_color="transparent")
        anchor_row.pack(fill="x", pady=(5, 10))
        ctk.CTkLabel(anchor_row, text="Anchor Item:", width=150, anchor="w").pack(side="left")
        self.anchor_entry = ctk.CTkEntry(
            anchor_row, width=250, placeholder_text="Optional: item to highlight"
        )
        self.anchor_entry.pack(side="left")

        # Initialize method options visibility
        self.update_method_options()

        # Colors (inside Analysis Options, always customizable)
        colors_section = ctk.CTkFrame(options_content, fg_color="transparent")
        colors_section.pack(fill="x", pady=(10, 0))
        ctk.CTkLabel(colors_section, text="Chart Colors:", font=ctk.CTkFont(size=14)).pack(
            anchor="w", pady=(0, 5)
        )

        self.colors_container = ctk.CTkFrame(colors_section, fg_color="transparent")
        self.colors_container.pack(fill="x")

        self.color_buttons = []
        color_configs = [
            ("Positive Points:", "#f4b400"),
            # A distinct default for negative points so the chart visibly
            # differentiates positive and negative scores out of the box.
            # Pre-Phase-6 both defaulted to #f4b400, making the negative
            # control invisible. Users can still pick the same color.
            ("Negative Points:", "#5b9bd5"),
            ("Error Bars:", "#a1a1a1"),
            ("Zero Line:", "#ff0000"),
            ("Anchor Point:", "#a1a1a1"),
            ("Anchor Error:", "#a1a1a1"),
        ]
        for label, default in color_configs:
            cb = ColorButton(self.colors_container, label, default)
            cb.pack(fill="x", pady=2)
            self.color_buttons.append(cb)

        # Run section
        run_frame = ctk.CTkFrame(self.main_frame)
        run_frame.pack(fill="x", pady=(0, 15))

        progress_container = ctk.CTkFrame(run_frame, fg_color="transparent")
        progress_container.pack(fill="x", padx=15, pady=15)
        self.progress_label = ctk.CTkLabel(
            progress_container, text="Ready", font=ctk.CTkFont(size=13)
        )
        self.progress_label.pack(anchor="w")
        self.progress_bar = ctk.CTkProgressBar(progress_container, height=15)
        self.progress_bar.pack(fill="x", pady=(5, 10))
        self.progress_bar.set(0)

        btn_container = ctk.CTkFrame(run_frame, fg_color="transparent")
        btn_container.pack(pady=(0, 15))
        self.run_btn = ctk.CTkButton(
            btn_container,
            text="🚀 Run Analysis",
            font=ctk.CTkFont(size=16, weight="bold"),
            width=200,
            height=45,
            command=self.run_analysis,
        )
        self.run_btn.pack(side="left", padx=10)
        ctk.CTkButton(
            btn_container,
            text="📂 Open Results",
            font=ctk.CTkFont(size=14),
            width=150,
            height=45,
            fg_color="gray",
            command=self.open_results_folder,
        ).pack(side="left", padx=10)

        # Log
        log_frame = ctk.CTkFrame(self.main_frame)
        log_frame.pack(fill="both", expand=True, pady=(0, 10))
        ctk.CTkLabel(
            log_frame, text="📋 Analysis Log", font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", padx=15, pady=(15, 10))
        self.log_text = ctk.CTkTextbox(
            log_frame, height=200, font=ctk.CTkFont(family="Courier", size=11)
        )
        self.log_text.pack(fill="both", expand=True, padx=15, pady=(0, 15))

    def update_method_options(self):
        """Show/hide options based on selected analysis method."""
        method = self.analysis_method_var.get()

        if method == "count":
            self.count_options_frame.pack(fill="x", padx=0, pady=5)
            self.hb_options_frame.pack_forget()
            self.method_desc_label.configure(
                text="Transparent count-based scores: (Best - Worst) / Displays. Fast and highly correlated with HB results."
            )
        else:  # hb
            self.count_options_frame.pack_forget()
            self.hb_options_frame.pack(fill="x", padx=0, pady=5)
            self.method_desc_label.configure(
                text="Hierarchical Bayes with MCMC sampling. Provides individual-level utilities for follow-up analyses. Takes ~5-20 minutes."
            )

    def generate_example_maxdiff(self):
        """Show example MaxDiff data in a popup window."""
        example_data = [
            [
                "Response ID",
                "Attribute1",
                "Attribute2",
                "Attribute3",
                "Attribute4",
                "Most",
                "Least",
            ],
            ["R001", "Price", "Quality", "Brand", "Speed", "Quality", "Price"],
            ["R001", "Design", "Support", "Durability", "Features", "Features", "Support"],
            ["R001", "Price", "Brand", "Durability", "Speed", "Price", "Speed"],
            ["R002", "Quality", "Brand", "Support", "Features", "Quality", "Support"],
            ["R002", "Price", "Durability", "Speed", "Design", "Speed", "Price"],
            ["R002", "Design", "Quality", "Brand", "Features", "Quality", "Design"],
            ["R003", "Brand", "Speed", "Design", "Durability", "Design", "Brand"],
            ["R003", "Price", "Quality", "Support", "Features", "Quality", "Price"],
            ["R003", "Support", "Durability", "Speed", "Brand", "Support", "Brand"],
        ]

        self._show_example_window(
            "Example MaxDiff Data",
            example_data,
            "example_maxdiff.csv",
            "Each row = one task. Response ID repeats for multiple tasks per respondent.\n"
            "Attribute columns = items shown in that task. Most/Least = respondent's choices.",
        )

    def generate_example_segment(self):
        """Show example segment data in a popup window."""
        example_data = [
            ["Response ID", "Gender", "Age Group", "Region"],
            ["R001", "Female", "25-34", "North"],
            ["R002", "Male", "35-44", "South"],
            ["R003", "Female", "18-24", "North"],
            ["R004", "Male", "45-54", "East"],
            ["R005", "Female", "25-34", "West"],
            ["R006", "Non-binary", "35-44", "South"],
        ]

        self._show_example_window(
            "Example Segment Data",
            example_data,
            "example_segments.csv",
            "One row per respondent. Response ID must match your MaxDiff data.\n"
            "Add any segment columns you want to analyze (demographics, behaviors, etc.).",
        )

    def _show_example_window(self, title, data, default_filename, description):
        """Display example data in a popup with copy/save options."""
        window = ctk.CTkToplevel(self)
        window.title(title)
        window.geometry("700x450")
        window.transient(self)
        window.grab_set()

        # Description
        ctk.CTkLabel(
            window,
            text=description,
            font=ctk.CTkFont(size=12),
            text_color="gray",
            wraplength=650,
            justify="left",
        ).pack(padx=20, pady=(15, 10), anchor="w")

        # Table frame
        table_frame = ctk.CTkFrame(window)
        table_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Create treeview
        columns = data[0]
        tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=10)

        # Scrollbars
        y_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        x_scroll = ttk.Scrollbar(table_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        y_scroll.pack(side="right", fill="y")
        x_scroll.pack(side="bottom", fill="x")
        tree.pack(fill="both", expand=True)

        # Configure columns
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100, minwidth=60)

        # Add data rows
        for row in data[1:]:
            tree.insert("", "end", values=row)

        # Convert to CSV string
        csv_string = "\n".join([",".join(row) for row in data])

        # Buttons frame
        btn_frame = ctk.CTkFrame(window, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=(10, 20))

        def copy_to_clipboard():
            window.clipboard_clear()
            window.clipboard_append(csv_string)
            messagebox.showinfo(
                "Copied", "Example data copied to clipboard!\n\nPaste into Excel or a text editor."
            )

        def save_to_file():
            filepath = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                initialfilename=default_filename,
            )
            if filepath:
                with open(filepath, "w", newline="") as f:
                    f.write(csv_string)
                messagebox.showinfo("Saved", f"Example saved to:\n{filepath}")

        def save_to_current_dir():
            filepath = Path(default_filename)
            with open(filepath, "w", newline="") as f:
                f.write(csv_string)
            messagebox.showinfo("Saved", f"Example saved to:\n{filepath.absolute()}")

        ctk.CTkButton(
            btn_frame, text="📋 Copy to Clipboard", width=150, command=copy_to_clipboard
        ).pack(side="left", padx=5)

        #         ctk.CTkButton(
        #             btn_frame, text="💾 Save As...", width=120,
        #             command=save_to_file
        #         ).pack(side="left", padx=5)

        ctk.CTkButton(
            btn_frame,
            text="💾 Save Here",
            width=120,
            command=save_to_current_dir,
            fg_color="#2d8a4e",
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            btn_frame, text="Close", width=100, command=window.destroy, fg_color="gray"
        ).pack(side="right", padx=5)

    def load_and_preview_maxdiff(self):
        filename = filedialog.askopenfilename(
            title="Select MaxDiff Data",
            filetypes=[("CSV", "*.csv"), ("Excel", "*.xlsx"), ("All", "*.*")],
        )
        if not filename:
            return
        try:
            df = read_tabular_file(filename)
            self.maxdiff_file = filename
            detected_format, message = DataFormatDetector.detect_format(df)
            DataPreviewWindow(self, df, detected_format, message, self.set_maxdiff_data)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load:\n{e!s}")

    def set_maxdiff_data(self, df):
        self.maxdiff_df = df
        self.maxdiff_entry.configure(state="normal")
        self.maxdiff_entry.delete(0, "end")
        self.maxdiff_entry.insert(0, self.maxdiff_file or "Converted data")
        self.maxdiff_entry.configure(state="disabled")
        self.maxdiff_status.configure(text="✓ Ready", text_color="green")

        n_participants = df["Response ID"].nunique()
        attr_cols = [c for c in df.columns if c.startswith("Attribute")]
        items = [i for i in pd.unique(df[attr_cols].values.ravel()) if pd.notna(i)]

        self.data_info_label.configure(
            text=f"📊 {n_participants} participants, {len(df)} tasks, {len(items)} items, {len(attr_cols)} per task"
        )
        self.log(f"✓ Data loaded: {n_participants} participants, {len(items)} items")

    def browse_segment(self):
        filename = filedialog.askopenfilename(
            title="Select Segment Data", filetypes=[("CSV", "*.csv"), ("Excel", "*.xlsx")]
        )
        if filename:
            self.segment_file = filename
            self.segment_entry.configure(state="normal")
            self.segment_entry.delete(0, "end")
            self.segment_entry.insert(0, filename)
            self.segment_entry.configure(state="disabled")
            self.segment_status.configure(text="✓ Loaded", text_color="green")
            self.log(f"✓ Segment file: {filename}")

    def clear_segment(self):
        self.segment_file = None
        self.segment_entry.configure(state="normal")
        self.segment_entry.delete(0, "end")
        self.segment_entry.configure(state="disabled")
        self.segment_status.configure(text="Optional", text_color="gray")

    def toggle_ci_options(self):
        state = "normal" if self.include_ci_var.get() else "disabled"
        self.iterations_entry.configure(state=state)

    def log(self, msg):
        self.message_queue.put(("log", msg))

    def update_progress(self, val, msg=None):
        self.message_queue.put(("progress", (val, msg)))

    def check_queue(self):
        try:
            while True:
                msg_type, data = self.message_queue.get_nowait()
                if msg_type == "log":
                    self.log_text.insert("end", data + "\n")
                    self.log_text.see("end")
                elif msg_type == "progress":
                    self.progress_bar.set(data[0])
                    if data[1]:
                        self.progress_label.configure(text=data[1])
                elif msg_type == "complete":
                    self.run_btn.configure(state="normal", text="🚀 Run Analysis")
                    self.progress_label.configure(text="✅ Complete!")
                    messagebox.showinfo(
                        "Complete", "Analysis finished!\nResults in 'results' folder."
                    )
                elif msg_type == "error":
                    self.run_btn.configure(state="normal", text="🚀 Run Analysis")
                    self.progress_label.configure(text="❌ Error")
                    messagebox.showerror("Error", str(data))
        except queue.Empty:
            pass
        self.after(100, self.check_queue)

    def open_results_folder(self):
        import platform
        import subprocess

        p = Path("results")
        if not p.exists():
            messagebox.showwarning("Warning", "Run analysis first!")
            return

        # Always use list form for subprocess to avoid shell injection /
        # quoting issues when the working directory contains spaces or
        # other shell metacharacters.
        system = platform.system()
        path_str = str(p.absolute())
        if system == "Windows":
            subprocess.Popen(["explorer", path_str])
        elif system == "Darwin":
            subprocess.Popen(["open", path_str])
        else:
            subprocess.Popen(["xdg-open", path_str])

    def run_analysis(self):
        if self.maxdiff_df is None:
            messagebox.showwarning("Warning", "Load MaxDiff data first!")
            return

        analysis_method = self.analysis_method_var.get()

        # Validate settings based on method
        if analysis_method == "count":
            include_ci = self.include_ci_var.get()
            if include_ci:
                try:
                    n_iterations = int(self.iterations_var.get())
                    if n_iterations < 100:
                        raise ValueError("Iterations must be at least 100")
                except ValueError as e:
                    messagebox.showwarning("Warning", f"Enter valid iterations (min 100): {e}")
                    return
            else:
                n_iterations = 0
            hb_settings = None
        else:  # HB
            include_ci = True  # HB always has credible intervals
            n_iterations = 0
            try:
                hb_iter = int(self.hb_iterations_var.get())
                hb_chains = int(self.hb_chains_var.get())
                if hb_iter < 500:
                    raise ValueError("HB iterations must be at least 500")
                if hb_chains < 1:
                    raise ValueError("Must have at least 1 chain")
            except ValueError as e:
                messagebox.showwarning("Warning", f"Invalid HB settings: {e}")
                return

            hb_settings = {
                "iterations": hb_iter,
                "warmup": hb_iter // 2,
                "chains": hb_chains,
                "save_individual": self.save_individual_var.get(),
            }

        output_terms = (
            ("Most", "Least") if self.output_term_var.get() == "Most/Least" else ("Best", "Worst")
        )

        colors = {}
        keys = [
            "positive_color",
            "negative_color",
            "error_bar_color",
            "zero_line_color",
            "anchor_item_color",
            "anchor_item_error_color",
        ]
        for i, key in enumerate(keys):
            colors[key] = self.color_buttons[i].get_color()

        anchor_item = self.anchor_entry.get().strip() or None

        self.run_btn.configure(state="disabled", text="Running...")
        self.progress_bar.set(0)
        self.log_text.delete("1.0", "end")

        self.analysis_thread = threading.Thread(
            target=self.run_analysis_thread,
            args=(
                n_iterations,
                colors,
                anchor_item,
                output_terms,
                include_ci,
                analysis_method,
                hb_settings,
            ),
            daemon=True,
        )
        self.analysis_thread.start()

    def run_analysis_thread(
        self,
        n_iterations,
        colors,
        anchor_item,
        output_terms,
        include_ci,
        analysis_method,
        hb_settings,
    ):
        try:
            base_dir = Path("results")
            base_dir.mkdir(parents=True, exist_ok=True)

            maxdiff_df = self.maxdiff_df.copy()

            pos_col, neg_col = "Most", "Least"
            if "Best" in maxdiff_df.columns:
                pos_col, neg_col = "Best", "Worst"

            attribute_columns = [c for c in maxdiff_df.columns if c.startswith("Attribute")]

            self.log("Validating data...")
            check_errors(maxdiff_df, attribute_columns, pos_col, neg_col)

            unique_attributes = pd.unique(maxdiff_df[attribute_columns].values.ravel("K"))
            unique_attributes = unique_attributes[~pd.isnull(unique_attributes)]
            attr_to_index = {attr: i for i, attr in enumerate(unique_attributes)}

            overall_sample_size = maxdiff_df["Response ID"].nunique()
            self.log(f"✓ {len(unique_attributes)} items, {overall_sample_size} participants")
            self.log(f"Analysis method: {analysis_method.upper()}")

            # === DISPLAY STATISTICS ===
            self.log("\n" + "=" * 50)
            self.log("CALCULATING DISPLAY STATISTICS...")
            self.log("=" * 50)

            display_stats_df, balance_metrics = calculate_display_statistics(
                maxdiff_df, attribute_columns, pos_col, neg_col
            )

            # Log the display report
            display_report = format_display_report(display_stats_df, balance_metrics, output_terms)
            for line in display_report.split("\n"):
                self.log(line)

            # Save display statistics
            save_dataframe(
                display_stats_df, f"overall_display_statistics_n{overall_sample_size}", base_dir
            )

            # Save display balance plot
            balance_fig = plot_display_balance(
                display_stats_df,
                title=f"Display Balance Overview (n={overall_sample_size})",
                output_terms=output_terms,
            )
            save_plot(balance_fig, f"overall_display_balance_n{overall_sample_size}", base_dir)

            self.log("\n" + "=" * 50)

            # === MAIN ANALYSIS ===
            if analysis_method == "hb":
                # Hierarchical Bayes Analysis
                self.log("\nHierarchical Bayes Analysis")
                self.log(
                    f"  Iterations: {hb_settings['iterations']} (+ {hb_settings['warmup']} warmup)"
                )
                self.log(f"  Chains: {hb_settings['chains']}")

                hb_model = HierarchicalBayesMaxDiff(
                    n_iterations=hb_settings["iterations"],
                    n_warmup=hb_settings["warmup"],
                    n_chains=hb_settings["chains"],
                )

                def hb_progress(v):
                    self.update_progress(v * 0.7, f"HB MCMC: {int(v * 100)}%")

                overall_results = hb_model.fit(
                    maxdiff_df,
                    attribute_columns,
                    pos_col,
                    neg_col,
                    progress_callback=hb_progress,
                    log_callback=self.log,
                )
                result_key = "hb_utilities"

                # Save individual utilities if requested
                if hb_settings.get("save_individual", True):
                    self.log("Saving individual-level utilities...")
                    individual_utils = hb_model.get_individual_utilities()
                    save_dataframe(
                        individual_utils,
                        f"overall_individual_utilities_n{overall_sample_size}",
                        base_dir,
                        include_index=True,
                    )

                    # Also save preference shares
                    pref_shares = hb_model.get_preference_shares()
                    save_dataframe(
                        pref_shares, f"overall_preference_shares_n{overall_sample_size}", base_dir
                    )

            elif include_ci:
                # Count-based with bootstrap CI
                def progress_cb(v):
                    self.update_progress(v * 0.7, f"Bootstrap: {int(v * 100)}%")

                self.log(f"\nBootstrap analysis ({n_iterations:,} iterations)...")
                overall_results = bootstrap_analysis(
                    maxdiff_df,
                    attribute_columns,
                    unique_attributes,
                    attr_to_index,
                    pos_col,
                    neg_col,
                    n_iterations,
                    progress_cb,
                )
                result_key = "bootstrap_results"
            else:
                # Count-based without CI
                self.log("\nCalculating scores...")
                overall_results = calculate_scores_no_ci(
                    maxdiff_df, attribute_columns, pos_col, neg_col
                )
                result_key = "scores"
                self.update_progress(0.5, "Processing...")

            self.log("Calculating percentages...")
            overall_observed = calculate_observed_percentages(
                maxdiff_df, attribute_columns, pos_col, neg_col, output_terms
            )

            self.log("Calculating correlations...")
            overall_corr = calculate_correlation_matrix(
                maxdiff_df, attribute_columns, pos_col, neg_col
            )

            self.update_progress(0.75, "Saving results...")
            save_results(
                {result_key: overall_results, "observed_percentages": overall_observed},
                base_dir,
                sample_size=overall_sample_size,
                **colors,
                anchor_item=anchor_item,
                output_terms=output_terms,
                include_ci=include_ci,
            )

            save_dataframe(
                overall_corr,
                f"overall_correlation_matrix_n{overall_sample_size}",
                base_dir,
                include_index=True,
            )
            corr_fig = plot_correlation_matrix(
                overall_corr, f"Correlation Matrix (n={overall_sample_size})"
            )
            save_plot(corr_fig, f"overall_correlation_matrix_n{overall_sample_size}_plot", base_dir)

            # Segment analysis
            if self.segment_file:
                self.log("\nProcessing segments...")
                try:
                    segment_df = read_tabular_file(self.segment_file)

                    def seg_progress(v):
                        self.update_progress(0.75 + v * 0.2, f"Segments: {int(v * 100)}%")

                    segment_results = segment_maxdiff_analysis(
                        maxdiff_df,
                        segment_df,
                        pos_col,
                        neg_col,
                        output_terms,
                        n_iterations,
                        anchor_item,
                        include_ci,
                        seg_progress,
                        self.log,
                        analysis_method,
                        hb_settings,
                    )
                    process_segment_results(
                        segment_results, base_dir, colors, output_terms, anchor_item, include_ci
                    )
                except Exception as seg_error:
                    self.log(f"⚠️ Segment analysis error: {seg_error}")

            self.update_progress(1.0, "Complete!")
            self.log("\n" + "=" * 50)
            self.log("✅ ANALYSIS COMPLETE!")
            self.log(f"   Method: {analysis_method.upper()}")
            self.log(f"   Results saved to: {base_dir.absolute()}")
            self.log("=" * 50)
            self.message_queue.put(("complete", None))

        except Exception as e:
            self.log(f"\n❌ ERROR: {e!s}")
            import traceback

            self.log(traceback.format_exc())
            self.message_queue.put(("error", str(e)))


if __name__ == "__main__":
    app = MaxDiffGUI()
    app.mainloop()
