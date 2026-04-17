#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="EEG Hybrid Model Dashboard", layout="wide")
st.title("EEG Motor Imagery: CNN vs LSTM vs CNN+LSTM+GNN")

default_results = Path("results").resolve()
results_dir = Path(st.text_input("Results directory", str(default_results))).resolve()

summary_path = results_dir / "summary.csv"
if not summary_path.exists():
    st.warning("summary.csv not found. Run train_compare.py first.")
    st.stop()

summary_df = pd.read_csv(summary_path)
st.subheader("Model Summary")
st.dataframe(summary_df, use_container_width=True)
st.bar_chart(summary_df.set_index("model")["test_acc"])

comparison_image = results_dir / "model_comparison.png"
if comparison_image.exists():
    st.image(str(comparison_image), caption="Test Accuracy Comparison")

model_names = summary_df["model"].tolist()
selected_model = st.selectbox("Inspect model", model_names)

history_path = results_dir / "{}_history.csv".format(selected_model)
if history_path.exists():
    hist_df = pd.read_csv(history_path)
    col1, col2 = st.columns(2)
    with col1:
        st.line_chart(hist_df.set_index("epoch")[["train_loss", "val_loss"]])
    with col2:
        st.line_chart(hist_df.set_index("epoch")[["train_acc", "val_acc"]])

cm_path = results_dir / "{}_confusion_matrix.png".format(selected_model)
curves_path = results_dir / "{}_curves.png".format(selected_model)
if cm_path.exists() and curves_path.exists():
    col3, col4 = st.columns(2)
    with col3:
        st.image(str(cm_path), caption="Confusion Matrix")
    with col4:
        st.image(str(curves_path), caption="Training Curves")

