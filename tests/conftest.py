"""Shared pytest setup."""

import matplotlib

matplotlib.use("Agg")  # draw plots without a display (CI, Colab terminals)
