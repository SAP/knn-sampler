"""Keep test plotting independent of GUI toolkits before module collection."""

import matplotlib

matplotlib.use("Agg")
