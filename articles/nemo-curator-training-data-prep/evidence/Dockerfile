# Layer NeMo Curator 1.x + cuDF + fasttext on top of the existing NeMo
# Framework Megatron Backend container (already on disk for A1 + A2).
# Build:  docker build -t nemo-curator-spark:1.1 evidence/
#
# Why a derived image:
#   nemo-curator's GPU-accelerated dedup needs RAPIDS cuDF; the langid
#   filter needs fasttext. Both install cleanly on aarch64 from public
#   indexes — but each install takes 60-90s, so layering once and
#   tagging keeps subsequent runs in seconds, not minutes.
FROM nvcr.io/nvidia/nemo:26.04.00

# NeMo Framework's container ships a uv venv at /opt/venv that the default
# `python3` shebang points to — `pip` writes to system site-packages by
# default, which is the WRONG path. Use the venv's pip explicitly so torch
# / megatron-core / TE remain on Python's import path with cuDF + Curator
# layered alongside.
#
# pulp pin: cosmos-xenna's autoscaler uses pulp's pre-3.0 API
# (lowBound= kwarg); PuLP 3.0 and 4.0a removed it. Pin to 2.x.
RUN /opt/venv/bin/python3 -m pip install --no-cache-dir --extra-index-url=https://pypi.nvidia.com "cudf-cu13" \
 && /opt/venv/bin/python3 -m pip install --no-cache-dir nemo-curator fasttext "pulp<3"

# Also pre-pull the fasttext language-ID model so runs don't hit the
# network. lid.176.bin = 131MB.
RUN mkdir -p /opt/curator-models \
    && curl -fL -o /opt/curator-models/lid.176.bin \
         https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

ENV NEMO_CURATOR_LANG_ID_MODEL=/opt/curator-models/lid.176.bin
